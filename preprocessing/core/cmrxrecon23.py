"""
CMRxRecon2023 preprocessing pipeline: class ``CMRxRecon23Preprocessing``
reads the metadata CSV (auto-scanned from raw .mat files when missing),
loads .mat k-space (mat73/loadmat fallback), applies sampling masks and
coil-sensitivity estimation, and writes the processed h5 datasets.
Test-set samples inherit `norm`/`abs_max` from the matching fully-sampled
file via `_get_fs_file_path`. Entry point: preprocessing/run_cmrxrecon23.py
(``CMRxRecon23Preprocessing(...).run()``); CSV scanning lives in
preprocessing/core/scan_cmrxrecon_mat_csv.py.
"""

import logging
from pathlib import Path
from typing import Any

try:
    import cupy as cp
except ImportError:  # importing cupy fails if no cuda is available
    cp = None
import h5py
import mat73
import numpy as np
import pandas as pd
from scipy.io import loadmat

from helpers.mri_basic import bart

from helpers import mri_basic as mri
from helpers.mri_basic.ndarray import ndarray
from helpers.save import save_movie


class CMRxRecon23Preprocessing:
    # Minimal CSV columns for CMRxRecon mat preprocessing.
    REQ_CSV_COLS = [
        "mat_path",
        "split",
        "smp",
        "acc",
        "patient",
        "viw",
    ]

    def __init__(
        self,
        csv_file: Path,
        data_source: str,
        raw_dir: Path,
        target_dir: Path,
        coil_sens_dir: Path,
        cs_dir: Path | None = None,
        csv_query: str | None = None,
        csv_idx_range: list[int | None] = [0, None],
        accelerations: list[int] | None = None,
        mask_types: list[str] | None = None,
        coil_sens_backend: str = "sigpy",
        n_maps: int = 1,
        try_cuda: bool = True,
        seed: int = 42,
    ):
        assert csv_file.is_file(), f"{csv_file} is not a file."
        self.csv_file = csv_file
        self.data_source = data_source
        self.raw_dir = raw_dir
        self.target_dir = target_dir
        self.coil_sens_dir = coil_sens_dir
        self.cs_dir = cs_dir
        self.accelerations = [] if accelerations is None else accelerations
        self.mask_types = [] if mask_types is None else mask_types
        self.coil_sens_backend = coil_sens_backend
        self.n_maps = n_maps
        self.device = "cpu"
        if try_cuda and cp is not None and cp.cuda.is_available():
            self.device = "cuda"
        self.rng = np.random.RandomState(seed)

        self.df_full = pd.read_csv(csv_file)
        missing_cols = [c for c in self.REQ_CSV_COLS if c not in self.df_full.columns]
        assert len(missing_cols) == 0, f"Dataframe misses the following cols: {missing_cols}"

        self.df = self.df_full.loc[csv_idx_range[0] : csv_idx_range[1]].copy()
        if csv_query:
            self.df = self.df.query(csv_query)

        # # Test-only workflow for current run.
        # split_series = self.df["split"].fillna("").astype(str).str.lower()
        # self.df = self.df.loc[split_series == "test"].reset_index(drop=True)
        # logging.info("Selected %d test rows for processing", len(self.df))

        # Infer available test acceleration factors from CSV.
        test_acc_series = pd.to_numeric(
            self.df_full.loc[self.df_full["split"].astype(str).str.lower() == "test", "acc"],
            errors="coerce",
        )
        inferred_accs = sorted({int(v) for v in test_acc_series.dropna().tolist() if int(v) > 1})

        if accelerations is None or len(accelerations) == 0:
            self.accelerations = inferred_accs
        else:
            self.accelerations = sorted({int(a) for a in accelerations if int(a) > 1})

        logging.info(
            "Detected %d test acceleration factors from CSV: %s",
            len(inferred_accs),
            inferred_accs,
        )

        # # Stage-1 workflow: only process training data.
        # split_series = self.df["split"].fillna("").astype(str).str.lower()
        # self.df = self.df.loc[split_series == "train"].reset_index(drop=True)
        # logging.info(f"Selected {len(self.df)} training rows from {len(self.df_full)} CSV rows")

    def load_dataset(self, filename: Path) -> np.ndarray:
        try:
            try:
                data = mat73.loadmat(filename)
            except Exception:
                data = loadmat(filename)
                data = {key: value for key, value in data.items() if not key.startswith("__")}
        except Exception as exc:
            raise ValueError(f"Unsupported file format or error loading file: {filename}") from exc

        key = list(data.keys())[0]
        k_data = data[key]

        if k_data.ndim == 4:
            k_data = k_data[..., np.newaxis]
        assert k_data.ndim == 5, f"Invalid kdata shape: {k_data.shape}"

        # [kx, ky, coil, slice, frame] -> [kx, ky, kz=1, coil, frame, slice]
        k_data = k_data[..., np.newaxis].transpose(0, 1, 5, 2, 4, 3)
        return k_data

    @staticmethod
    def sample_base_name(mat_path: str) -> str:
        """Build the output base name used everywhere (h5, csm, gif).

        Drops the acquisition-type directory ('FullSample' or 'AccFactorXX'),
        e.g. 'TestSet/AccFactor08/P001/cine_sax.mat' -> 'TestSet_P001_cine_sax',
        so outputs match the released processed_cmrxrecon23 dataset naming.
        """
        parts = str(mat_path).replace("\\", "/").removesuffix(".mat").split("/")
        return "_".join(p for p in parts if p != "FullSample" and not p.startswith("AccFactor"))

    def _get_fs_file_path(self, sample_info: dict[str, Any], slice_idx: int) -> Path:
        """Get the corresponding fs output file path for a test pse sample."""
        fs_output_name = f"{self.sample_base_name(str(sample_info.get('mat_path', '')))}_slice{slice_idx:02d}"
        return self.target_dir / f"{self.data_source}_test_fs" / f"{fs_output_name}.h5"


    def get_sampling_patterns(self, sample_info: dict[str, Any]) -> list[mri.Sampling]:
        """Get sampling patterns based on data split."""
        split = str(sample_info.get("split", "")).lower()
        smp = str(sample_info.get("smp", "")).lower()
        acc_val = sample_info.get("acc", 1)
        try:
            acc = int(acc_val)
        except (TypeError, ValueError):
            acc = 1

        if not smp:
            mat_path = str(sample_info.get("mat_path", "")).lower()
            if "accfactor" in mat_path:
                smp = "pse"
            elif "fullsample" in mat_path:
                smp = "fs"
        
        if split == "test":
            # Use CSV smp directly: fs for FullSample, pse for AccFactorxx.
            if smp == "pse":
                return [mri.Sampling(name=f"pse_{max(1, acc):02d}", acceleration=max(1, acc))]
            return [mri.Sampling(name="fs", acceleration=1)]
        else:
            # Training and validation: only fully-sampled
            return [mri.Sampling(name="fs", acceleration=1)]

    def mask_kspace(
        self,
        k_data: ndarray,
        sampling: mri.Sampling,
        pad_left: int = 0,
        pad_right: int = 0,
    ) -> tuple[ndarray, ndarray]:
        xp = cp.get_array_module(k_data) if cp is not None else np

        if sampling.name == "fs":
            mask = xp.ones((k_data.shape[1], k_data.shape[3]), dtype=xp.uint8)  # [ky, frame]
            kdata_masked = xp.copy(k_data)  # [kx, ky, coil, frame, slice]

        elif sampling.name.startswith("pse"):
            mask = (abs(xp.mean(xp.abs(k_data), axis=2)) > 0).astype(xp.uint8)  # [kx, ky, frame, slice]
            assert xp.all(mask.all(axis=3) | (1 - mask).all(axis=3)), "Mask is not equal along slice dim"
            mask = mask[mask.shape[0] // 2, :, :, 0]  # [ky, frame]

            mask_unpadded = mask[max(0, pad_left) : mask.shape[0] - max(0, pad_right), :]
            sampling.acceleration = int(mask_unpadded.size / mask_unpadded.sum())
            logging.info(f"Acceleration rate: {sampling.acceleration:.2f}")

            mask = xp.ones_like(mask)
            mask[max(0, pad_left) : mask.shape[0] - max(0, pad_right), :] = mask_unpadded
            kdata_masked = xp.copy(k_data)

        else:
            num_cols = k_data.shape[1] - max(0, pad_right) - max(0, pad_left)
            assert sampling.mask_type is not None and sampling.acceleration is not None
            mask_unpadded = mri.get_mask(
                k_data.shape[3],
                num_cols,
                sampling.mask_type,
                sampling.acceleration,
                padding_left=pad_left,
                padding_right=pad_right,
                rng=self.rng,
            )
            if cp is not None and isinstance(k_data, cp.ndarray):
                mask_unpadded = cp.asarray(mask_unpadded)
            mask_unpadded = mask_unpadded.T  # [ky, frame]

            acc_eff = mask_unpadded.size / mask_unpadded.sum()
            logging.info(f"Effective Acceleration Rate: {acc_eff:5.2f}")

            mask = xp.ones((k_data.shape[1], k_data.shape[3]), dtype=mask_unpadded.dtype)  # [ky, frame]
            mask[max(0, pad_left) : mask.shape[0] - max(0, pad_right), :] = mask_unpadded
            kdata_masked = k_data * mask[None, :, None, :, None]

        return kdata_masked, mask

    def get_coil_sensitivities(self, kdata_masked: ndarray, output_name: str) -> ndarray:
        coil_sens_fname = self.coil_sens_dir / self.data_source / output_name / "coil_sens_avg.h5"

        try:
            coil_sens = mri.coil_sensitivities.read(
                coil_sens_fname,
                nmaps=self.n_maps,
                layout="bart",
            )
            if cp is not None and isinstance(kdata_masked, cp.ndarray):
                coil_sens = cp.asarray(coil_sens)
            logging.info("Loaded previously computed coil sensitivity maps")

        except FileNotFoundError:
            logging.info(f"Estimating coil sensitivity maps using {self.coil_sens_backend}")
            bart.clear_stdouterr()
            coil_sens = mri.coil_sensitivities.estimate(
                kdata_masked,
                self.coil_sens_backend,
                fname=coil_sens_fname,
                nmaps=self.n_maps,
                write_png=False,
            )
            if self.coil_sens_backend == "bart":
                for line in bart.bart.stdout.splitlines():
                    logging.info("Bart: " + line)
                for line in bart.bart.stderr.splitlines():
                    logging.warning("Bart: " + line)

        return coil_sens

    def save_data(
        self,
        output_name: str,
        split: str,
        sampling: mri.Sampling,
        kdata_masked: ndarray,
        mask: ndarray,
        recons: dict[str, ndarray],
        sample_info: dict[str, Any],
        slice_idx: int,
        fs_attrs: dict[str, Any] | None = None,
    ):
        partition = f"test_{sampling.name}" if split == "test" else split
        out_file = self.target_dir / f"{self.data_source}_{partition}" / f"{output_name}.h5"
        out_file.parent.mkdir(exist_ok=True, parents=True)

        logging.info(f"Writing file {out_file}")

        with h5py.File(out_file, "w") as hf:
            kspace = kdata_masked.astype(np.complex64)
            if cp is not None:
                kspace = cp.asnumpy(kspace)
            hf.create_dataset("kspace", data=kspace)

            for key, recon in recons.items():
                if key not in ["rss", "weighted"]:
                    continue
                dtype = np.complex64 if np.iscomplexobj(recon) else np.float32
                recon = recon.astype(dtype)
                if cp is not None:
                    recon = cp.asnumpy(recon)
                hf.create_dataset(f"reconstruction_{key}", data=recon)

            if split == "test":
                mask_ = mask.astype(np.uint8)
                if cp is not None:
                    mask_ = cp.asnumpy(mask_)
                hf.create_dataset("mask", data=mask_)
                hf.attrs["acceleration"] = sampling.acceleration
                if sampling.mask_type is not None:
                    hf.attrs["mask_type"] = sampling.mask_type

            if "weighted" in recons:
                hf.attrs["norm"] = np.linalg.norm(np.abs(recons["weighted"]))
                hf.attrs["abs_max"] = np.max(np.abs(recons["weighted"]))
            elif fs_attrs is not None:
                # For test pse, use norm and abs_max from corresponding fs version.
                if "norm" in fs_attrs:
                    hf.attrs["norm"] = fs_attrs["norm"]
                if "abs_max" in fs_attrs:
                    hf.attrs["abs_max"] = fs_attrs["abs_max"]

            hf.attrs["patient_id"] = str(sample_info.get("patient", ""))
            hf.attrs["view"] = str(sample_info.get("viw", ""))
            hf.attrs["slice"] = int(slice_idx)
            hf.attrs["split"] = str(split)
            hf.attrs["mat_path"] = str(sample_info.get("mat_path", ""))

    def process_slice_with_sampling(
        self,
        k_data: np.ndarray,
        sampling: mri.Sampling,
        output_name: str,
        recon_phase_size: int,
        slice_idx: int,
        sample_info: dict[str, Any],
    ):
        split = str(sample_info.get("split", "")).lower()
        kdata_masked, mask = self.mask_kspace(k_data, sampling)
        
        # For test pse, only save kspace+mask; reconstruction comes from fs.
        recons = {}
        if not (split == "test" and sampling.name.startswith("pse")):
            coil_sens = self.get_coil_sensitivities(kdata_masked, output_name)
            
            logging.info("Computing reference reconstructions")
            recons["rss"] = mri.rss(k_data, fft_axes=(0, 1), coil_axis=2)  # [x, y, frame, slice]
            recons["weighted"] = mri.sensitivity_weighted(
                k_data,
                coil_sens[:, :, :, 0],
                fft_axes=(0, 1),
                coil_axis=2,
                frame_axis=3,
            )

        if sampling.name != "fs" and self.cs_dir is not None:
            logging.info("Computing CS reconstruction")
            bart.clear_stdouterr()
            recons["cs"] = mri.pics(
                kdata_masked,
                mri.get_pics_regularizer(sampling),
                coil_sens=coil_sens,
                coil_sens_nmaps=1,
                fname=self.cs_dir / f"{self.data_source}_{sampling.name}" / output_name / "reconstruction.h5",
                phase_crop=recon_phase_size,
                use_gpu=("cuda" in self.device),
                load_if_exists=True,
            )

        mask = mask.T  # [frame, ky]
        logging.info(f"New mask shape: {mask.shape} (frame, ky)")
        
        if len(recons) > 0:
            kdata_masked = kdata_masked.transpose(4, 2, 3, 0, 1)  # [slice, coil, frame, kx, ky]
            logging.info(f"New k-space shape: {kdata_masked.shape} (slice, coil, frame, kx, ky)")
            for key in recons:
                recons[key] = recons[key].transpose(3, 2, 0, 1)  # [slice, frame, x, y]
                logging.info(f"New recon shape ({key}): {recons[key].shape} (slice, frame, x, y)")

            logging.info("Cropping phase oversampling")
            for key in recons:
                recons[key] = mri.utils.center_crop(recons[key], 3, recon_phase_size)
                logging.info(f"New recon shape ({key}): {recons[key].shape} (slice, frame, x, y)")
        else:
            # For test pse, just transpose k-space
            kdata_masked = kdata_masked.transpose(4, 2, 3, 0, 1)  # [slice, coil, frame, kx, ky]
            logging.info(f"New k-space shape: {kdata_masked.shape} (slice, coil, frame, kx, ky)")

        if sampling.name == "fs" and "weighted" in recons:
            recon = recons["weighted"]
            gif_dir = self.target_dir / f"{self.data_source}_recons"
            gif_dir.mkdir(exist_ok=True, parents=True)
            if cp is not None:
                recon = cp.asnumpy(recon)
            save_movie(recon[0], gif_dir / output_name, clip=True, equalize_histogram=False, apng=False)

        if split in ("train", "val") and sampling.name != "fs":
            return

        # For test pse, load fs attrs before saving.
        fs_attrs = None
        if split == "test" and sampling.name.startswith("pse"):
            fs_file = self._get_fs_file_path(sample_info, slice_idx)
            if fs_file.exists():
                try:
                    with h5py.File(fs_file, "r") as hf:
                        fs_attrs = dict(hf.attrs)
                        logging.info(f"Loaded attrs from fs file: {fs_file}")
                except Exception as exc:
                    logging.warning(f"Failed to load fs attrs: {exc}")
        
        self.save_data(output_name, split, sampling, kdata_masked, mask, recons, sample_info, slice_idx, fs_attrs=fs_attrs)

    def process_slice(self, k_data: np.ndarray, slc_idx: int, output_name: str, sample_info: dict[str, Any]):
        sampling_patterns = self.get_sampling_patterns(sample_info)

        logging.info(f"K-space shape: {k_data.shape} (kx, ky, kz, coil, frame, slice)")

        logging.info("Removing readout oversampling")
        k_data = mri.utils.crop_readout_oversampling(k_data)

        assert k_data.shape[2] == 1, "kz must be 1"
        k_data = k_data.squeeze(2)  # [kx, ky, coil, frame, slice]
        logging.info(f"K-space shape: {k_data.shape} (kx, ky, coil, frame, slice)")

        recon_phase_size = k_data.shape[1]

        for sampling in sampling_patterns:
            logging.info(f"-- Processing sampling '{sampling.name}' on slice {slc_idx + 1} --")
            self.process_slice_with_sampling(
                k_data,
                sampling,
                output_name,
                recon_phase_size,
                slc_idx,
                sample_info,
            )

    def process_sample(self, sample_info: dict[str, Any]):
        mat_path_rel = Path(str(sample_info["mat_path"]))
        mat_file = self.raw_dir / mat_path_rel
        if not mat_file.exists():
            raise FileNotFoundError(f"MAT file not found: {mat_file}")

        k_data = self.load_dataset(mat_file)

        base_name = self.sample_base_name(str(mat_path_rel))
        for slc_idx in range(k_data.shape[-1]):
            logging.info(f"---- Processing slice {slc_idx + 1} / {k_data.shape[-1]} ----")
            output_name = f"{base_name}_slice{slc_idx:02d}"
            self.process_slice(k_data[..., slc_idx, None], slc_idx, output_name, sample_info)

    def run(self):
        if len(self.df) == 0:
            logging.warning("No samples selected for processing.")
            return

        for count, (_, row) in enumerate(self.df.iterrows(), start=1):
            sample_info = row.to_dict()
            logging.info("-" * 50)
            logging.info("")
            logging.info("")
            logging.info(f"Processing {sample_info.get('mat_path')} ({count} / {len(self.df)})")
            logging.info("")
            logging.info("")
            self.process_sample(sample_info)
