# Data: building datasets & formats

Read `environment.md` first — `data_root` is where raw and preprocessed data
live. Dataset building is self-contained under `preprocessing/`: first-layer
`run_*.py` files are the commands, `core/` holds the implementations.

## Layout the code expects

```
<data_root>/OCMR/orig                 raw OCMR source files (.h5, ISMRMRD)
<data_root>/OCMR/processed_ocmr/      built: ocmr_train/, ocmr_val/, ocmr_test_gro_*/  + csm/
<data_root>/CMRxRecon/orig/            raw CMRxRecon2023 source (.mat: TrainingSet/ValidationSet/TestSet)
<data_root>/CMRxRecon/processed_cmrxrecon23/   built: cmrxrecon23_{train,val,test_fs,test_pse_*}/ + csm/
```

Relative `data_path` values in configs resolve against `YHAO_DATA` if set,
otherwise against the working directory — put absolute paths in
`environment.md` to avoid ambiguity.

## Build 1 — OCMR

```bash
python preprocessing/run_ocmr.py \
  --raw_dir <data_root>/OCMR/orig \
  --target_dir <data_root>/OCMR/processed_ocmr \
  --coil_sens_dir <data_root>/OCMR/processed_ocmr/csm \
  --csv_query 'smp=="fs"' \
  --accelerations 6 8 10 --mask_types gro
```

- Metadata CSV: `preprocessing/data_attributes/ocmr.csv` (ships with the repo; override with `--attributes_csv`). If raw files are missing, the loader downloads them from the public OCMR S3 bucket.
- Output naming: `<target>/<data_source>_<split>/<stem>_sliceNN.h5`, test splits additionally carry the protocol: `ocmr_test_gro_06` etc.
- Coil sensitivities: `<coil_sens_dir>/<data_source>/<stem>_sliceNN/coil_sens_avg.h5`.
- `--coil_sens_backend {sigpy,bart}` — `bart` needs a local BART install.

## Build 2 — CMRxRecon2023

```bash
python preprocessing/run_cmrxrecon23.py \
  --raw_dir <data_root>/CMRxRecon/orig \
  --target_dir <data_root>/CMRxRecon/processed_cmrxrecon23 \
  --coil_sens_dir <data_root>/CMRxRecon/processed_cmrxrecon23/csm
```

- The metadata CSV (`preprocessing/data_attributes/cmrxrecon23cine.csv`) is
  auto-scanned from raw `.mat` files when missing (`core/scan_cmrxrecon_mat_csv.py`).
- Output naming drops the acquisition-type directory: `TestSet/AccFactor08/P001/cine_sax.mat`
  → `TestSet_P001_cine_sax_slice00.h5` (never `FullSample_` / `AccFactor08_`).
  Test partitions are `cmrxrecon23_test_fs` / `cmrxrecon23_test_pse_08` etc.
- Test-set samples inherit `norm`/`abs_max` attributes from the matching
  fully-sampled file, so build `fs` before `pse` (the CSV already sorts it that way).

## Verifying a build

Each h5 holds: `kspace` (complex64), `mask`, `reconstruction_weighted` /
`reconstruction_rss`, plus attributes (`split`, `norm`, `abs_max`, `view`,
`slice`, …). A rebuilt dataset is correct when file names **and** per-file
checksums match the reference dataset:

```python
import h5py, hashlib
with h5py.File(path) as f:           # keys + dtypes
    print(sorted(f.keys()), f['kspace'].shape, f.attrs['norm'])
```

## Consuming the data

- `SliceDataset` (`helpers/data/mri_data.py`): fastMRI-style loader over the per-slice h5 files, plus coil maps from `coil_sens_path`.
- `VarNetDataTransform` (`helpers/data/transforms.py`): returns
  `(masked_kspace, mask, target, fname, slice_num, max_value, sens_maps, num_slc)`;
  `use_seed: false` for train, `true` for val/test.
- `FastMriDataModule` (`pl_modules/data_module.py`): builds the loaders, uses
  `ClusteredBatchSampler` for `batch_size > 1` and `VolumeSampler` under DDP.

Batch shapes:

| tensor | shape |
|--------|-------|
| k-space | `[B, coils, T, H, W, 2]` |
| csm | `[B, maps, coils, T, H, W, 2]` |
| mask | `[B, 1, T, 1, W, 1]` |

`num_slc = -1` marks a single-slice volume (no temporal axis).

## Mask selection

Chosen by `class_path` in the data yaml — there is no string-dispatch helper:

`helpers.data.subsample.{GROMaskFunc, RandomMaskFunc, RandomACSMaskFunc, VISTAMaskFunc, CAVAMaskFunc,
EquispacedMaskFunc, EquispacedACSMaskFunc, StaticEquispacedACSMaskFunc}`.
`accelerations: [6,7,8,9,10]` samples one acceleration per case.

## Adding your own dataset

1. Write a `preprocessing/run_<name>.py` entry + `preprocessing/core/<name>.py` implementation (model the OCMR one for ISMRMRD input, CMRxRecon for `.mat`), and a metadata CSV under `preprocessing/data_attributes/`.
2. Keep `helpers/` generic — dataset-specific logic belongs in `preprocessing/`.
3. Emit the h5 keys listed above; name files `<split>_<patient>_<view>_sliceNN.h5`.
4. Add a `configs/data/<name>.yaml`, then run one sample and diff it against a hand-built reference before scaling up.
