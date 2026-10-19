#!/usr/bin/env python3
"""Generate the CMRxRecon2023 processed h5 dataset from raw .mat k-space.

Pipeline: (optional) scan raw .mat files -> CSV metadata -> per-slice h5 samples
(kspace, mask, reference reconstructions, attributes).

Output naming (preprocessing/core/cmrxrecon23.py: `sample_base_name`,
`save_data`, `process_sample`):
    <target_dir>/<data_source>_<partition>/<base>_sliceNN.h5
    base = the raw .mat path joined by '_', with acquisition-type folders
    dropped: 'TestSet/AccFactor08/P001/cine_sax.mat' -> 'TestSet_P001_cine_sax'
    (neither `FullSample/` nor `AccFactorXX/` appears in the name); NN is the
    zero-padded 2-digit slice index.
    partition = 'test_<sampling.name>' for the test split (e.g.
    `cmrxrecon23_test_fs`, `cmrxrecon23_test_pse_08`) and '<split>' otherwise.
Coil sensitivities:
    <coil_sens_dir>/<data_source>/<base>_sliceNN/coil_sens_avg.h5
Metadata CSV: `preprocessing/data_attributes/cmrxrecon23cine.csv`, scanned
automatically from `--raw_dir` when missing.

Example:
    python preprocessing/run_cmrxrecon23.py \\
        --raw_dir data/CMRxRecon/orig \\
        --target_dir data/CMRxRecon/processed_cmrxrecon23 \\
        --coil_sens_dir data/CMRxRecon/processed_cmrxrecon23/csm
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# make `helpers` importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.core.cmrxrecon23 import CMRxRecon23Preprocessing
from preprocessing.core.scan_cmrxrecon_mat_csv import scan_cmrxrecon_to_csv


logging.basicConfig(
    handlers=[logging.StreamHandler()],
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s|%(message)s",
    datefmt="%Y-%m-%d %H-%M-%S",
)


def run_processing_cmrxrecon23(
    csv_file: Path,
    raw_dir: Path,
    target_dir: Path,
    coil_sens_dir: Path,
    data_source: str = "cmrxrecon23",
    cs_dir: Path | None = None,
    csv_query: str | None = None,
    start_idx: int = 0,
    end_idx: int | None = None,
    accelerations: list[int] | None = None,
    mask_types: list[str] | None = None,
    coil_sens_backend: str = "sigpy",
    n_maps: int = 1,
    try_cuda: bool = False,
) -> None:
    """Scan (if needed), then run CMRxRecon23 preprocessing."""
    if not csv_file.is_file():
        logging.info("CSV not found, scanning MAT files to create it: %s", csv_file)
        n_rows = scan_cmrxrecon_to_csv(raw_dir, csv_file)
        logging.info("Generated CSV with %d rows at: %s", n_rows, csv_file)

    preprocessing = CMRxRecon23Preprocessing(
        csv_file=csv_file,
        data_source=data_source,
        raw_dir=raw_dir,
        target_dir=target_dir,
        coil_sens_dir=coil_sens_dir,
        cs_dir=cs_dir,
        csv_query=csv_query,
        csv_idx_range=[start_idx, end_idx],
        accelerations=accelerations,
        mask_types=mask_types,
        coil_sens_backend=coil_sens_backend,
        n_maps=n_maps,
        try_cuda=try_cuda,
    )
    preprocessing.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CMRxRecon23 preprocessing from CSV")
    parser.add_argument("--csv_file", type=Path, default=None,
                        help="Metadata CSV; defaults to preprocessing/data_attributes/cmrxrecon23cine.csv "
                             "(auto-scanned from --raw_dir if missing)")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/CMRxRecon/orig"),
                        help="Raw dataset root containing TrainingSet/ValidationSet/TestSet")
    parser.add_argument("--target_dir", type=Path, default=Path("data/CMRxRecon/processed_cmrxrecon23"),
                        help="Output directory for processed h5 files")
    parser.add_argument("--coil_sens_dir", type=Path, default=Path("data/CMRxRecon/processed_cmrxrecon23/csm"),
                        help="Directory to store/load coil sensitivity maps")
    parser.add_argument("--data_source", type=str, default="cmrxrecon23")
    parser.add_argument("--cs_dir", type=Path)
    parser.add_argument("--csv_query", type=str, help="Pandas query to filter the CSV")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int)
    parser.add_argument("--accelerations", nargs="+", type=int, default=[6])
    parser.add_argument("--mask_types", nargs="+", type=str, default=["gro"],
                        choices=["random", "random_acs", "vista", "gro", "cava", "equispaced", "equispaced_acs"])
    parser.add_argument("--coil_sens_backend", type=str, choices=["sigpy", "bart"], default="bart")
    parser.add_argument("--n_maps", type=int, default=1)
    parser.add_argument("--try_cuda", action="store_true")
    args = parser.parse_args()

    logging.info("Arguments:")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")

    if args.csv_file is None:
        args.csv_file = Path(__file__).resolve().parent / "data_attributes" / "cmrxrecon23cine.csv"

    run_processing_cmrxrecon23(
        csv_file=args.csv_file,
        raw_dir=args.raw_dir,
        target_dir=args.target_dir,
        coil_sens_dir=args.coil_sens_dir,
        data_source=args.data_source,
        cs_dir=args.cs_dir,
        csv_query=args.csv_query,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        accelerations=args.accelerations,
        mask_types=args.mask_types,
        coil_sens_backend=args.coil_sens_backend,
        n_maps=args.n_maps,
        try_cuda=args.try_cuda,
    )
