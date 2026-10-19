#!/usr/bin/env python3
"""Build the CMRxRecon metadata CSV by scanning raw .mat files.

Walks `root_dir` recursively, extracts split/patient/view/sampling-type/
acceleration from each path, reads the k-space shape (v7.3 files via
h5py, classic files via scipy.io.loadmat), and writes one CSV row per
.mat. Rows are sorted so fully-sampled files come first --
preprocessing/core/cmrxrecon23.py needs those to inherit `norm` and
`abs_max` for test-set samples.

Called from preprocessing/run_cmrxrecon23.py when the CSV is
missing; also usable standalone: `python preprocessing/core/scan_cmrxrecon_mat_csv.py --raw_dir ... --out_csv ...`
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mat73
from scipy.io import loadmat
from tqdm import tqdm
import h5py

SPLIT_MAP: Dict[str, str] = {
    "trainingset": "train",
    "validationset": "val",
    "testset": "test",
    "train": "train",
    "val": "val",
    "test": "test",
}

def load_shape(filename):
    """
    Load the shape of a .mat file.
    """
    try:
        data = h5py.File(filename, 'r')
    except:
        raise ValueError(f"Unsupported file format or error loading file: {filename}")
    key = list(data.keys())[0]
    shape = data[key].shape
    return shape


def iter_mat_files(root_dir: Path) -> Iterable[Path]:
    """Yield .mat files recursively with split-specific filtering rules."""
    for p in sorted(q for q in root_dir.rglob("*.mat") if q.is_file()):
        parts = [part.lower() for part in p.relative_to(root_dir).parts]
        include = True
        for idx, part in enumerate(parts[:-1]):
            if part in {"trainingset", "validationset"}:
                include = idx + 1 < len(parts) and parts[idx + 1] == "fullsample"
                break
            if part == "testset":
                in_fullsample = "fullsample" in parts
                if in_fullsample:
                    include = True
                else:
                    # For TestSet/AccFactorxx, skip undersampling mask files only.
                    include = "_mask" not in p.stem.lower()
                break
        if include:
            yield p


def parse_row_info(mat_path: Path, root_dir: Path) -> Tuple[Optional[str], str, str, str, Optional[int]]:
    """
    Parse split/patient/viw/smp/acc from a mat file path.

    Example:
    TrainingSet/FullSample/P001/cine_lax.mat -> (train, P001, lax, fs, 1)
    TestSet/AccFactor10/P001/cine_lax.mat -> (test, P001, lax, pse, 10)
    """
    relative = mat_path.relative_to(root_dir)
    parts = list(relative.parts)

    split = None
    split_idx = -1
    for idx, part in enumerate(parts[:-1]):
        mapped = SPLIT_MAP.get(part.lower())
        if mapped is not None:
            split = mapped
            split_idx = idx
            break

    patient = ""
    if split_idx >= 0:
        patient_idx = split_idx + 1
        # Skip smp layer (fullsample or AccFactorXX)
        if patient_idx < len(parts) - 1:
            part_lower = parts[patient_idx].lower()
            if part_lower == "fullsample" or part_lower.startswith("accfactor"):
                patient_idx += 1
        if patient_idx < len(parts) - 1:
            patient = parts[patient_idx]

    stem = mat_path.stem
    stem_parts = stem.split("_")
    viw = stem_parts[-1] if len(stem_parts) > 1 else stem

    parts_lower = [p.lower() for p in parts]
    in_fullsample = "fullsample" in parts_lower
    smp = "fs" if in_fullsample else ("pse" if split == "test" else "")
    acc: Optional[int] = 1 if smp == "fs" else None

    if split == "test" and smp == "pse":
        for p in parts_lower:
            if p.startswith("accfactor"):
                suffix = p[len("accfactor") :]
                if suffix.isdigit():
                    acc = int(suffix)
                break

    return split, patient, viw, smp, acc


def shape_to_columns(shape: Tuple[int, ...]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Map shape to (kx, ky, coils, slices, frames)."""
    vals: List[Optional[int]] = [None, None, None, None, None]
    for i in range(min(5, len(shape))):
        vals[i] = int(shape[i])
    return vals[0], vals[1], vals[2], vals[3], vals[4]


def scan_cmrxrecon_to_csv(root_dir: Path, out_csv: Path) -> int:
    """Scan CMRxRecon .mat files under root_dir and write metadata CSV to out_csv."""
    root_dir = root_dir.resolve()
    out_csv = out_csv.resolve()

    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {root_dir}")

    rows = []
    mat_files = list(iter_mat_files(root_dir))
    for mat_path in tqdm(mat_files, desc="Scanning .mat", unit="file"):
        split, patient, viw, smp, acc = parse_row_info(mat_path, root_dir)
        if split is None:
            print(f"[WARN] Skip (cannot parse split): {mat_path}")
            continue

        try:
            shape = tuple(load_shape(str(mat_path)))
        except Exception as exc:
            print(f"[WARN] Skip (load_shape failed): {mat_path} | {exc}")
            continue

        frames, slices, coils, ky, kx = shape_to_columns(shape)
        rows.append(
            {
                "mat_path": str(mat_path.relative_to(root_dir)),
                "split": split,
                "smp": smp,
                "acc": acc,
                "patient": patient,
                "viw": viw,
                "kx": kx,
                "ky": ky,
                "coils": coils,
                "slices": slices,
                "frames": frames,
            }
        )

    # Ensure fs samples are processed first (especially for test split).
    split_order = {"train": 0, "val": 1, "test": 2}
    smp_order = {"fs": 0, "pse": 1}
    rows.sort(
        key=lambda r: (
            split_order.get(str(r.get("split", "")).lower(), 99),
            smp_order.get(str(r.get("smp", "")).lower(), 99),
            str(r.get("mat_path", "")),
        )
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mat_path",
        "split",
        "smp",
        "acc",
        "patient",
        "viw",
        "kx",
        "ky",
        "coils",
        "slices",
        "frames",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan CMRxRecon mat files and export CSV")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/CMRxRecon/orig"),
        help="Root directory of CMRxRecon dataset",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data_attributes" / "cmrxrecon_mat_shapes2.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    n_rows = scan_cmrxrecon_to_csv(args.root, args.out)
    print(f"Done. Wrote {n_rows} rows to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
