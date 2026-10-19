"""Compute PSNR / SSIM / HFEN for reconstruction .h5 files against GT volumes.

Inputs:
    --recon-dir   directory of per-method reconstruction .h5 files, either
                  holding them directly (single dataset) or with one
                  sub-directory per dataset/split (e.g. test_gro_06); every
                  .h5 key is one sample.
    --gt-root     directory of GT .h5 files resolved as `gt_dir / key`; each
                  GT file must contain `reconstruction_weighted` and `mask`
                  (the mask is turned into the acceleration factor `acc`).
    --subdirs     optional subset of the sub-directories to process.
    cache         an existing metrics_per_sample.csv in a dataset directory is
                  reused incrementally for methods not yet in it (--force
                  recomputes everything).

Outputs (written into each dataset directory):
    metrics_per_sample.csv  one row per (method, sample): recon_file, key, acc,
                            psnr, ssim, hfen
    metrics_by_method.csv   mean +- std per method: PSNR (dB), SSIM (%),
                            HFEN (x1e-3)
    metrics_summary.json    same aggregation plus the skipped samples
    summary_table.csv       only when several sub-directories were processed:
                            method x acceleration pivot table over all
                            metrics_by_method.csv files

Usage:
    python scripts/calc_metric.py --recon-dir reconstructions/highacc \\
        --gt-root /path/to/processed_ocmr/ocmr_test_gro_06 [--force]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from helpers.metrics import hfen, psnr, ssim


def _to_3d_abs_tensor(arr: np.ndarray) -> torch.Tensor:
	arr = np.abs(np.asarray(arr))

	# Keep only semantic dimensions (T, H, W). OCMR GT often has shape (1, T, H, W).
	while arr.ndim > 3 and arr.shape[0] == 1:
		arr = arr[0]

	if arr.ndim == 2:
		arr = arr[None, ...]

	if arr.ndim != 3:
		raise ValueError(f"Expected a 3D volume, but got shape={arr.shape}")

	return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)


def _mean_std(values: list[float]) -> dict[str, float]:
	vals = np.asarray(values, dtype=np.float64)
	if vals.size == 0:
		return {"mean": float("nan"), "std": float("nan")}
	return {"mean": float(vals.mean()), "std": float(vals.std())}


def _compute_acc_from_mask(mask_arr: np.ndarray) -> float:
	mask_t = torch.as_tensor(np.asarray(mask_arr), dtype=torch.float32)
	denom = torch.sum(mask_t)
	if denom.item() == 0:
		return float("nan")
	return float((torch.numel(mask_t) / denom).item())


def _build_summary(rows: list[dict], skipped: list[dict]) -> dict:
	by_recon_file: dict[str, dict] = {}
	for recon_name in sorted({r["recon_file"] for r in rows}):
		sub = [r for r in rows if r["recon_file"] == recon_name]
		by_recon_file[recon_name] = {
			"count": len(sub),
			"psnr": _mean_std([r["psnr"] for r in sub]),
			"ssim": _mean_std([r["ssim"] for r in sub]),
			"hfen": _mean_std([r["hfen"] for r in sub]),
		}

	summary = {
		"total_samples": len(rows),
		"skipped_samples": len(skipped),
		"by_recon_file": by_recon_file,
		"skipped": skipped,
	}
	return summary


def _load_existing_rows(csv_path: Path) -> list[dict]:
	if not csv_path.exists():
		return []

	rows: list[dict] = []
	with csv_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if not row.get("recon_file") or not row.get("key"):
				continue
			rows.append(
				{
					"recon_file": row["recon_file"],
					"key": row["key"],
					"acc": float(row["acc"]),
					"psnr": float(row["psnr"]),
					"ssim": float(row["ssim"]),
					"hfen": float(row["hfen"]),
				}
			)
	return rows


def _print_profile_summary(profile_stats: dict[str, float], sample_count: int, file_count: int) -> None:
	if sample_count <= 0:
		print("[profile] No valid samples to profile.")
		return

	total = profile_stats.get("total", 0.0)
	print("\n[profile] Time breakdown")
	print(f"[profile] files={file_count}, samples={sample_count}, total={total:.6f}s, avg/sample={total / sample_count:.6f}s")

	items = [(k, v) for k, v in profile_stats.items() if k != "total"]
	items.sort(key=lambda x: x[1], reverse=True)
	for name, sec in items:
		pct = (sec / total * 100.0) if total > 0 else 0.0
		print(f"[profile] {name:>14}: {sec:10.6f}s | {pct:6.2f}% | {sec / sample_count:.6f}s/sample")


def calculate_metrics(
	recon_dir: Path,
	gt_dir: Path,
	include_methods: set[str] | None = None,
	profile_time: bool = False,
	ssim_device: str = "cpu",
) -> tuple[list[dict], dict]:
	"""Compute PSNR/SSIM/HFEN for every sample of every method h5; skips samples whose GT file, keys or shapes are unusable."""
	recon_files = sorted(recon_dir.glob("*.h5"))
	if not recon_files:
		raise FileNotFoundError(f"No .h5 files found in {recon_dir}")

	if include_methods is not None:
		recon_files = [f for f in recon_files if f.name in include_methods]
		if not recon_files:
			return [], _build_summary([], [])

	rows: list[dict] = []
	skipped: list[dict] = []
	if ssim_device not in {"cpu", "cuda", "auto"}:
		raise ValueError(f"Invalid ssim_device={ssim_device}. Expected one of: cpu, cuda, auto")
	if ssim_device == "auto":
		ssim_target_device = "cuda" if torch.cuda.is_available() else "cpu"
	elif ssim_device == "cuda" and not torch.cuda.is_available():
		print("[profile] Requested ssim_device=cuda but CUDA is unavailable, falling back to cpu.")
		ssim_target_device = "cpu"
	else:
		ssim_target_device = ssim_device
	ssim_cuda_enabled = ssim_target_device == "cuda"
	if ssim_cuda_enabled:
		print("[profile] SSIM device: cuda")
	else:
		print("[profile] SSIM device: cpu")
	profile_stats: dict[str, float] = {
		"total": 0.0,
		"read_pred": 0.0,
		"to_tensor_pred": 0.0,
		"read_gt": 0.0,
		"to_tensor_gt": 0.0,
		"acc_from_mask": 0.0,
		"psnr": 0.0,
		"ssim": 0.0,
		"ssim_h2d": 0.0,
		"hfen": 0.0,
	}
	valid_sample_count = 0
	start_all = time.perf_counter()

	for recon_file in tqdm(recon_files, desc="Methods", unit="file"):
		with h5py.File(recon_file, "r") as recon_h5:
			for key in recon_h5.keys():
				sample_start = time.perf_counter()
				gt_file = gt_dir / key
				if not gt_file.exists():
					skipped.append(
						{
							"recon_file": recon_file.name,
							"key": key,
							"reason": "gt_file_not_found",
						}
					)
					continue

				try:
					t0 = time.perf_counter()
					pred_arr = recon_h5[key][()]
					profile_stats["read_pred"] += time.perf_counter() - t0

					t0 = time.perf_counter()
					pred_t = _to_3d_abs_tensor(pred_arr)
					profile_stats["to_tensor_pred"] += time.perf_counter() - t0

					with h5py.File(gt_file, "r") as gt_h5:
						if "reconstruction_weighted" not in gt_h5:
							raise KeyError("reconstruction_weighted not found")
						if "mask" not in gt_h5:
							raise KeyError("mask not found")

						t0 = time.perf_counter()
						gt_arr = gt_h5["reconstruction_weighted"][()]
						mask_arr = gt_h5["mask"][()]
						profile_stats["read_gt"] += time.perf_counter() - t0

						t0 = time.perf_counter()
						gt_t = _to_3d_abs_tensor(gt_arr)
						profile_stats["to_tensor_gt"] += time.perf_counter() - t0

						t0 = time.perf_counter()
						acc = _compute_acc_from_mask(mask_arr)
						profile_stats["acc_from_mask"] += time.perf_counter() - t0

					if pred_t.shape != gt_t.shape:
						raise ValueError(f"shape mismatch pred={tuple(pred_t.shape)} gt={tuple(gt_t.shape)}")

					maxval = torch.max(gt_t)

					t0 = time.perf_counter()
					psnr_val = float(psnr(gt_t, pred_t, maxval=maxval).item())
					profile_stats["psnr"] += time.perf_counter() - t0

					t0 = time.perf_counter()
					if ssim_cuda_enabled:
						t1 = time.perf_counter()
						gt_ssim = gt_t.to(device="cuda", non_blocking=True)
						pred_ssim = pred_t.to(device="cuda", non_blocking=True)
						if profile_time:
							torch.cuda.synchronize()
						profile_stats["ssim_h2d"] += time.perf_counter() - t1
						ssim_maxval = torch.max(gt_ssim)
						ssim_val = float(ssim(gt_ssim, pred_ssim, maxval=ssim_maxval).item())
						if profile_time:
							torch.cuda.synchronize()
					else:
						ssim_val = float(ssim(gt_t, pred_t, maxval=maxval).item())
					profile_stats["ssim"] += time.perf_counter() - t0

					t0 = time.perf_counter()
					hfen_val = float(hfen(gt_t, pred_t).item())
					profile_stats["hfen"] += time.perf_counter() - t0

					row = {
						"recon_file": recon_file.name,
						"key": key,
						"acc": acc,
						"psnr": psnr_val,
						"ssim": ssim_val,
						"hfen": hfen_val,
					}
					rows.append(row)
					valid_sample_count += 1
				except Exception as exc:  # noqa: BLE001
					skipped.append(
						{
							"recon_file": recon_file.name,
							"key": key,
							"reason": str(exc),
						}
					)
				finally:
					profile_stats["total"] += time.perf_counter() - sample_start

	summary = _build_summary(rows, skipped)
	if profile_time:
		elapsed = time.perf_counter() - start_all
		profile_stats["total"] = elapsed
		_print_profile_summary(profile_stats, valid_sample_count, len(recon_files))

	return rows, summary


def run_one_dataset(
	recon_dir: Path,
	gt_root: Path,
	force: bool = False,
	profile_time: bool = False,
	ssim_device: str = "cpu",
) -> None:
	"""Compute (or reuse cached) metrics for one dataset dir and write the three result files."""
	out_csv = recon_dir / "metrics_per_sample.csv"
	out_method_csv = recon_dir / "metrics_by_method.csv"
	out_summary = recon_dir / "metrics_summary.json"

	if force:
		rows_new, summary_new = calculate_metrics(
			recon_dir,
			gt_root,
			profile_time=profile_time,
			ssim_device=ssim_device,
		)
		rows = rows_new
		summary = summary_new
		print(f"Dataset: {recon_dir.name}")
		print("  Incremental mode: disabled by --force, full recomputation.")
		save_per_sample_csv(rows, out_csv)
		save_summary_json(summary, out_summary)
		save_method_summary_csv(summary, out_method_csv)
		print(f"  GT root: {gt_root}")
		print(f"  Per-sample metrics saved to: {out_csv}")
		print(f"  Per-method summary CSV saved to: {out_method_csv}")
		print(f"  Summary saved to: {out_summary}")
		print(f"  Total valid samples: {summary['total_samples']}")
		print(f"  Skipped samples: {summary['skipped_samples']}")
		for recon_file, m in summary["by_recon_file"].items():
			print(
				f"  {recon_file} | n={m['count']} | "
				f"PSNR(mean/std)={m['psnr']['mean']:.6f}/{m['psnr']['std']:.6f} | "
				f"SSIM(mean/std)={m['ssim']['mean']:.6f}/{m['ssim']['std']:.6f} | "
				f"HFEN(mean/std)={m['hfen']['mean']:.6f}/{m['hfen']['std']:.6f}"
			)
		return

	existing_rows = _load_existing_rows(out_csv)
	existing_methods = {r["recon_file"] for r in existing_rows}
	all_methods = {f.name for f in recon_dir.glob("*.h5")}
	new_methods = sorted(all_methods - existing_methods)

	if not out_csv.exists():
		rows_new, summary_new = calculate_metrics(
			recon_dir,
			gt_root,
			profile_time=profile_time,
			ssim_device=ssim_device,
		)
		rows = rows_new
		summary = summary_new
		print(f"Dataset: {recon_dir.name}")
		print("  Incremental mode: no existing csv, full computation.")
	elif not new_methods:
		rows = existing_rows
		summary = _build_summary(rows, skipped=[])
		print(f"Dataset: {recon_dir.name}")
		print("  Incremental mode: no new methods found, skipped computation.")
	else:
		rows_new, summary_new = calculate_metrics(
			recon_dir,
			gt_root,
			include_methods=set(new_methods),
			profile_time=profile_time,
			ssim_device=ssim_device,
		)
		rows = existing_rows + rows_new
		summary = _build_summary(rows, skipped=summary_new["skipped"])
		print(f"Dataset: {recon_dir.name}")
		print(f"  Incremental mode: computed {len(new_methods)} new method(s).")
		print(f"  New methods: {', '.join(new_methods)}")

	save_per_sample_csv(rows, out_csv)
	save_summary_json(summary, out_summary)
	save_method_summary_csv(summary, out_method_csv)

	print(f"  GT root: {gt_root}")
	print(f"  Per-sample metrics saved to: {out_csv}")
	print(f"  Per-method summary CSV saved to: {out_method_csv}")
	print(f"  Summary saved to: {out_summary}")
	print(f"  Total valid samples: {summary['total_samples']}")
	print(f"  Skipped samples: {summary['skipped_samples']}")
	for recon_file, m in summary["by_recon_file"].items():
		print(
			f"  {recon_file} | n={m['count']} | "
			f"PSNR(mean/std)={m['psnr']['mean']:.6f}/{m['psnr']['std']:.6f} | "
			f"SSIM(mean/std)={m['ssim']['mean']:.6f}/{m['ssim']['std']:.6f} | "
			f"HFEN(mean/std)={m['hfen']['mean']:.6f}/{m['hfen']['std']:.6f}"
		)


def save_per_sample_csv(rows: list[dict], out_csv: Path) -> None:
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["recon_file", "key", "acc", "psnr", "ssim", "hfen"])
		writer.writeheader()
		writer.writerows(rows)


def save_summary_json(summary: dict, out_json: Path) -> None:
	out_json.parent.mkdir(parents=True, exist_ok=True)
	with out_json.open("w", encoding="utf-8") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)


def save_method_summary_csv(summary: dict, out_csv: Path) -> None:
	"""Write per-method mean +- std rows (already scaled for display) to metrics_by_method.csv."""
	metric_config: dict[str, tuple[float, str, str]] = {
		"psnr": (1.0, "PSNR", "dB"),
		"ssim": (100.0, "SSIM", "%"),
		"hfen": (1000.0, "HFEN", "x1e-3"),
	}

	def _format_pm(mean_val: float, std_val: float, scale: float) -> str:
		mean_scaled = mean_val * scale
		std_scaled = std_val * scale
		return f"{mean_scaled:.2f} ± {std_scaled:.2f}"

	out_csv.parent.mkdir(parents=True, exist_ok=True)
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"recon_file",
				"count",
				"PSNR (dB)",
				"SSIM (%)",
				"HFEN (x1e-3)",
			],
		)
		writer.writeheader()
		for recon_file, m in summary["by_recon_file"].items():
			psnr_scale, _, _ = metric_config["psnr"]
			ssim_scale, _, _ = metric_config["ssim"]
			hfen_scale, _, _ = metric_config["hfen"]
			writer.writerow(
				{
					"recon_file": recon_file,
					"count": m["count"],
					"PSNR (dB)": _format_pm(m["psnr"]["mean"], m["psnr"]["std"], psnr_scale),
					"SSIM (%)": _format_pm(m["ssim"]["mean"], m["ssim"]["std"], ssim_scale),
					"HFEN (x1e-3)": _format_pm(m["hfen"]["mean"], m["hfen"]["std"], hfen_scale),
				}
			)


def main() -> None:
	"""Parse CLI args, run one dataset or every sub-directory, then aggregate them into summary_table.csv."""
	parser = argparse.ArgumentParser(
		description="Calculate PSNR/SSIM/HFEN for reconstruction files in one or multiple subdirectories."
	)
	parser.add_argument(
		"--recon-dir",
		type=Path,
		default=Path("reconstructions/highacc"),
		help="Root directory containing subdirectories (e.g., test_gro_06) or directly containing .h5 files.",
	)
	parser.add_argument(
		"--gt-root",
		type=Path,
		default=Path("/share/s2d0m12/yhao-z/data/OCMR/processed_ocmr/ocmr_test_gro_06"),
		help="GT root directory containing files with the same names as reconstruction keys.",
	)
	parser.add_argument(
		"--subdirs",
		nargs="*",
		default=None,
		help="Optional list of subdirectory names under recon-dir to run. If omitted, all subdirs are processed.",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Force full recomputation and ignore existing csv cache for each dataset directory.",
	)
	parser.add_argument(
		"--profile-time",
		action="store_true",
		help="Print per-stage timing breakdown for metric computation.",
	)
	parser.add_argument(
		"--ssim-device",
		type=str,
		default="cuda",
		choices=["cpu", "cuda", "auto"],
		help="Device used for SSIM only. Options: cpu, cuda, auto.",
	)
	args = parser.parse_args()

	if any(args.recon_dir.glob("*.h5")):
		run_one_dataset(
			args.recon_dir,
			args.gt_root,
			force=args.force,
			profile_time=args.profile_time,
			ssim_device=args.ssim_device,
		)
		return

	all_subdirs = sorted([d for d in args.recon_dir.iterdir() if d.is_dir()])
	if args.subdirs:
		target_set = set(args.subdirs)
		target_subdirs = [d for d in all_subdirs if d.name in target_set]
	else:
		target_subdirs = all_subdirs

	target_subdirs = [d for d in target_subdirs if any(d.glob("*.h5"))]
	if not target_subdirs:
		raise FileNotFoundError(f"No subdirectories with .h5 files found under {args.recon_dir}")

	for subdir in tqdm(target_subdirs, desc="Datasets", unit="dir"):
		run_one_dataset(
			subdir,
			args.gt_root,
			force=args.force,
			profile_time=args.profile_time,
			ssim_device=args.ssim_device,
		)

	# Generate summary table across all subdirs
	if len(target_subdirs) > 1:
		_generate_summary_table(args.recon_dir)


def _generate_summary_table(recon_dir: Path) -> None:
	"""Summarize metrics_by_method.csv across all subdirs into one comparison table."""
	import re as _re

	def _parse_pm(value: str) -> tuple[float, float]:
		parts = str(value).split("±")
		return float(parts[0].strip()), float(parts[1].strip())

	def _extract_acc(subdir_name: str) -> str:
		m = _re.search(r"(\d+)$", subdir_name)
		if m:
			return str(int(m.group(1)))
		return subdir_name

	def _method_display_name(recon_file: str) -> str:
		stem = Path(recon_file).stem
		if stem.endswith("_reconstructions"):
			stem = stem[: -len("_reconstructions")]
		return stem.replace("_", "-")

	rows = []
	for subdir in sorted(recon_dir.iterdir()):
		if not subdir.is_dir():
			continue
		csv_path = subdir / "metrics_by_method.csv"
		if not csv_path.exists():
			continue
		acc = _extract_acc(subdir.name)
		df = pd.read_csv(csv_path)
		for _, row in df.iterrows():
			method = _method_display_name(str(row["recon_file"]))
			psnr_mean, psnr_std = _parse_pm(row["PSNR (dB)"])
			ssim_mean, ssim_std = _parse_pm(row["SSIM (%)"])
			hfen_mean, hfen_std = _parse_pm(row["HFEN (x1e-3)"])
			rows.append({
				"method": method, "acc": acc,
				"PSNR_mean": psnr_mean, "PSNR_std": psnr_std,
				"SSIM_mean": ssim_mean, "SSIM_std": ssim_std,
				"HFEN_mean": hfen_mean, "HFEN_std": hfen_std,
			})

	if not rows:
		return

	long_df = pd.DataFrame(rows)
	accs = sorted(long_df["acc"].unique(), key=lambda x: int(x) if x.isdigit() else x)
	metrics = ["PSNR", "SSIM", "HFEN"]

	table_rows = []
	for method in sorted(long_df["method"].unique()):
		sub = long_df[long_df["method"] == method]
		row = {"method": method}
		for metric in metrics:
			for acc in accs:
				match = sub[sub["acc"] == acc]
				if not match.empty:
					r = match.iloc[0]
					row[f"{metric}@{acc}X"] = f"{r[f'{metric}_mean']:.2f} ± {r[f'{metric}_std']:.2f}"
				else:
					row[f"{metric}@{acc}X"] = "-"
		table_rows.append(row)

	columns = ["method"] + [f"{m}@{a}X" for m in metrics for a in accs]
	result = pd.DataFrame(table_rows, columns=columns)

	# Sort by PSNR@first_acc ascending
	first_col = f"PSNR@{accs[0]}X"
	result["_sort"] = result[first_col].apply(lambda v: -1e9 if v == "-" else float(v.split("±")[0].strip()))
	result = result.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)

	out_path = recon_dir / "summary_table.csv"
	result.to_csv(out_path, index=False)
	print(f"\nSummary table saved to: {out_path}")
	print(result.to_string(index=False))


if __name__ == "__main__":
	main()
