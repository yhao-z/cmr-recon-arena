"""Render GT / mask / per-method reconstructions and their error maps to figures.

Inputs:
    --recon-dir             per-method reconstruction .h5 files (one key per
                            sample), typically produced by `main.py test`.
    --gt-dir                GT .h5 files resolved as `gt_dir / sample key`;
                            each must contain `reconstruction_weighted`.
    --mask-dir              mask .h5 files resolved as `mask_dir / sample key`;
                            each must contain `mask`.
    --metrics-csv           metrics_by_method.csv (default:
                            <recon-dir>/metrics_by_method.csv) used to order the
                            method columns by mean PSNR, highest on the left.
    --metrics-per-sample-csv metrics_per_sample.csv (default:
                            <recon-dir>/metrics_per_sample.csv) used by
                            --auto-sample top2_gap / top2_gap_trend to pick the
                            samples to render.

Output:
    one PNG/PDF-compatible figure per rendered sample (Agg backend, 180 dpi):
    top row GT + one column per method, bottom row mask + |pred - GT| error map,
    each panel optionally padded with x-t (bottom) and y-t (right) strips.
    Files are written under --recon-dir using --output, with a `_rank{R}` suffix
    when several samples are rendered in one run.

Usage:
    python scripts/visualize_recon_methods_with_gt_error.py \\
        --recon-dir reconstructions/<dataset> --methods CineVN L+S-Net
"""
import argparse
import csv
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FONT_FAMILY = "Times New Roman"
TITLE_FONT_SIZE = 14
INFO_FONT_SIZE = 9

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": TITLE_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": TITLE_FONT_SIZE,
        "xtick.labelsize": TITLE_FONT_SIZE,
        "ytick.labelsize": TITLE_FONT_SIZE,
    }
)


def _str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = v.strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def resolve_gt_dir(gt_dir: Path, recon_subdir_name: str) -> Path:
    """Keep compatibility with existing call sites and validate the GT directory.

    NOTE: ``recon_subdir_name`` is unused intentionally. We now follow the same
    behavior as calc_metric: directly use the provided GT directory and resolve
    sample files by ``gt_dir / sample_key``.
    """
    _ = recon_subdir_name
    if not gt_dir.exists() or not gt_dir.is_dir():
        raise FileNotFoundError(f"GT dir not found: {gt_dir}")
    return gt_dir


def resolve_mask_dir(mask_dir: Path, recon_subdir_name: str) -> Path:
    """Validate the mask directory and keep call-site compatibility."""
    _ = recon_subdir_name
    if not mask_dir.exists() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")
    return mask_dir


def _to_3d_abs(arr: np.ndarray) -> np.ndarray:
    x = np.abs(np.asarray(arr))

    while x.ndim > 3 and x.shape[0] == 1:
        x = x[0]

    if x.ndim == 2:
        x = x[None, ...]

    if x.ndim != 3:
        raise ValueError(f"Expected 2D/3D volume-like data, got shape={x.shape}")

    return x


def _to_2d_mask(arr: np.ndarray, frame_idx: int) -> np.ndarray:
    x = np.asarray(arr)

    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]

    # Collapse trailing singleton dims aggressively.
    x = np.squeeze(x)

    if x.ndim == 2:
        return x.astype(np.float32)

    if x.ndim >= 3:
        t_idx = max(0, min(frame_idx, x.shape[0] - 1))
        x2d = np.squeeze(x[t_idx])
        if x2d.ndim == 2:
            return x2d.astype(np.float32)

    # Last-resort fallback for unusual mask layouts: flatten to first 2D plane.
    y = np.asarray(x)
    while y.ndim > 2:
        y = y[0]
    if y.ndim != 2:
        raise ValueError(f"Unable to convert mask to 2D, got shape={arr.shape}")
    return y.astype(np.float32)


def _pick_sample_key_by_index(method_files: list[Path], sample_idx: int) -> str:
    if not method_files:
        raise FileNotFoundError("No method .h5 files found.")

    with h5py.File(method_files[0], "r") as hf:
        keys = sorted(list(hf.keys()))
        if not keys:
            raise ValueError(f"No keys found in {method_files[0]}")

        idx = int(sample_idx)
        if idx < 0:
            idx = len(keys) + idx
        idx = max(0, min(idx, len(keys) - 1))
        return keys[idx]


def _load_pred_frame(h5_path: Path, key: str, frame_idx: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as hf:
        if key not in hf:
            raise KeyError(f"Key '{key}' not found in {h5_path.name}")
        vol = _to_3d_abs(hf[key][...])

    t_idx = max(0, min(frame_idx, vol.shape[0] - 1))
    return vol[t_idx]


def _load_pred_volume(h5_path: Path, key: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as hf:
        if key not in hf:
            raise KeyError(f"Key '{key}' not found in {h5_path.name}")
        vol = _to_3d_abs(hf[key][...])
    return vol


def _to_mask_volume(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Coerce an arbitrary mask layout to (t, h, w), center-cropping/padding to `target_shape`."""
    target_t, target_h, target_w = target_shape
    x = np.asarray(arr)

    def _fit_last_axis(mask_3d: np.ndarray, axis: int, target_size: int) -> np.ndarray:
        """Center-crop when larger; zero-pad when smaller."""
        cur = mask_3d.shape[axis]
        if cur == target_size:
            return mask_3d

        if cur > target_size:
            start = (cur - target_size) // 2
            end = start + target_size
            slicer = [slice(None)] * mask_3d.ndim
            slicer[axis] = slice(start, end)
            return mask_3d[tuple(slicer)]

        # cur < target_size: pad with zeros around center.
        pad_before = (target_size - cur) // 2
        pad_after = target_size - cur - pad_before
        pad_width = [(0, 0)] * mask_3d.ndim
        pad_width[axis] = (pad_before, pad_after)
        return np.pad(mask_3d, pad_width=pad_width, mode="constant", constant_values=0)

    while x.ndim > 3 and x.shape[0] == 1:
        x = x[0]

    # Canonicalize to (t, h_like, w_like).
    if x.ndim == 2:
        if x.shape[0] == target_t:
            # Common case in this project: (t, w) == (t, 1, w)
            x = x[:, None, :]
        else:
            # Fallback: treat as (h, w)
            x = np.repeat(x[None, ...], target_t, axis=0)
    elif x.ndim == 3:
        # Handle layouts like (h, 1, t) or (h, w, t)
        if x.shape[0] != target_t and x.shape[-1] == target_t:
            x = np.moveaxis(x, -1, 0)

        # Handle singleton time dimension
        if x.shape[0] == 1 and target_t > 1:
            x = np.repeat(x, target_t, axis=0)

        # Fallback if time still mismatched
        if x.shape[0] != target_t:
            x = np.repeat(x[0:1], target_t, axis=0)
    else:
        raise ValueError(f"Unexpected mask shape: {arr.shape}")

    # Align H (supports (t,1,w)->(t,h,w)).
    if x.shape[1] != target_h:
        if x.shape[1] == 1:
            x = np.repeat(x, target_h, axis=1)
        else:
            x = _fit_last_axis(x, axis=1, target_size=target_h)

    # Align W (also supports (t,h,1)->(t,h,w)).
    if x.shape[2] != target_w:
        if x.shape[2] == 1:
            x = np.repeat(x, target_w, axis=2)
        else:
            x = _fit_last_axis(x, axis=2, target_size=target_w)

    return x.astype(np.float32)


def _load_gt_volume(gt_file: Path) -> np.ndarray:
    with h5py.File(gt_file, "r") as hf:
        if "reconstruction_weighted" not in hf:
            raise KeyError(f"reconstruction_weighted not found in {gt_file}")
        gt_vol = _to_3d_abs(hf["reconstruction_weighted"][...])
    return gt_vol


def _load_mask_volume(mask_file: Path, target_shape: tuple[int, int, int]) -> np.ndarray:
    with h5py.File(mask_file, "r") as hf:
        if "mask" not in hf:
            raise KeyError(f"mask not found in {mask_file}")
        mask_vol = _to_mask_volume(hf["mask"][...], target_shape=target_shape)
    return mask_vol


def _normalize_gray(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    p1 = float(np.percentile(x, 1.0))
    p99 = float(np.percentile(x, 99.5))
    if p99 <= p1:
        p1 = float(np.min(x))
        p99 = float(np.max(x) + 1e-12)
    x = np.clip((x - p1) / (p99 - p1 + 1e-12), 0.0, 1.0)
    return x


def _normalize_volume(vol: np.ndarray, p1: float | None = None, p99: float | None = None) -> np.ndarray:
    x = np.asarray(vol, dtype=np.float32)
    if p1 is None:
        p1 = float(np.percentile(x, 1.0))
    if p99 is None:
        p99 = float(np.percentile(x, 99.5))
    if p99 <= p1:
        p1 = float(np.min(x))
        p99 = float(np.max(x) + 1e-12)
    x = np.clip((x - p1) / (p99 - p1 + 1e-12), 0.0, 1.0)
    return x


def _compute_global_norm_stats(gt_vol: np.ndarray, pred_by_method: list[dict]) -> tuple[float, float]:
    """Compute one shared normalization range for GT + all methods of one sample."""
    vols = [np.asarray(gt_vol, dtype=np.float32).reshape(-1)]
    for item in pred_by_method:
        vols.append(np.asarray(item["pred_vol"], dtype=np.float32).reshape(-1))

    all_vals = np.concatenate(vols)
    p1 = float(np.percentile(all_vals, 1.0))
    p99 = float(np.percentile(all_vals, 99.5))
    if p99 <= p1:
        p1 = float(np.min(all_vals))
        p99 = float(np.max(all_vals) + 1e-12)
    return p1, p99


def _resize_nearest(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = img.shape
    if h == out_h and w == out_w:
        return img
    r_idx = np.clip(np.round(np.linspace(0, h - 1, out_h)).astype(np.int64), 0, h - 1)
    c_idx = np.clip(np.round(np.linspace(0, w - 1, out_w)).astype(np.int64), 0, w - 1)
    return img[np.ix_(r_idx, c_idx)]


def _resolve_axis_index(pos: float, size: int) -> int:
    """Resolve index from position.

    If 0<=pos<=1, treat as normalized ratio; otherwise treat as pixel index.
    """
    if size <= 0:
        return 0

    if 0.0 <= pos <= 1.0:
        idx = int(round(pos * (size - 1)))
    else:
        idx = int(round(pos))

    return max(0, min(idx, size - 1))


def _build_panel_image(
    frame_2d: np.ndarray,
    vol_3d: np.ndarray,
    strip_ratio: float = 0.30,
    strip_gap_ratio: float = 0.03,
    crosshair_x: float = 0.5,
    crosshair_y: float = 0.5,
    show_ht: bool = True,
    show_wt: bool = True,
) -> tuple[np.ndarray, dict]:
    """Build a panel with y-t(right) and x-t(bottom) strips and return layout metadata."""
    img = np.asarray(frame_2d)
    vol = np.asarray(vol_3d)

    if vol.ndim != 3:
        raise ValueError(f"vol_3d must be 3D, got shape={vol.shape}")
    if img.ndim != 2:
        raise ValueError(f"frame_2d must be 2D, got shape={img.shape}")

    h, w = img.shape
    t, vh, vw = vol.shape
    if vh != h or vw != w:
        raise ValueError(f"volume/frame shape mismatch: vol={(t, vh, vw)} frame={(h, w)}")

    y_idx = _resolve_axis_index(crosshair_y, h)
    x_idx = _resolve_axis_index(crosshair_x, w)

    # Use one shared strip size so right y-t width equals bottom x-t height.
    strip_size = max(2, int(round(min(h, w) * strip_ratio)))
    right_w = strip_size if show_ht else 0
    bottom_h = strip_size if show_wt else 0
    gap = max(2, int(round(min(h, w) * strip_gap_ratio))) if (show_ht or show_wt) else 0

    # y-t: [H, T], x-t: [T, W]
    yt = vol[:, :, x_idx].T
    xt = vol[:, y_idx, :]

    yt_strip = _resize_nearest(yt, out_h=h, out_w=right_w) if show_ht else None
    xt_strip = _resize_nearest(xt, out_h=bottom_h, out_w=w) if show_wt else None

    # Use NaN for blank regions; rendered as white via colormap bad color.
    panel = np.full((h + gap + bottom_h, w + gap + right_w), np.nan, dtype=np.float32)
    panel[:h, :w] = img
    if show_ht and yt_strip is not None:
        panel[:h, w + gap:] = yt_strip
    if show_wt and xt_strip is not None:
        panel[h + gap:, :w] = xt_strip

    meta = {
        "h": h,
        "w": w,
        "right_w": right_w,
        "bottom_h": bottom_h,
        "gap": gap,
        "x_idx": x_idx,
        "y_idx": y_idx,
        "show_ht": show_ht,
        "show_wt": show_wt,
    }
    return panel, meta


def _draw_crosshair(ax: plt.Axes, meta: dict, color: str = "red") -> None:
    """Draw crosshair lines over the main image region to indicate x-t/y-t slice positions."""
    h = int(meta["h"])
    w = int(meta["w"])
    x_idx = int(meta["x_idx"])
    y_idx = int(meta["y_idx"])

    x = x_idx
    y = y_idx

    if bool(meta.get("show_ht", True)):
        ax.plot([x, x], [0, h - 1], color=color, linewidth=0.7)
    if bool(meta.get("show_wt", True)):
        ax.plot([0, w - 1], [y, y], color=color, linewidth=0.7)


def _imshow_with_white_blank(
    ax: plt.Axes,
    panel: np.ndarray,
    cmap_name: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    masked = np.ma.masked_invalid(panel)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white")
    # Use aspect='auto' so subplot spacing (wspace/hspace) is visually effective.
    ax.imshow(masked, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax, aspect="auto")


def _sort_methods_by_psnr(method_files: list[Path], metrics_csv: Path | None) -> list[Path]:
    """Sort method files by psnr_mean (descending) from metrics_by_method.csv."""
    if metrics_csv is None or not metrics_csv.exists():
        return sorted(method_files)

    psnr_by_name: dict[str, float] = {}
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            recon_name = row.get("recon_file", "").strip()
            psnr_str = row.get("PSNR (dB)", "").strip().split(" ± ")[0]
            if not recon_name or not psnr_str:
                continue
            try:
                psnr_by_name[recon_name] = float(psnr_str)
            except ValueError:
                continue

    if not psnr_by_name:
        return sorted(method_files)

    # Higher PSNR shown on the left. Unknown methods are placed at the right.
    return sorted(
        method_files,
        key=lambda p: (
            0 if p.name in psnr_by_name else 1,
            -psnr_by_name.get(p.name, -1e18),
            p.name,
        ),
    )


def _rank_methods_by_mean_psnr(
    metrics_csv: Path,
    allowed_methods: set[str] | None = None,
) -> list[str]:
    """Rank method recon_file names by mean PSNR (descending), keeping only allowed ones."""
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics_by_method.csv not found: {metrics_csv}")

    rows = []
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            recon_file = row.get("recon_file", "").strip()
            psnr_str = row.get("PSNR (dB)", "").strip().split(" ± ")[0]
            if not recon_file or not psnr_str:
                continue
            # Filter by allowed methods if specified
            if allowed_methods is not None:
                stem = Path(recon_file).stem
                prefix = stem.replace("_reconstructions", "")
                # Normalize for fuzzy matching
                if not any(m.replace("-", "_") == prefix.replace("-", "_") for m in allowed_methods):
                    continue
            try:
                psnr_mean = float(psnr_str)
            except ValueError:
                continue
            rows.append((recon_file, psnr_mean))

    if len(rows) < 2:
        raise ValueError(f"Need at least 2 valid methods in {metrics_csv} (after filtering)")

    rows.sort(key=lambda x: x[1], reverse=True)
    return [recon_file for recon_file, _ in rows]


def _filter_samples_by_psnr_trend(
    metrics_per_sample_csv: Path,
    method_order: list[str],
    min_agree: float = 1.0,
) -> tuple[list[str], int]:
    """Keep samples whose per-sample PSNR order matches the global mean-PSNR order.

    ``method_order`` is the recon_file list sorted by mean PSNR (descending).
    For every sample, the fraction of method pairs whose per-sample PSNR order
    agrees with ``method_order`` is computed; samples with
    ``agree_fraction >= min_agree`` are kept (1.0 == exact same trend).

    Returns (kept_keys, n_evaluated).
    """
    if not metrics_per_sample_csv.exists():
        raise FileNotFoundError(f"metrics_per_sample.csv not found: {metrics_per_sample_csv}")
    if len(method_order) < 2:
        raise ValueError("Need at least 2 methods to compare PSNR trend.")
    if not 0.0 <= min_agree <= 1.0:
        raise ValueError(f"--trend-agree must be in [0, 1], got {min_agree}")

    psnr_by_key: dict[str, dict[str, float]] = {}
    wanted = set(method_order)
    with metrics_per_sample_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("key", "").strip()
            recon_file = row.get("recon_file", "").strip()
            psnr_str = row.get("psnr", "").strip()
            if not key or recon_file not in wanted or not psnr_str:
                continue
            try:
                psnr_val = float(psnr_str)
            except ValueError:
                continue
            psnr_by_key.setdefault(key, {})[recon_file] = psnr_val

    n_pairs = len(method_order) * (len(method_order) - 1) // 2
    kept: list[str] = []
    n_evaluated = 0
    for key in sorted(psnr_by_key):
        per_sample = psnr_by_key[key]
        if any(m not in per_sample for m in method_order):
            continue
        n_evaluated += 1
        agree = 0
        for i, m_a in enumerate(method_order):
            for m_b in method_order[i + 1 :]:
                if per_sample[m_a] > per_sample[m_b]:
                    agree += 1
        if agree / n_pairs >= min_agree:
            kept.append(key)

    return kept, n_evaluated


def _rank_samples_by_top2_gap(
    metrics_per_sample_csv: Path,
    method_a: str,
    method_b: str,
) -> list[dict]:
    """Rank all common sample keys by (psnr_a - psnr_b) descending.

    Returns a list of dicts: {"key", "psnr_a", "psnr_b", "gap"}.
    """
    if not metrics_per_sample_csv.exists():
        raise FileNotFoundError(f"metrics_per_sample.csv not found: {metrics_per_sample_csv}")

    psnr_a_by_key: dict[str, float] = {}
    psnr_b_by_key: dict[str, float] = {}

    with metrics_per_sample_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            recon_file = row.get("recon_file", "").strip()
            key = row.get("key", "").strip()
            psnr_str = row.get("psnr", "").strip()
            if not recon_file or not key or not psnr_str:
                continue
            try:
                psnr_val = float(psnr_str)
            except ValueError:
                continue

            if recon_file == method_a:
                psnr_a_by_key[key] = psnr_val
            elif recon_file == method_b:
                psnr_b_by_key[key] = psnr_val

    common_keys = sorted(set(psnr_a_by_key.keys()) & set(psnr_b_by_key.keys()))
    if not common_keys:
        raise ValueError(
            f"No overlapping sample keys found between {method_a} and {method_b} in {metrics_per_sample_csv}"
        )

    ranked: list[dict] = []
    for key in common_keys:
        a_val = psnr_a_by_key[key]
        b_val = psnr_b_by_key[key]
        ranked.append({"key": key, "psnr_a": a_val, "psnr_b": b_val, "gap": a_val - b_val})

    if not ranked:
        raise RuntimeError("Failed to rank samples by PSNR gap.")

    ranked.sort(key=lambda item: (-item["gap"], item["key"]))
    return ranked


def _load_sample_metrics_for_key(metrics_per_sample_csv: Path, sample_key: str) -> dict[str, tuple[float, float]]:
    """Load per-method (psnr, ssim) for one sample key.

    Returns a dict keyed by both file name and stem to ease lookup.
    """
    if not metrics_per_sample_csv.exists():
        return {}

    metrics: dict[str, tuple[float, float]] = {}
    with metrics_per_sample_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("key", "").strip()
            recon_file = row.get("recon_file", "").strip()
            psnr_str = row.get("psnr", "").strip()
            ssim_str = row.get("ssim", "").strip()
            if key != sample_key or not recon_file or not psnr_str or not ssim_str:
                continue
            try:
                psnr_val = float(psnr_str)
                ssim_val = float(ssim_str)
            except ValueError:
                continue

            metrics[recon_file] = (psnr_val, ssim_val)
            metrics[Path(recon_file).stem] = (psnr_val, ssim_val)

    return metrics


def _format_method_display_name(method_name: str) -> str:
    """Format method names for figure titles.

    Rules:
    1) Replace '_' with '-'
    2) Remove trailing '-reconstructions' if present
    """
    display = method_name.replace("_", "-")
    suffix = "-reconstructions"
    if display.endswith(suffix):
        display = display[: -len(suffix)]
    return display


def build_figure(
    gt_vol: np.ndarray,
    mask_vol: np.ndarray,
    pred_by_method: list[dict],
    sample_metrics: dict[str, tuple[float, float]],
    out_path: Path,
    sample_key: str,
    frame_idx: int,
    error_cmap: str,
    strip_ratio: float,
    strip_gap_ratio: float,
    crosshair_x: float,
    crosshair_y: float,
    show_ht: bool,
    show_wt: bool,
    recon_vmax: float,
    subplot_wspace: float,
    subplot_hspace: float,
) -> None:
    """Draw the 2-row figure (GT/mask row, method/error row) and save it to `out_path`."""
    n_methods = len(pred_by_method)
    n_cols = 1 + n_methods  # GT/mask + each method

    width = 1.6 * n_cols if show_ht else 1.4 * n_cols
    fig, axes = plt.subplots(2, n_cols, figsize=(width, 2.6 * 2), squeeze=False)

    t_idx = max(0, min(frame_idx, gt_vol.shape[0] - 1))
    global_p1, global_p99 = _compute_global_norm_stats(gt_vol, pred_by_method)

    # Shared normalization across GT and all methods for fair visual comparison.
    gt_show_vol = _normalize_volume(gt_vol, p1=global_p1, p99=global_p99)
    gt_show = gt_show_vol[t_idx]

    mask_show_vol = _normalize_volume(mask_vol)
    mask_show = mask_show_vol[t_idx]

    errors = []
    for item in pred_by_method:
        pred_norm_vol = _normalize_volume(item["pred_vol"], p1=global_p1, p99=global_p99)
        err_norm_vol = np.abs(pred_norm_vol - gt_show_vol)
        errors.append(err_norm_vol[t_idx])
    if errors:
        all_err = np.concatenate([e.reshape(-1) for e in errors])
        err_vmax = float(np.percentile(all_err, 99.0))
        err_vmax = max(err_vmax, 1e-12)
    else:
        err_vmax = 1.0

    recon_vmax = max(float(recon_vmax), 1e-6)

    ax = axes[0][0]
    gt_panel, gt_meta = _build_panel_image(
        gt_show,
        gt_show_vol,
        strip_ratio=strip_ratio,
        strip_gap_ratio=strip_gap_ratio,
        crosshair_x=crosshair_x,
        crosshair_y=crosshair_y,
        show_ht=show_ht,
        show_wt=show_wt,
    )
    _imshow_with_white_blank(ax, gt_panel, cmap_name="gray", vmin=0.0, vmax=recon_vmax)
    if show_ht or show_wt:
        _draw_crosshair(ax, gt_meta, color="red")
    ax.set_title("GT", fontsize=TITLE_FONT_SIZE)
    ax.axis("off")

    ax = axes[1][0]
    mask_panel, _ = _build_panel_image(
        mask_show,
        mask_show_vol,
        strip_ratio=strip_ratio,
        strip_gap_ratio=strip_gap_ratio,
        crosshair_x=crosshair_x,
        crosshair_y=crosshair_y,
        show_ht=show_ht,
        show_wt=show_wt,
    )
    _imshow_with_white_blank(ax, mask_panel, cmap_name="gray", vmin=0.0, vmax=1.0)
    ax.set_title("", fontsize=TITLE_FONT_SIZE)
    ax.axis("off")

    for i, item in enumerate(pred_by_method, start=1):
        method_name = item["name"]
        method_title = _format_method_display_name(method_name)
        pred_show_vol = _normalize_volume(item["pred_vol"], p1=global_p1, p99=global_p99)
        pred_show = pred_show_vol[t_idx]
        err_vol = np.abs(pred_show_vol - gt_show_vol)
        err_img = err_vol[t_idx]

        ax_top = axes[0][i]
        pred_panel, _ = _build_panel_image(
            pred_show,
            pred_show_vol,
            strip_ratio=strip_ratio,
            strip_gap_ratio=strip_gap_ratio,
            crosshair_x=crosshair_x,
            crosshair_y=crosshair_y,
            show_ht=show_ht,
            show_wt=show_wt,
        )
        _imshow_with_white_blank(ax_top, pred_panel, cmap_name="gray", vmin=0.0, vmax=recon_vmax)
        ax_top.set_title(method_title, fontsize=TITLE_FONT_SIZE)
        ax_top.axis("off")

        ax_bottom = axes[1][i]
        err_panel, _ = _build_panel_image(
            err_img,
            err_vol,
            strip_ratio=strip_ratio,
            strip_gap_ratio=strip_gap_ratio,
            crosshair_x=crosshair_x,
            crosshair_y=crosshair_y,
            show_ht=show_ht,
            show_wt=show_wt,
        )
        _imshow_with_white_blank(ax_bottom, err_panel, cmap_name=error_cmap, vmin=0.0, vmax=err_vmax)
        ax_bottom.set_title("", fontsize=TITLE_FONT_SIZE)

        metric = sample_metrics.get(method_name)
        if metric is not None:
            psnr_val, ssim_val = metric
            ax_bottom.text(
                0.5,
                -0.03,
                f"PSNR={psnr_val:.2f} SSIM={ssim_val:.3f}",
                transform=ax_bottom.transAxes,
                ha="center",
                va="top",
                fontsize=INFO_FONT_SIZE,
                color="black",
                clip_on=False,
            )

        ax_bottom.axis("off")

    # fig.suptitle(f"Sample: {sample_key} | frame={frame_idx}", fontsize=TITLE_FONT_SIZE)
    fig.subplots_adjust(
        left=0.01,
        right=0.995,
        top=0.90,
        bottom=0.10,
        wspace=max(0.0, float(subplot_wspace)),
        hspace=max(0.0, float(subplot_hspace)),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse all CLI options (paths, sample selection, layout and output)."""
    parser = argparse.ArgumentParser(
        description="Visualize GT/mask and method reconstructions with error maps (2-row layout)."
    )
    parser.add_argument(
        "--crosshair-x",
        type=float,
        default=0.5,
        help="Crosshair x position. If in [0,1], treated as normalized ratio; otherwise pixel index.",
    )
    parser.add_argument(
        "--crosshair-y",
        type=float,
        default=0.45,
        help="Crosshair y position. If in [0,1], treated as normalized ratio; otherwise pixel index.",
    )
    parser.add_argument(
        "--recon-dir",
        type=Path,
        default=Path("reconstructions/cmrxrecon/cmrxrecon23_test_pse_08"),
        help="Directory containing method .h5 reconstruction files.",
    )
    parser.add_argument(
        "--gt-dir",
        "--gt-root",
        dest="gt_dir",
        type=Path,
        default=Path("/share/s2d0m12/yhao-z/data/CMRxRecon/processed_cmrxrecon23/cmrxrecon23_test_fs"),
        help="GT directory containing GT .h5 files (matched directly by sample key filename).",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("/share/s2d0m12/yhao-z/data/CMRxRecon/processed_cmrxrecon23/cmrxrecon23_test_pse_08"),
        help="Mask directory containing mask .h5 files (matched directly by sample key filename).",
    )
    parser.add_argument(
        "--sample-key",
        type=str,
        default=None,
        help="Deprecated and never read; use --sample-idx instead.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=765,
        help="Sample index in sorted keys of method file (supports negative indices).",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Frame index to visualize from each 3D reconstruction volume.",
    )
    parser.add_argument(
        "--error-cmap",
        type=str,
        default="jet",
        help="Colormap for error maps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cmrxrecon23_pse08_3methods.pdf"),
        help="Output image path, written under --recon-dir (absolute paths are kept as-is). "
             "A _rank{R} suffix is added when several samples are rendered.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional metrics csv path (defaults to recon_dir/metrics_by_method.csv).",
    )
    parser.add_argument(
        "--metrics-per-sample-csv",
        type=Path,
        default=None,
        help="Optional per-sample metrics csv path (defaults to recon_dir/metrics_per_sample.csv).",
    )
    parser.add_argument(
        "--auto-sample",
        type=str,
        default="top2_gap_trend",
        choices=["none", "top2_gap", "top2_gap_trend"],
        help=(
            "Sample selection mode: none (use --sample-idx, first key otherwise); "
            "top2_gap (max A-B PSNR gap); "
            "top2_gap_trend (first keep samples whose per-sample PSNR order matches "
            "the mean-PSNR order of the displayed methods, then rank by top2_gap)."
        ),
    )
    parser.add_argument(
        "--trend-agree",
        type=float,
        default=1.0,
        help=(
            "top2_gap_trend only: minimum fraction of method pairs whose per-sample PSNR "
            "order must match the mean-PSNR order (1.0 = exact same trend, 0.9 = allow up to "
            "10 percent of pairs to disagree)."
        ),
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="*",
        default=[3],
        metavar="R",
        help=(
            "top2_gap / top2_gap_trend only: 1-based ranks in the sorted list to visualize, "
            "e.g. '--ranks 1 2 3' renders the top-3 samples "
            "(default: 3; a bare '--ranks' with no values renders ranks 1-5; "
            "only usable with these auto-sample modes)."
        ),
    )
    parser.add_argument(
        "--rank-preview",
        type=int,
        default=10,
        help="top2_gap / top2_gap_trend only: how many ranked samples to print in the ranking table (default: 10).",
    )
    parser.add_argument(
        "--strip-ratio",
        type=float,
        default=0.30,
        help="Relative thickness of y-t/x-t strips to main image size.",
    )
    parser.add_argument(
        "--strip-gap-ratio",
        type=float,
        default=0.03,
        help="Relative gap between main image and y-t/x-t strips.",
    )
    parser.add_argument(
        "--show-ht",
        type=_str2bool,
        default=False,
        metavar="{true,false}",
        help="Whether to show h-t strip on the right (default: false).",
    )
    parser.add_argument(
        "--show-wt",
        type=_str2bool,
        default=True    ,
        metavar="{true,false}",
        help="Whether to show w-t strip on the bottom (default: true).",
    )
    parser.add_argument(
        "--recon-vmax",
        type=float,
        default=0.7,
        help="Display vmax for GT and reconstruction images. Smaller values make images appear brighter (e.g. 0.8).",
    )
    parser.add_argument(
        "--subplot-wspace",
        type=float,
        default=0.03,
        help="Horizontal spacing between subplot columns.",
    )
    parser.add_argument(
        "--subplot-hspace",
        type=float,
        default=0.03,
        help="Vertical spacing between subplot rows.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=['CineVN', 'L+S-Net'],
        help="Only visualize specified methods (h5 file prefixes, e.g. 'CineVN L+S-Net'). If omitted, all methods are shown.",
    )
    return parser.parse_args()


def _resolve_out_path(args: argparse.Namespace, n_methods: int, rank: int | None, multi: bool) -> Path:
    """Build the output path for one rendered sample (relative paths land under ``--recon-dir``).

    A ``_rank{R}`` suffix is appended when several samples are rendered in one
    run, or when a single non-top-1 rank is rendered, so files never collide
    with the default name.
    """
    rank_suffix = f"_rank{rank}" if (rank is not None and (multi or rank > 1)) else ""

    out = Path(args.output)
    if rank_suffix:
        out = Path(f"{out.stem}{rank_suffix}{out.suffix}")
    return args.recon_dir / out


def main() -> None:
    """Select methods and sample(s), then render one figure per selected sample."""
    args = parse_args()

    if not args.recon_dir.exists() or not args.recon_dir.is_dir():
        raise FileNotFoundError(f"recon_dir not found: {args.recon_dir}")

    method_files = sorted(args.recon_dir.glob("*.h5"))
    if not method_files:
        raise FileNotFoundError(f"No .h5 files found in {args.recon_dir}")

    # Filter methods if --methods is specified
    if args.methods:
        method_set = set(args.methods)
        def _match_method(f: Path) -> bool:
            stem = f.stem
            if stem in method_set:
                return True
            prefix = stem.replace("_reconstructions", "")
            if prefix in method_set:
                return True
            # Normalize: replace - with _ and vice versa for fuzzy matching
            normalized = prefix.replace("-", "_")
            for m in method_set:
                if m.replace("-", "_") == normalized:
                    return True
            return False
        method_files = [f for f in method_files if _match_method(f)]
        if not method_files:
            raise FileNotFoundError(f"No matching .h5 files found for methods: {args.methods}")
        print(f"Selected {len(method_files)} method(s): {[f.stem for f in method_files]}")

    metrics_csv = args.metrics_csv if args.metrics_csv is not None else (args.recon_dir / "metrics_by_method.csv")
    method_files = _sort_methods_by_psnr(method_files, metrics_csv)

    metrics_per_sample_csv = (
        args.metrics_per_sample_csv
        if args.metrics_per_sample_csv is not None
        else (args.recon_dir / "metrics_per_sample.csv")
    )

    # ---- Decide which sample(s) to render ----
    ranks_arg = getattr(args, "ranks", None)
    auto_info = None
    if args.auto_sample in ("top2_gap", "top2_gap_trend"):
        allowed = set(args.methods) if args.methods else None
        method_order = _rank_methods_by_mean_psnr(metrics_csv, allowed_methods=allowed)
        method_a, method_b = method_order[0], method_order[1]
        ranked = _rank_samples_by_top2_gap(
            metrics_per_sample_csv=metrics_per_sample_csv,
            method_a=method_a,
            method_b=method_b,
        )

        trend_info = None
        if args.auto_sample == "top2_gap_trend":
            kept_keys, n_evaluated = _filter_samples_by_psnr_trend(
                metrics_per_sample_csv=metrics_per_sample_csv,
                method_order=method_order,
                min_agree=float(args.trend_agree),
            )
            kept_set = set(kept_keys)
            n_before = len(ranked)
            ranked = [item for item in ranked if item["key"] in kept_set]
            if not ranked:
                raise ValueError(
                    f"No sample satisfies the PSNR-trend filter "
                    f"(kept {len(kept_keys)}/{n_evaluated} samples with all methods, "
                    f"agree>={args.trend_agree}); lower --trend-agree."
                )
            trend_info = {
                "kept": len(ranked),
                "n_before": n_before,
                "n_evaluated": n_evaluated,
                "min_agree": float(args.trend_agree),
                "method_order": method_order,
            }

        ranks = sorted({int(r) for r in ranks_arg}) if ranks_arg else [1, 2, 3, 4, 5]
        invalid = [r for r in ranks if r < 1 or r > len(ranked)]
        if invalid:
            raise ValueError(
                f"Invalid --ranks {invalid}: valid range is 1..{len(ranked)} "
                f"(A={method_a}, B={method_b}, {len(ranked)} candidate samples)"
            )

        preview_n = min(max(int(args.rank_preview), max(ranks)), len(ranked))
        print(f"top2_gap ranking over {len(ranked)} samples | A={method_a} | B={method_b}")
        if trend_info is not None:
            order_disp = " > ".join(_format_method_display_name(Path(m).stem) for m in trend_info["method_order"])
            print(
                f"PSNR-trend filter: kept {trend_info['kept']}/{trend_info['n_before']} samples "
                f"(evaluated {trend_info['n_evaluated']}, agree>={trend_info['min_agree']}) | "
                f"expected per-sample order: {order_disp}"
            )
        for i, item in enumerate(ranked[:preview_n], start=1):
            marker = "  <== selected" if i in ranks else ""
            print(
                f"  #{i:<4d} key={item['key']}  "
                f"A={item['psnr_a']:.4f}  B={item['psnr_b']:.4f}  gap={item['gap']:.4f}{marker}"
            )
        if preview_n < len(ranked):
            print(f"  ... ({len(ranked) - preview_n} more; raise --rank-preview to show them)")

        samples_to_render = [(r, ranked[r - 1]["key"]) for r in ranks]
        auto_info = {
            "method_a": method_a,
            "method_b": method_b,
            "ranked": ranked,
            "ranks": ranks,
            "trend": trend_info,
        }
    else:
        if ranks_arg:
            raise ValueError("--ranks requires --auto-sample top2_gap or top2_gap_trend")
        samples_to_render = [(None, _pick_sample_key_by_index(method_files, args.sample_idx))]

    gt_dir = resolve_gt_dir(args.gt_dir, args.recon_dir.name)
    mask_dir = resolve_mask_dir(args.mask_dir, args.recon_dir.name)

    multi = len(samples_to_render) > 1

    for rank, sample_key in samples_to_render:
        _render_one_sample(
            args=args,
            method_files=method_files,
            metrics_per_sample_csv=metrics_per_sample_csv,
            gt_dir=gt_dir,
            mask_dir=mask_dir,
            sample_key=sample_key,
            rank=rank,
            multi=multi,
            auto_info=auto_info,
        )


def _render_one_sample(
    args: argparse.Namespace,
    method_files: list[Path],
    metrics_per_sample_csv: Path,
    gt_dir: Path,
    mask_dir: Path,
    sample_key: str,
    rank: int | None,
    multi: bool,
    auto_info: dict | None,
) -> None:
    """Load GT/mask/predictions for one sample key, assemble the figure, and print the result summary."""
    if multi or rank not in (None, 1):
        print(f"\n===== rank={rank} | sample={sample_key} =====")

    gt_file = gt_dir / sample_key
    if not gt_file.exists():
        raise FileNotFoundError(f"GT file not found for sample key: {gt_file}")

    mask_file = mask_dir / sample_key
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found for sample key: {mask_file}")

    gt_vol = _load_gt_volume(gt_file)
    mask_vol = _load_mask_volume(mask_file, target_shape=gt_vol.shape)

    pred_by_method: list[dict] = []
    skipped: list[str] = []
    for method_h5 in method_files:
        method_name = method_h5.stem
        try:
            pred_vol = _load_pred_volume(method_h5, key=sample_key)
            if pred_vol.shape != gt_vol.shape:
                raise ValueError(f"shape mismatch pred={pred_vol.shape}, gt={gt_vol.shape}")

            pred_by_method.append(
                {
                    "name": method_name,
                    "pred_vol": pred_vol,
                }
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{method_h5.name}: {exc}")

    if not pred_by_method:
        raise RuntimeError("No valid methods to visualize after filtering/skipping.")

    out_path = _resolve_out_path(args, n_methods=len(method_files), rank=rank, multi=multi)

    sample_metrics = _load_sample_metrics_for_key(metrics_per_sample_csv, sample_key)

    build_figure(
        gt_vol=gt_vol,
        mask_vol=mask_vol,
        pred_by_method=pred_by_method,
        sample_metrics=sample_metrics,
        out_path=out_path,
        sample_key=sample_key,
        frame_idx=args.frame_idx,
        error_cmap=args.error_cmap,
        strip_ratio=args.strip_ratio,
        strip_gap_ratio=args.strip_gap_ratio,
        crosshair_x=args.crosshair_x,
        crosshair_y=args.crosshair_y,
        show_ht=args.show_ht,
        show_wt=args.show_wt,
        recon_vmax=args.recon_vmax,
        subplot_wspace=args.subplot_wspace,
        subplot_hspace=args.subplot_hspace,
    )

    print(f"Saved: {out_path}")
    print(f"Sample key: {sample_key}")
    if rank is not None:
        print(f"Rank (top2_gap): {rank}")
    if args.auto_sample == "none":
        print(f"Sample idx: {args.sample_idx}")
    if auto_info is not None:
        item = auto_info["ranked"][rank - 1] if rank is not None else auto_info["ranked"][0]
        print(
            f"Auto sample mode ({args.auto_sample}): "
            f"A={auto_info['method_a']} (psnr={item['psnr_a']:.6f}), "
            f"B={auto_info['method_b']} (psnr={item['psnr_b']:.6f}), "
            f"A-B={item['gap']:.6f}"
        )
    print(f"Methods visualized: {len(pred_by_method)}")
    print("Method order (left to right):")
    for item in pred_by_method:
        print(f"  - {item['name']}")
    if skipped:
        print("Skipped methods:")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
