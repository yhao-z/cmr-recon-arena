# Evaluation: metrics & figures

Read `environment.md` first. Metrics compare a `reconstructions/` directory
against the ground-truth dataset.

## Metrics — `scripts/calc_metric.py`

```bash
python scripts/calc_metric.py --recon-dir <dir_with_h5> --gt-root <gt_split_dir> --force
```

- `--recon-dir` — a directory holding `<model>_reconstructions.h5` files.
- `--gt-root` — **the directory that literally contains the GT `.h5` files**, one
  command per split. The script resolves every recon key as `gt_root / key` with
  *no* extra level appended, so pointing it at a dataset parent (`…/processed_ocmr`)
  instead of the split dir makes every sample skip with `gt_file_not_found`.
- Each GT file must contain `reconstruction_weighted` **and** `mask`.
- `--force` — recompute; without it the run appends incrementally to the existing
  `metrics_per_sample.csv` (rows from an earlier, different run stay otherwise).

Ground-truth directory per split:

| dataset | `--gt-root` | has `reconstruction_weighted`? |
|---------|-------------|-------------------------------|
| OCMR | `<data_root>/OCMR/processed_ocmr/<split>` | yes |
| CMRxRecon23 pse (04/08/10) | `<data_root>/CMRxRecon/processed_cmrxrecon23/cmrxrecon23_test_fs` | **only in `test_fs`** |

The pse test splits store just `kspace` and `mask` — they are prospectively
undersampled and carry no reference image, so all three must be scored against
the fully-sampled `cmrxrecon23_test_fs` files. That works because the pipeline
drops the `FullSample`/`AccFactorXX` path segment from file names: a pse recon
key and its fs counterpart are the same name (1090/1090 match).

Outputs, written next to the inputs:

| file | content |
|------|---------|
| `metrics_per_sample.csv` | one row per sample × method × split |
| `metrics_by_method.csv` | aggregated per method (PSNR / SSIM / HFEN) |
| `metrics_summary.json` | machine-readable summary |
| `summary_table.csv` | cross-split comparison table (parent dir) |

The parent-level `summary_table.csv` (one row per method, one column group per
acceleration) is only written when the *parent* directory is processed. Since
each split usually needs its own `--gt-root`, the practical path is: run every
split on its own, then re-run the parent **without** `--force` — every split
already has its CSVs, so computation is skipped and only the summary is built:

```bash
python scripts/calc_metric.py --recon-dir reconstructions/OCMR-lowacc
```

Then refresh the ranking page with `python scripts/build_leaderboard.py`.
It scans every `reconstructions/<group>/` (using `summary_table.csv`, falling
back to the per-split `metrics_by_method.csv`), embeds the numbers as JSON and
writes a self-contained `leaderboard.html` at the repository root: one tab per
group, PSNR/SSIM/HFEN switch, click a header to sort by that acceleration
(click again to reverse), best values highlighted, `weights/` presence shown
per method. No server needed -- open the file directly. Re-run the script
after any metric refresh; the page is a snapshot, it does not read live data.

Metric implementations live in `helpers/metrics.py` (PSNR, SSIM via
scikit-image, HFEN) — reuse them rather than reimplementing.

**Group keys must match GT file names.** Recon keys mirror the test dataset's
file names (see `TESTING.md`); evaluating against a differently-named dataset
build yields zero matches.

## Figures — `scripts/visualize_recon_methods_with_gt_error.py`

```bash
python scripts/visualize_recon_methods_with_gt_error.py \
  --recon-dir <recon_dir> --gt-dir <gt> --mask-dir <mask> \
  --auto-sample 10 --ranks 3
```

- Consumes the CSVs above for method ordering (`metrics_by_method.csv`) and sample selection (`--auto-sample` reads `metrics_per_sample.csv`).
- Writes one figure per sample next to the recon files, suffixed with the rank slot; `--show-ht` adds time/frequency strips.
- Run `--help` for the full flag list — several defaults are concrete paths, so an omitted flag still has an effect.

## Reporting numbers

- Quote PSNR in **dB**, SSIM in 0–1, HFEN ×10⁻³ — and say which splits/accelerations were pooled.
- Prefer per-split columns (e.g. 6×/8×/10×) over a single mean when protocols differ.
- When comparing methods, always evaluate them in one `calc_metric.py` pass so they share GT resolution and metric versions.
