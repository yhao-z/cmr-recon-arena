#!/usr/bin/env python3
"""One-shot benchmark: test every checkpoint, score every split, rebuild the page.

Scenario: a fresh clone whose `weights/` folder has never been tested (it is
gitignored - the plan lists checkpoints from the Hugging Face repo
`yhao-z/cmr-recon-arena`, where they live as `<group>/<method>.ckpt`, and
fetches the missing ones into `weights/` before testing; mainland networks
fall back to https://hf-mirror.com automatically). Run

    python scripts/run_benchmark.py            # everything (slow, hours)
    python scripts/run_benchmark.py --dry-run  # show the plan, touch nothing

and it will, for every (test set, method) checkpoint found under weights/:

1. run `main.py test` with the group's data config and the method's model
   config, writing reconstructions to `reconstructions/<group>/<split>/`;
2. compute per-split metrics with scripts/calc_metric.py against the correct
   ground truth (CMRxRecon prospective splits score against test_fs - they
   have no reference image of their own);
3. rebuild each group's summary_table.csv (parent-directory pass without
   --force, which only aggregates);
4. regenerate leaderboard.html via scripts/build_leaderboard.py.

Interrupt freely and re-run: a (group, method) run is skipped when all of its
split files are already there (with one key per expected sample when the
dataset is reachable to count). Symlinked results are treated as complete and
are never written through - that keeps a checkout that links to an external
reconstructions archive safe.

Data root resolution: --data-root, else $YHAO_DATA (the same variable the
data configs rely on for their relative data_path).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
RECON = ROOT / "reconstructions"

# weights/<group> -> data config providing `data:` (test_split list + data_source)
GROUP_DATA_CONFIG = {
    "OCMR-lowacc": "ocmr_test_lowacc.yaml",
    "OCMR-highacc": "ocmr_test_highacc.yaml",
    "CMRxRecon": "cmrxrecon23_test.yaml",
}

# weights file stem -> configs/model yaml (explicit: names are not mechanical)
MODEL_CONFIG = {
    "CineVN": "cinevn.yaml",
    "CST-Net": "CSTNet.yaml",
    "ISTA-Net+": "istanetplus.yaml",
    "JotlasNet": "jotlasnet.yaml",
    "L+S-Net": "l+s-net.yaml",
    "MoDL": "modl.yaml",
    "MOL": "mol.yaml",
    "T2LR-Net": "t2lrnet.yaml",
}


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def group_plan(group: str) -> dict:
    """Read the group's data config: splits, data_source, expected sample counts."""
    cfg = load_yaml(ROOT / "configs" / "data" / GROUP_DATA_CONFIG[group])
    init = cfg["data"]["init_args"]
    source = init["data_source"]
    splits = [f"{source}_{s}" for s in init["test_split"]]   # dir names, as mri_data sees them
    return {"source": source, "splits": splits, "data_path": init["data_path"]}


def gt_root_for(source: str, split: str, data_root: Path) -> Path | None:
    """Ground-truth directory for one split (None => script cannot score it)."""
    if source == "ocmr":
        return data_root / "OCMR" / "processed_ocmr" / split
    if source == "cmrxrecon23":
        # prospective splits carry only kspace+mask; the fully-sampled set is
        # the reference and shares file names (acquisition segment is dropped)
        return data_root / "CMRxRecon" / "processed_cmrxrecon23" / "cmrxrecon23_test_fs"
    return None


def count_h5(path: Path) -> int | None:
    """Number of sample keys in a dataset h5, or None if it cannot be read."""
    if not path.is_dir():
        return None
    files = [p for p in path.iterdir() if p.suffix in (".h5", ".hdf5")]
    return len(files) if files else None


def recon_status(group: str, split: str, method: str, expected: int | None) -> str:
    """done | linked | partial | missing for one output file."""
    p = RECON / group / split / f"{method}_reconstructions.h5"
    if p.is_symlink():
        return "linked"
    if not p.exists():
        return "missing"
    if expected is None:
        return "done-unverified"       # dataset unreachable: existency only
    try:
        import h5py
        with h5py.File(p, "r") as hf:
            return "done" if len(hf.keys()) >= expected else "partial"
    except Exception:
        return "partial"


def resolve_hf(explicit: str | None) -> str:
    """Endpoint choice: --hf-endpoint > $HF_ENDPOINT > probe > hf-mirror.com."""
    if explicit:
        return explicit
    env_ep = os.environ.get("HF_ENDPOINT")
    if env_ep:
        return env_ep
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co/api/models?limit=1", timeout=4)
        return "https://huggingface.co"
    except Exception:
        print("huggingface.co not reachable within 4s -> using https://hf-mirror.com "
              "(set HF_ENDPOINT or pass --hf-endpoint to override)")
        return "https://hf-mirror.com"


def hf_group_names(group: str, repo: str) -> list[str] | None:
    """Method stems the HF repo lists for one group (<group>/<method>.ckpt)."""
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo_id=repo)
    except Exception as e:
        print(f"! cannot list {repo} ({type(e).__name__}: {e}); using local weights only")
        return None
    return sorted(Path(f).stem for f in files
                  if f.startswith(f"{group}/") and f.endswith(".ckpt")
                  and "/" not in f[len(group) + 1:])


def ensure_ckpt(group: str, method: str, repo: str) -> Path:
    """Local ckpt path, downloading from HF when missing locally (cache kept)."""
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(repo_id=repo, filename=f"{group}/{method}.ckpt")
    dest = WEIGHTS / group / f"{method}.ckpt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)
    print(f"  downloaded {group}/{method}.ckpt from {repo} -> {dest.relative_to(ROOT)}")
    return dest


def run(cmd: list[str], label: str, dry: bool, env: dict) -> int:
    pretty = " ".join(cmd)
    if dry:
        print(f"  [dry] {label}: {pretty}")
        return 0
    print(f"--> {label}\n    {pretty}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode:
        print(f"!! {label} exited with {r.returncode}", flush=True)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=None, metavar="GROUP",
                    help=f"subset of test sets (default: all of {sorted(GROUP_DATA_CONFIG)})")
    ap.add_argument("--methods", nargs="+", default=None, metavar="METHOD",
                    help="subset of methods (weights file stems, e.g. 'MOL' 'MoDL')")
    ap.add_argument("--data-root", default=os.environ.get("YHAO_DATA"),
                    help="dataset root (default: $YHAO_DATA)")
    ap.add_argument("--num-workers", type=int, default=None, help="override data loader workers")
    ap.add_argument("--skip-test", action="store_true", help="only (re)compute metrics + leaderboard")
    ap.add_argument("--skip-metrics", action="store_true", help="only run the tests")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--no-download", action="store_true",
                    help="never touch the network for weights; local files only")
    ap.add_argument("--hf-repo", default="yhao-z/cmr-recon-arena",
                    help="Hugging Face repo holding weights as <group>/<method>.ckpt")
    ap.add_argument("--hf-endpoint", default=None,
                    help="HF endpoint (default: $HF_ENDPOINT, else probe huggingface.co, "
                         "falling back to https://hf-mirror.com)")
    args = ap.parse_args()

    groups = args.groups or sorted(GROUP_DATA_CONFIG)
    bad = [g for g in groups if g not in GROUP_DATA_CONFIG]
    if bad:
        ap.error(f"unknown group(s) {bad}; known: {sorted(GROUP_DATA_CONFIG)}")

    data_root = Path(args.data_root) if args.data_root else None
    if data_root is None and not args.dry_run:
        sys.exit("no data root: pass --data-root or export YHAO_DATA "
                 "(the test data configs resolve their relative data_path against it)")

    env = dict(os.environ)
    if data_root:
        env["YHAO_DATA"] = str(data_root)

    if not args.no_download:
        # must happen before the first import of huggingface_hub: the library
        # freezes HF_ENDPOINT into its constants at import time
        endpoint = resolve_hf(args.hf_endpoint)
        os.environ["HF_ENDPOINT"] = endpoint   # read by huggingface_hub at import

    failures: list[str] = []

    # ---------------- plan ----------------
    plan: list[dict] = []            # one row per (group, method)
    split_info: dict[str, dict] = {}  # group -> {"splits": [...], "source": ...}
    for group in groups:
        info = group_plan(group)
        split_info[group] = info
        names = {p.stem for p in WEIGHTS.glob(f"{group}/*.ckpt")}
        if not args.no_download:
            hf_names = hf_group_names(group, args.hf_repo)   # None = repo unreachable
            if hf_names is not None:
                names |= set(hf_names)
        if args.methods:
            names &= set(args.methods)
        weights: list[Path] = []
        for m in sorted(names):
            p = WEIGHTS / group / f"{m}.ckpt"
            if p.exists() or args.dry_run:
                weights.append(p)
                continue
            try:
                weights.append(ensure_ckpt(group, m, args.hf_repo))
            except Exception as e:
                failures.append(f"weight download failed {group}/{m}: {type(e).__name__}: {e}")
        pending = [w.stem for w in weights if not w.exists()]
        if pending:
            print(f"  {group}: {len(pending)} checkpoint(s) to fetch from "
                  f"{args.hf_repo}: {', '.join(pending)}")
        if not weights:
            print(f"! no checkpoints matched for {group}")
            continue
        for w in weights:
            statuses = {}
            for split in info["splits"]:
                expected = None
                if data_root:
                    gt = gt_root_for(info["source"], split, data_root)
                    if gt is not None:
                        expected = count_h5(gt)
                statuses[split] = recon_status(group, split, w.stem, expected)
            plan.append({"group": group, "method": w.stem, "ckpt": w, "status": statuses,
                         "from_hf": not w.exists()})

    print("=" * 74)
    print(f"BENCHMARK PLAN  ({len(plan)} checkpoints x "
          f"{sum(len(v['splits']) for v in split_info.values())} splits per full pass)")
    print("=" * 74)
    to_run = []
    for row in plan:
        st = row["status"]
        bad_splits = [s for s, v in st.items() if v in ("missing", "partial")]
        linked = [s for s, v in st.items() if v == "linked"]
        if linked and bad_splits:
            verdict = (f"REFUSE: {len(linked)} split(s) are symlinks while "
                       f"{bad_splits} are incomplete - resolve by hand")
        elif bad_splits:
            verdict = f"RUN (needs {bad_splits})"
            to_run.append(row)
        elif linked:
            verdict = f"skip (linked: {len(linked)}/{len(st)})"
        else:
            verdict = f"skip (complete: {len(st)}/{len(st)})"
        if row.get("from_hf"):
            verdict += " (ckpt: download from HF)"
        print(f"  {row['group']:<13} {row['method']:<11} {verdict}")
    print("-" * 74)

    # ---------------- stage 1: tests ----------------
    if args.skip_test:
        print("[stage 1/3] test runs skipped (--skip-test)")
    else:
        print(f"[stage 1/3] {len(to_run)} test run(s) to execute")
        for i, row in enumerate(to_run, 1):
            group, method = row["group"], row["method"]
            model_cfg = MODEL_CONFIG.get(method)
            if model_cfg is None:
                failures.append(f"{group}/{method}: no configs/model mapping for this name")
                continue
            # Test never loads configs/model/*.yaml: those are training configs
            # (SwanLabLogger + ModelCheckpoint), and starting a test run with
            # them creates an empty swanlab record. configs/test.yaml carries a
            # logger-free trainer skeleton; only the module class_path is taken
            # from the model yaml.
            try:
                class_path = load_yaml(ROOT / "configs" / "model" / model_cfg)["model"]["class_path"]
            except Exception as e:
                failures.append(f"{group}/{method}: cannot read class_path from "
                                f"configs/model/{model_cfg}: {e}")
                continue
            cmd = [sys.executable, "main.py", "test",
                   "--config", f"configs/data/{GROUP_DATA_CONFIG[group]}",
                   "--config", "configs/test.yaml",
                   "--ckpt_path", str(row["ckpt"].relative_to(ROOT)),
                   f"--model.class_path={class_path}",
                   f"--model.init_args.test_savedir={RECON.relative_to(ROOT)}/{group}",
                   f"--model.init_args.model_name={method}"]
            if args.num_workers:
                cmd.append(f"--data.init_args.num_workers={args.num_workers}")
            if run(cmd, f"test {i}/{len(to_run)} {group}/{method}", args.dry_run, env):
                failures.append(f"test failed: {group}/{method}")

    # ---------------- stage 2: metrics ----------------
    if args.skip_metrics:
        print("[stage 2/3] metrics skipped (--skip-metrics)")
    elif args.skip_test:
        # tests didn't run: aggregate whatever exists (incremental, no --force)
        for group in groups:
            info = split_info[group]
            for split in info["splits"]:
                gt = gt_root_for(info["source"], split, data_root) if data_root else None
                if gt is None:
                    if args.dry_run:
                        print(f"  [dry] metrics {group}/{split}: unknown GT path "
                              f"(no --data-root in dry-run; not an error)")
                        continue
                    failures.append(f"no data root for {group}/{split} (pass --data-root)"
                                    if data_root is None else
                                    f"no GT rule for source {info['source']}")
                    continue
                cmd = [sys.executable, "scripts/calc_metric.py",
                       "--recon-dir", f"reconstructions/{group}/{split}",
                       "--gt-root", str(gt)]
                if run(cmd, f"metrics {group}/{split}", args.dry_run, env):
                    failures.append(f"metrics failed: {group}/{split}")
    else:
        # fresh runs were mixed in: score only groups that had runs, then rebuild
        # their summaries with --force so stale rows cannot survive; untouched
        # groups only get their summary rebuilt (parent pass without --force)
        ran_groups = {r["group"] for r in to_run}
        for group in groups:
            info = split_info[group]
            fresh = group in ran_groups
            for split in info["splits"]:
                gt = gt_root_for(info["source"], split, data_root) if data_root else None
                if gt is None:
                    if args.dry_run:
                        print(f"  [dry] metrics {group}/{split}: unknown GT path "
                              f"(no --data-root in dry-run; not an error)")
                        continue
                    failures.append(f"no data root for {group}/{split} (pass --data-root)"
                                    if data_root is None else
                                    f"no GT rule for source {info['source']}")
                    continue
                cmd = [sys.executable, "scripts/calc_metric.py",
                       "--recon-dir", f"reconstructions/{group}/{split}",
                       "--gt-root", str(gt)]
                if fresh:
                    cmd.append("--force")
                if run(cmd, f"metrics {group}/{split}", args.dry_run, env):
                    failures.append(f"metrics failed: {group}/{split}")
            # parent pass: builds/refreshes summary_table.csv, skips recompute
            if run([sys.executable, "scripts/calc_metric.py",
                    "--recon-dir", f"reconstructions/{group}"],
                   f"summary {group}", args.dry_run, env):
                failures.append(f"summary failed: {group}")

    # ---------------- stage 3: leaderboard ----------------
    if args.skip_metrics:
        print("[stage 3/3] leaderboard skipped (--skip-metrics)")
    elif run([sys.executable, "scripts/build_leaderboard.py"], "leaderboard", args.dry_run, env):
        failures.append("leaderboard build failed")

    # ---------------- report ----------------
    print("=" * 74)
    if failures:
        print(f"FINISHED WITH {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        print("Re-run the same command to retry only what is missing.")
        return 1
    print("BENCHMARK COMPLETE" + (" (dry-run)" if args.dry_run else "") +
          " -> reconstructions/, summaries, leaderboard.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
