# Testing

Read `environment.md` first.

## Command

```bash
cd <repo_root> && <env_activate> && <gpu> \
python main.py test \
  --config configs/data/<data_test>.yaml \
  --config configs/test.yaml \
  --ckpt_path <ckpt> \
  --model.class_path=<pl_modules.XxxModule> \
  --model.init_args.model_name=<Method> \
  --model.init_args.test_savedir=reconstructions/<test set>

Test-side data configs: `ocmr_test_lowacc.yaml` (test_gro_06/08/10),
`ocmr_test_highacc.yaml` (test_gro_18/20/22), `cmrxrecon23_test.yaml`
(test_pse_04/08/10 - only the prospective splits; see GT pairing in
`EVALUATION.md`).

Why these flags: `configs/test.yaml` provides a **logger-free** trainer
skeleton plus a worked model entry — it must be the last `--config`. Never
substitute `configs/model/<model>.yaml` here: those are training configs and
bring `SwanLabLogger` + `ModelCheckpoint`, which would open a pointless
experiment record that stays empty after the run (tests log nothing).
`--model.class_path` names the module the checkpoint was trained with (one of
the eight `pl_modules.*Module` classes; `scripts/run_benchmark.py` reads it
from the model yaml automatically), and the three `init_args`/`ckpt_path`
values decide output location, h5 file naming and weights. `configs/test.yaml`
ships with example values only — always override them.

Tests create no swanlab record at all; swanlab runs belong to `fit`.

## What comes out

- Reconstructions are written by `pl_modules/mri_module.py` during
  `test` (method `on_test_batch_end`) into
  `<test_savedir>/<model_name>_reconstructions.h5`.
- **Inside that h5, each sample is a dataset/group keyed by the source
  dataset's file name** — e.g. `TestSet_P001_cine_sax_slice00.h5`. The key
  mirrors whatever the test dataset's file names are, so results from two
  differently-named dataset builds will not line up.
- Per-batch metrics are aggregated into the run's metric tables (see
  `EVALUATION.md`).

## Where checkpoints come from

`weights/` is gitignored, so a fresh clone has none. `scripts/run_benchmark.py`
fills them from the Hugging Face repo (`yhao-z/cmr-recon-arena`, files laid out
as `<group>/<method>.ckpt`): it lists the repo, downloads only what is missing
locally, respects `$HF_ENDPOINT`, and falls back to `https://hf-mirror.com`
when `huggingface.co` does not answer within 4s (mainland networks). Flags:
`--no-download` (local files only), `--hf-repo`, `--hf-endpoint`.

## Picking the checkpoint

- Trained with `ModelCheckpoint(save_last: true)` → `<ckpt_dir>/last.ckpt`.
- For ablations/final numbers, pin the exact epoch file rather than `last.ckpt`.
- The module's hparams are restored from the checkpoint — do not re-specify
  architecture arguments on the CLI.

## DDP / multi-GPU test

Single-device test is the default and the normal path. If the user asks for
multi-GPU test, set `trainer.devices` and `distributed_sampler: true`
consistently with training; sample keys must stay unique across ranks.

## One-shot: every checkpoint, metrics and the leaderboard

```bash
python scripts/run_benchmark.py --dry-run   # plan: which runs are missing
python scripts/run_benchmark.py             # tests -> calc_metric -> summaries -> page
```

Iterates everything under `weights/<group>/`, so a fresh clone with weights
but no `reconstructions/` ends in a finished `leaderboard.html`. Resume-safe:
a (group, method) run is skipped when its split files already carry one key
per test sample, and **symlinked reconstructions are treated as complete and
never written through**. Options: `--groups/--methods` (subset), `--data-root`
(default `$YHAO_DATA`), `--num-workers`, `--skip-test`, `--skip-metrics`,
`--dry-run`. Failures are collected per run and the script keeps going; the
final report lists what to retry (re-running the same command only redoes
missing work).

## After testing

1. Confirm placement **and** completeness: one `<Method>_reconstructions.h5`
   per tested weight inside `reconstructions/<group>/<split>/`, each holding
   one key per expected sample (`h5py` key count == dataset file count). A
   `BENCHMARK COMPLETE` line only means the run exited cleanly — output sitting
   under a bare `<split>/` directory at the reconstructions root is the
   signature of a stale `test_savedir` inside a checkpoint's hparams
   redirecting writes to the fallback path (the ignore list in
   `MriModule.__init__` keeps new checkpoints clean). Move such files into the
   group directory and re-score that group.
2. Compute metrics: see `EVALUATION.md`.
3. Commit code changes; leave `reconstructions/` untracked (gitignored).
