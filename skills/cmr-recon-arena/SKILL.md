---
name: cmr-recon-arena
description: Run the cmr-recon-arena MRI repo for training, testing and new models.
version: 0.1.0
author: yhao, Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    editorial_name: cmr-recon-arena MRI Reconstruction
    editorial_description: First-run setup protocol plus training, testing, data building and model-extension guidance for the cmr-recon-arena repository.
    tags: [MRI, reconstruction, PyTorch-Lightning, onboarding]
    related_skills: []
---

# cmr-recon-arena Skill

cmr-recon-arena is a 2D+T cine MRI reconstruction repository (PyTorch Lightning + LightningCLI) covering the full loop: dataset building → training → testing → metric evaluation → adding a model. This skill carries the repo map, the task routing, and the first-run setup protocol that asks you for machine-specific values and stores them in `references/environment.md`.

## When to Use

- Setting up cmr-recon-arena on a new machine or for a new user (first run — see protocol below)
- Building a dataset from raw OCMR / CMRxRecon2023 inputs
- Training, resuming or testing a reconstruction model from `configs/`
- Computing PSNR/SSIM/HFEN or rendering recon-vs-GT comparison figures
- Adding a model of your own to the repo

Don't use for: code outside this repository, non-MRI projects, or experiments that live in a different Lightning project.

## ⚑ First-Run Protocol (mandatory)

Before running any command, read `references/environment.md`. Its first line is the gate:

- `STATUS: UNINITIALIZED` → **you must ask the user first.** Ask all six questions in one round (do not drip them across turns), using the wording below. Then write the answers into that file's table, keep the field names, and set the first line to `STATUS: INITIALIZED <YYYY-MM-DD>`. Append ready-to-copy commands (train / test) built from those answers to its `## Ready-to-run commands` section.
- `STATUS: INITIALIZED …` → use those values for every command; do not ask again. Only re-ask if the file is missing or the user says the environment changed (then update it in place).

Questions to ask on first run:

1. **Code location** — where is this repository cloned (absolute path), and do you run locally or on a remote machine/SSH host?
2. **Data location** — which directory holds the raw and preprocessed datasets? (The repo defaults to `data/OCMR` and `data/CMRxRecon` relative to the repo root.)
3. **Environment** — conda env or venv name, the activation command, Python/CUDA versions.
4. **GPU** — how many GPUs, how you pick a card (`CUDA_VISIBLE_DEVICES`), and whether long jobs go into tmux/screen or plain background.
5. **Tracking** — SwanLab (cloud / local / offline / no account), TensorBoard, or no tracking at all.
6. **Run policy** — should I execute training directly, or print the command for your approval first?

## Reference Routing

| Task | Reference |
|------|-----------|
| First run / machine setup | `references/environment.md` |
| Training a model | `references/TRAINING.md` |
| Testing & checkpoint inference | `references/TESTING.md` |
| Dataset building (OCMR, CMRxRecon) & data format | `references/DATA.md` |
| Metrics & figures | `references/EVALUATION.md` |
| Adding a new model | `references/NEW_MODEL.md` |

## Project Structure

```
cmr-recon-arena/
├── main.py                     # LightningCLI entry: fit / validate / test / predict
├── configs/
│   ├── data/<dataset>.yaml     # data: section only (paths, mask, accelerations)
│   ├── model/<model>.yaml      # seed + trainer + model (no data:)
│   └── test.yaml               # shared test overrides
├── preprocessing/              # dataset building — start here for new data
│   ├── run_ocmr.py             #   CLI entry points (first layer)
│   ├── run_cmrxrecon23.py      #
│   ├── core/                   #   per-dataset implementations
│   │   ├── ocmr.py             #     OCMRPreprocessing (reads ISMRMRD .h5)
│   │   ├── cmrxrecon23.py      #     CMRxRecon23Preprocessing (reads Challenge .mat)
│   │   └── scan_cmrxrecon_mat_csv.py
│   └── data_attributes/        #   metadata CSVs (ocmr.csv, cmrxrecon23cine.csv)
├── models/                     # model implementations (one file or one dir each)
│   ├── cinevn/                 #   VarNet: varnet, unet, conv
│   ├── mol/                    #   MOL-DEQ
│   ├── MoDL.py  CST_Net.py  ISTA_Net_plus.py
│   ├── JotlasNet.py  LplusS_Net.py  T2LR_Net.py
├── pl_modules/                 # Lightning wrapper per model + data_module + mri_module base
├── helpers/                    # utilities + data/ subpackage (SliceDataset, transforms, masks, samplers)
├── scripts/
│   ├── calc_metric.py          # PSNR/SSIM/HFEN over reconstructions/
│   └── visualize_recon_methods_with_gt_error.py
├── requirements.txt
├── weights/                    # checkpoints (gitignored)
└── reconstructions/            # test outputs + metrics (gitignored)
```

Layering rule: `preprocessing/` holds dataset-specific pipelines (entries in the first layer, implementations in `core/`); `helpers/` holds only generic MRI / I/O code and knows nothing about specific datasets; a model must never import from another model.

## Models

| Model | `pl_modules` class | Source |
|-------|--------------------|--------|
| CineVN / VarNet | `VarNetModule` | marcvornehm/CineVN + fastMRI VarNet |
| MoDL | `MoDLModule` | bo-10000/MoDL_PyTorch |
| MOL-DEQ | `MOLModule` | Monotone-Operator-Learning |
| ISTA-Net+ | `ISTANetplusModule` | jianzhangcs/ISTA-Net-PyTorch |
| L+S-Net | `LplusSNetModule` | wenqihuang/LS-Net-Dynamic-MRI |
| T2LR-Net | `T2LRNetModule` | yhao-z/T2LR-Net |
| JotlasNet | `JotlasNetModule` | yhao-z/JotlasNet |
| CST-Net | `CSTNetModule` | XWangBin/CST-Net |

Each model's own `Reference:` URL lives in its file header.

## Quick Reference

All commands assume the values from `references/environment.md`.

```bash
# build a dataset (see references/DATA.md)
python preprocessing/run_ocmr.py            # OCMR   (defaults target data/OCMR/processed_ocmr)
python preprocessing/run_cmrxrecon23.py     # CMRxRecon2023 (defaults target data/CMRxRecon/processed_cmrxrecon23)

# train: data config first, model config second (deep merge, later wins per key)
python main.py fit --config configs/data/<data>.yaml --config configs/model/<model>.yaml

# test: checkpoint is mandatory, there is no default
python main.py test --config configs/data/<data_test>.yaml --config configs/test.yaml \
  --ckpt_path <ckpt> --model.class_path=<pl_modules.XxxModule> \
  --model.init_args.model_name=<Method> --model.init_args.test_savedir=reconstructions/<group>

# one-shot: every missing test run -> metrics -> summaries -> leaderboard
# (resume-safe, never writes through symlinks; see references/TESTING.md)
python scripts/run_benchmark.py --dry-run
python scripts/run_benchmark.py

# rebuild the static leaderboard page from those CSVs (-> leaderboard.html)
python scripts/build_leaderboard.py

# metrics: one command per split; --gt-root is the dir that literally holds the GT h5
# (CMRxRecon pse splits: point it at cmrxrecon23_test_fs — see references/EVALUATION.md)
python scripts/calc_metric.py --recon-dir <dir_with_h5> --gt-root <gt_split_dir> --force
```

`--config` takes exactly one file per flag; `--config a b` is rejected.

## Pitfalls

- `STATUS: UNINITIALIZED` in `references/environment.md` — never skip the protocol; commands built without it will point at the wrong machine or data.
- Test needs an explicit `--ckpt_path`; Lightning will not pick one for you.
- `dataset_cache.pkl` is dataset-specific: after switching `data_source`, delete the stale cache if the loader still reports the previous dataset's shapes.
- `num_slc = -1` in a batch means a single-slice volume (no temporal dimension) — models handle it, but shapes differ from cine batches.
- Switching datasets = swap only the first `--config`; a single field can be overridden on the CLI (e.g. `--data.init_args.num_workers 8`).
- The mask type is chosen by `class_path` inside the data yaml (`helpers.data.subsample.GROMaskFunc`, `RandomMaskFunc`, `RandomACSMaskFunc`, `VISTAMaskFunc`, `CAVAMaskFunc`, `EquispacedMaskFunc`, `EquispacedACSMaskFunc`, `StaticEquispacedACSMaskFunc`) — there is no string-dispatch helper anymore.
- CG-DC lives in `helpers/mri_torch/` (`cgdc.py` = k-space RHS for VarNet, `cgdc_img.py` = image RHS for MoDL). Models must not import each other's code.
- Relative `data_path` in a config resolves against `YHAO_DATA` when that env var is set; otherwise against the working directory — prefer absolute paths in `environment.md`.

## Verification

First run done right:

- `references/environment.md` starts with `STATUS: INITIALIZED <date>` and every table row has a value.
- `python main.py --help` lists `fit, validate, test, predict`.
- A smoke run (`--trainer.fast_dev_run 1` on a 1-GPU config) finishes one batch without a path or import error.
- After any edit: `python -m py_compile` on changed files, and commit with a one-line English message.

## Installing this skill into an agent

- **Claude Code**: copy this directory to `.claude/skills/cmr-recon-arena/` in the repo (project scope) or `~/.claude/skills/cmr-recon-arena/`.
- **Hermes Agent**: copy to `~/.hermes/profiles/<profile>/skills/projects/cmr-recon-arena/`.
- **Anything else**: the frontmatter is plain YAML — point your agent at `SKILL.md` and let it read `references/` on demand.
