# Training

Read `environment.md` first and use its `repo_root` / `env_activate` / `gpu` /
`run_policy` values for every command below.

## Step 1 — pick two configs

Configs are split; both are required and are deep-merged (later file wins per key):

- **data config** `configs/data/<dataset>.yaml` — only a `data:` section
- **model config** `configs/model/<model>.yaml` — `seed_everything`, `trainer`, `model` (no `data:`)

| data config | protocol |
|-------------|----------|
| `ocmr_lowacc.yaml` | OCMR, accelerations 6–10 (GRO) |
| `ocmr_highacc.yaml` | OCMR, accelerations 18–22 |
| `ocmr_test_lowacc.yaml` / `ocmr_test_highacc.yaml` | OCMR test splits |
| `cmrxrecon23.yaml` | CMRxRecon23, accelerations 4/8/10 |
| `cmrxrecon23_test.yaml` | CMRxRecon23 test splits |

Model configs: one per model under `configs/model/` (see the model table in
`SKILL.md`). Commented-out `# data:` lines inside them document the data config
they were last run with.

## Step 2 — logger / tracking

Model configs default to `SwanLabLogger`. Before launching, check the user's
`tracking` value in `environment.md` and adjust `trainer.logger` accordingly:

- cloud/local SwanLab account → keep `class_path: swanlab.integration.pytorch_lightning.SwanLabLogger`, set `project`, `experiment_name`, `description` (write the description from model + dataset + key hyperparameters), `mode: cloud|local`.
- no account → swap the block for `lightning.pytorch.callbacks`-independent logging, e.g. `class_path: lightning.pytorch.loggers.CSVLogger` with `save_dir: experiments`.
- never pass `--trainer.logger false` on the CLI: with the LR-monitor callback wired in this repo it raises at runtime.

Ask the user to confirm `project` / `experiment_name` when the tracking backend
is a shared service.

## Step 3 — launch

```bash
cd <repo_root> && <env_activate> && <gpu> \
python main.py fit --config configs/data/<data>.yaml --config configs/model/<model>.yaml
```

- Data config first, model config second.
- Single field overrides: append `--data.init_args.num_workers 8`, `--trainer.max_epochs 50`, …
- Resume: add `--ckpt_path <path>/last.ckpt`.

Per `run_policy`: execute directly, or print the command and wait.

## Step 4 — watch the first minutes

Wait for a few batches; confirm loss is finite and decreasing and no exception
appears, then leave it running (background per `long_task`). Report where the
logs/metrics are going per `tracking`.

## Scheduling several runs

1. Read free memory: `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv` (on the execution host from `environment.md`).
2. Free memory ≥ ~15–25 GB counts as one free slot — judge by memory, not by utilization %.
3. ≥2 slots → run two jobs in parallel with logs redirected; 1 slot → one script running the jobs sequentially; 0 slots → tell the user and wait.

```bash
<gpu0> python main.py fit --config configs/data/<d>.yaml --config configs/model/<A>.yaml > experiments/log_<A>.txt 2>&1 &
<gpu1> python main.py fit --config configs/data/<d>.yaml --config configs/model/<B>.yaml > experiments/log_<B>.txt 2>&1 &
```

## Config shape

```yaml
# configs/model/<model>.yaml
seed_everything: 3407
trainer:
  accelerator: gpu
  devices: 1
  max_epochs: 150
  logger:
    class_path: swanlab.integration.pytorch_lightning.SwanLabLogger
    init_args: { save_dir: experiments, project: <name>, experiment_name: <name>, mode: cloud }
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args: { save_last: true, verbose: true }
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args: { logging_interval: epoch }

model:
  class_path: pl_modules.<ModuleClass>
  init_args: { lr: 0.0002, loss_fns: [ssim], ssim_mode: 3d }
```

```yaml
# configs/data/<dataset>.yaml
data:
  class_path: pl_modules.FastMriDataModule
  init_args:
    data_path: <data_root>/OCMR/processed_ocmr
    coil_sens_path: <data_root>/OCMR/processed_ocmr/csm
    data_source: ocmr
    batch_size: 1
    num_workers: 8
    train_transform:
      class_path: helpers.data.transforms.VarNetDataTransform
      init_args:
        mask_func: { class_path: helpers.data.subsample.GROMaskFunc, init_args: { accelerations: [6,7,8,9,10] } }
        use_seed: false
        random_flips: true
```

## Checkpoints & habits

- Trained checkpoints go under the logger's `save_dir` — `experiments/<project>/<version>/checkpoints/` by default, since these configs set no `dirpath`. `save_last: true` keeps `last.ckpt` there.
- `weights/` (gitignored) is where ready-made checkpoints are *stored*, not written — point `--ckpt_path` at it when testing.
- `dataset_cache.pkl` is dataset-specific — delete it after switching `data_source`.
- DDP: `distributed_sampler: true` in the data module + `strategy: ddp` in the trainer.
- Commit after every code change with a one-line English message; never commit checkpoints, `reconstructions/`, or logs (already gitignored).
