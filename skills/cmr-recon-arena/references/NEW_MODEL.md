# Adding a new model

The repo follows one pattern: a plain PyTorch module, a thin Lightning
wrapper, and two yaml files. Nothing else has to change.

## Files to create

```
models/<your_model>.py               # the network (or models/<your_model>/ as a package)
pl_modules/<your_model>_module.py    # Lightning wrapper
configs/model/<your_model>.yaml      # trainer + model init args
configs/data/<data>.yaml             # usually reuse an existing one
```

## 1. The model (`models/<your_model>.py`)

- Pure `nn.Module`: `forward(...)` takes batch tensors and returns the reconstruction (complex tensor, same layout as input).
- File header: 2–6 lines — what it implements, `Reference: <source URL>`, `Paper: <citation>` (follow `models/MoDL.py` as the template).
- Import only from `helpers/` (`mri_torch`, `mri_basic`, `losses`); **never from another model's directory**.
- Coil handling convention used across models: `sens_expand` / `sens_reduce` from `helpers.mri_torch`.

## 2. The Lightning wrapper (`pl_modules/<your_model>_module.py`)

Subclass `MriModule` (`pl_modules/mri_module.py`) — it supplies `training_step`,
`validation_step`, `test_step`, logging, and reconstruction saving:

```python
from .mri_module import MriModule

class MyModelModule(MriModule):
    def __init__(self, model_name, lr=2e-4, **kwargs):
        super().__init__(**kwargs)          # MriModule owns metric/logging setup
        self.model = MyModel(...)
```

- Do not put architecture hyperparameters into `MriModule.__init__`'s fixed set — declare them on your class; the checkpoint restores them from hparams.
- **Never call `save_hyperparameters()` in your wrapper.** `MriModule` calls it
  once for the whole inheritance tree (its own `__init__` runs inside your
  chain, so your architecture args are collected there) and ignores the runtime
  options `model_name`, `save_val_images`, `test_savedir`, `ckpt_path`.
  A second call in the subclass would *overwrite* the
  parent's and un-ignore them. Why that ignore list matters: jsonargparse
  applies checkpoint hparams on top of CLI values, so a checkpoint storing
  `test_savedir: None` silently undoes `--model.init_args.test_savedir` at test
  time (outputs landing in the wrong directory). If your model adds a test-only
  switch, append it to the ignore list in `MriModule.__init__` instead of
  writing a local `save_hyperparameters` call.
- Keep test-time behaviour inside `MriModule.test_step` unless the model needs a custom loop.

## 3. The config (`configs/model/<your_model>.yaml`)

```yaml
seed_everything: 3407
trainer: { accelerator: gpu, devices: 1, max_epochs: 150, ... }
model:
  class_path: pl_modules.MyModelModule
  init_args: { model_name: MyModel, lr: 0.0002, loss_fns: [ssim], ssim_mode: 3d }
```

`class_path` is resolved lazily by LightningCLI — a typo only fails at
instantiation, so verify it (next step) rather than trusting the name.

## 4. Verify before training

```bash
python -m py_compile models/<your_model>.py pl_modules/<your_model>_module.py

# import + instantiate from the real config (catches class_path and init_args typos)
python -c "import yaml, importlib; \
mc = yaml.safe_load(open('configs/model/<your_model>.yaml'))['model']; \
m, c = mc['class_path'].rsplit('.', 1); \
getattr(importlib.import_module(m), c)(**mc.get('init_args', {})); print('OK')"

# one-batch smoke run
<gpu> python main.py fit --config configs/data/<data>.yaml \
  --config configs/model/<your_model>.yaml --trainer.fast_dev_run 1
```

Completion criteria: all three print/pass with no import, shape, or path error.

## 5. Train, test, evaluate

Follow `TRAINING.md` → `TESTING.md` → `EVALUATION.md`. Commit the three new
files (plus any config you touched) with a one-line English message.

## Conventions worth keeping

- One file/dir per model under `models/`; one wrapper per model under `pl_modules/`; `configs/model/` mirrors it 1:1.
- Data-specific code never goes into `models/` or `helpers/` — datasets live in `preprocessing/` and `data/`.
- Comments explain *why*, not *what*; a file header states purpose, source and callers. Skip licence/author blocks — source URLs go in `Reference:`.
