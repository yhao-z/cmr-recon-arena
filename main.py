"""Main entry point: run cmr-recon-arena training/evaluation through LightningCLI.

Usage (subcommand first, then one or more `--config` files):

    python main.py <subcommand> --config configs/data/<data>.yaml \\
        --config configs/model/<model>.yaml [--config <more>.yaml ...]

Available subcommands (printed by `python main.py --help`): `fit`, `validate`,
`test`, `predict`. The concrete classes come from the configs, e.g.
`data.class_path: pl_modules.FastMriDataModule` and
`model.class_path: pl_modules.VarNetModule`.

Config combination: every `--config` file is read here before LightningCLI
parses anything; keys are deep-merged per key with later files overriding
earlier ones, so data configs come first and model configs second.
`trainer.logger.save_dir` may come from any of those files or from the
`--trainer.logger.save_dir` CLI flag, and the directory is pre-created before
the logger starts (see `preprocess_save_dir`). Tags under
`trainer.logger.init_args.tags` are unioned across all `--config` files and
pushed to the logger experiment, and the merged config is written to
`<save_dir>/<logger name>/<version>/config.yaml` (skipped for `predict`).
The `test` subcommand deliberately uses the stock `LightningCLI`, i.e. no
custom config saving/tag merging; test reconstructions are written by
`pl_modules.mri_module.MriModule.on_test_batch_end`.
"""
import warnings
warnings.filterwarnings("ignore", message=".*_pack_.*", category=DeprecationWarning)

import os
import sys
import yaml
from typing_extensions import override

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

# os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Set multiple GPUs visible

def preprocess_save_dir():
    """Ensure `save_dir` exists, handling both command-line arguments and YAML configuration."""
    # Collect every YAML referenced by repeated `--config` options.
    # (LightningCLI merges them per-key: later files override earlier ones.)
    config_paths = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--config":
            j = i + 1
            while j < len(sys.argv) and not sys.argv[j].startswith("--"):
                config_paths.append(sys.argv[j])
                j += 1
            i = j
        else:
            i += 1

    save_dir = None  # Default to None

    if config_paths:
        for config_path in config_paths:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding='utf-8') as f:
                    try:
                        config = yaml.safe_load(f)
                        if config is not None:
                            # Safely navigate to trainer.logger.save_dir
                            trainer = config.get("trainer", {})
                            logger = trainer.get("logger", {})
                            if isinstance(logger, dict) :  # Ensure logger is a dictionary
                                yaml_save_dir = logger.get(
                                    "init_args", {}).get("save_dir")
                                if yaml_save_dir:
                                    save_dir = yaml_save_dir  # Use the first valid save_dir found
                                    break
                    except yaml.YAMLError as e:
                        print(f"Error parsing YAML file {config_path}: {e}")

    for i, arg in enumerate(sys.argv):
        if arg == "--trainer.logger.save_dir":
            save_dir = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
            break

    if not save_dir:
        print("Logger save_dir is None. No action taken.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Pre-created logger save_dir: {save_dir}")


class CustomSaveConfigCallback(SaveConfigCallback):
    '''save the config file to the logger's run directory, merge tags from different configs'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merged_tags = self._collect_tags_from_configs()

    def _collect_tags_from_configs(self):
        """Union the `trainer.logger.init_args.tags` lists of every `--config` file."""
        config_files = []
        merged_tags = set()

        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                config_files.append(sys.argv[i + 1])

        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                        if isinstance(config_data, dict):
                            logger = config_data.get('trainer', {}).get(
                                'logger', {})
                            if logger and isinstance(logger, dict):
                                tags = logger.get('init_args', {}).get('tags', [])
                                if isinstance(tags, list):
                                    merged_tags.update(tags)
                except (yaml.YAMLError, IOError) as e:
                    print(f"Warning: Error reading {config_file}: {str(e)}")
        return merged_tags

    @override
    def setup(self, trainer, pl_module, stage):
        """Inject the merged tags into the logger config/experiment, then defer to the base class."""
        if hasattr(self.config, 'trainer') and hasattr(self.config.trainer, 'logger'):
            logger_config = self.config.trainer.logger
            if hasattr(logger_config, 'init_args'):
                logger_config.init_args['tags'] = list(self.merged_tags)
                if hasattr(trainer, 'logger') and trainer.logger is not None:
                    trainer.logger.experiment.tags = list(self.merged_tags)

        super().setup(trainer, pl_module, stage)

    @override
    def save_config(self, trainer, pl_module, stage) -> None:
        """Save the configuration file under the logger's run directory."""
        if stage == "predict":
            print("Skipping saving configuration in predict mode.")
            return  
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            project_name = trainer.logger.name
            run_id = trainer.logger.version
            save_dir = trainer.logger.save_dir
            run_dir = os.path.join(save_dir, project_name, run_id)
            
            os.makedirs(run_dir, exist_ok=True)
            config_path = os.path.join(run_dir, "config.yaml")
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
            print(f"Configuration saved to {config_path}")


def run_cli():
    """Pre-create the logger dir, then hand argv to LightningCLI with/without the custom config callback.

    `test` runs the stock LightningCLI (plain config saving); every other
    subcommand uses `CustomSaveConfigCallback`. Reconstructing to disk during
    `test` is done by `MriModule.on_test_batch_end`.
    """
    preprocess_save_dir()

    # `test` subcommand: no tag merging / custom config saving (stock LightningCLI).
    # overwrite still applies: the config snapshot is a reproducible run record,
    # and consecutive test runs (scripts/run_benchmark.py) would otherwise abort
    # on the file written by the previous run.
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        cli = LightningCLI(save_config_kwargs={"overwrite": True})
    else:
        cli = LightningCLI(
            save_config_callback=CustomSaveConfigCallback,
            # save_to_log_dir=False drops SaveConfigCallback.setup()'s built-in
            # copy at <trainer.log_dir>/config.yaml (which lands in the bare
            # save_dir root before the run exists); the custom save_config
            # below writes the single snapshot inside the run directory.
            save_config_kwargs={"overwrite": True, "save_to_log_dir": False},
        )

if __name__ == "__main__":
    run_cli()
