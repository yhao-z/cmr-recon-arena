"""SwanLabLogger variant that names Lightning's run directories.

Lightning derives the run directory as ``<save_dir>/<name>/<version>``. The
stock SwanLabLogger hard-codes ``name`` to its integration class name and
returns the bare swanlab run id for ``version``, so runs land in::

    <save_dir>/swanlab-integration-pytorch-lightning/<runid>/

This subclass maps both onto the run's own labels instead::

    <save_dir>/<project>/<experiment_name>-<runid>/

It reuses the ``project`` / ``experiment_name`` init args the stock logger
already accepts, so configs only swap their ``class_path``::

    logger:
      class_path: pl_modules.swanlab_logger.ProjectSwanLabLogger
      init_args: { save_dir: experiments, project: my-proj, experiment_name: my-run }

Depends on the swanlab 0.10 internals ``_init_kwargs`` (where __init__ stores
non-None arguments) and ``_get_active_run()`` (the active run carrying its
``id``); both are pinned by requirements.txt.
"""
from __future__ import annotations

from typing import Optional

from swanlab.integration.pytorch_lightning import SwanLabLogger as _SwanLabLogger


class ProjectSwanLabLogger(_SwanLabLogger):
    """Lightning logger laying out runs as ``save_dir/<project>/<experiment_name>-<runid>``."""

    @property
    def name(self) -> str:
        """First directory level: the swanlab project."""
        project = self._init_kwargs.get("project")
        return str(project) if project else super().name

    @property
    def version(self) -> Optional[str]:
        """Second level: ``experiment_name`` prefixed to the swanlab run id.

        Returns None while no run is active, exactly like the stock logger,
        so Lightning applies the same fallback behaviour at early calls.
        """
        label = self._init_kwargs.get("experiment_name")
        run = self._get_active_run()
        run_id = getattr(run, "id", None) if run is not None else None
        if run_id is None:
            return None
        return f"{label}-{run_id}" if label else str(run_id)
