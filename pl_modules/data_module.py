"""
LightningDataModule building the train/val/test DataLoaders of the fastMRI-style dataset.
Defines FastMriDataModule and worker_init_fn; loaded by main.py through the class_path
pl_modules.FastMriDataModule in configs/data/*.yaml.
"""

import functools
import os
from pathlib import Path
from typing import Callable

import lightning as L
import torch
import torch.distributed as dist
import torch.utils.data

from helpers.data import ClusteredBatchSampler, SliceDataset, VolumeSampler


def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info is not None
    data: SliceDataset = worker_info.dataset  # type: ignore

    # Check if we are using DDP
    is_ddp = False
    if dist.is_available() and dist.is_initialized():
        is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if data.transform.mask_func is not None:  # type: ignore
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + dist.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2 ** 32 - 1))  # type: ignore


class FastMriDataModule(L.LightningDataModule):
    """
    Data module class for fastMRI-like data sets.

    This class handles configurations for training on fastMRI data. It is set
    up to process configurations independently of training modules.

    Note that transform and mask configurations are expected to be done by the
    YAML configs (LightningCLI class_path / init_args) and passed into this data module.

    For training with ddp be sure to set distributed_sampler=True: validation
    volumes then stay on a single rank (VolumeSampler, batch_size == 1), while
    with batch_size > 1 the ClusteredBatchSampler shards shape-homogeneous batches.
    """

    def __init__(
            self,
            data_path: str | Path = 'data',
            coil_sens_path: str | Path = 'coil_sens',
            train_transform: Callable | None = None,
            val_transform: Callable | None = None,
            test_transform: Callable | None = None,
            data_source: str | None = None,
            train_source: str | None = None,
            val_source: str | None = None,
            test_source: str | None = None,
            test_split: str | list[str] = 'test',
            sample_rate: float | None = None,
            volume_sample_rate: float | None = None,
            use_dataset_cache_file: bool = True,
            dataset_cache_file: str | Path = 'dataset_cache.pkl',
            batch_size: int = 1,
            num_workers: int = 4,
            distributed_sampler: bool = False,
            num_espirit_maps: int = 1,
    ):
        """
        Args:
            data_path: Path to root data directory.
            coil_sens_path: Path to coil sensitivity maps.
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            data_source: Name of data set source for all datasets.
            train_source: Name of data set source for training (overwrites
                data_source for this dataset)
            val_source: Name of data set source for validation (overwrites
                data_source for this dataset)
            test_source: Name of data set source for testing (overwrites
                data_source for this dataset)
            test_split: Name of test split.
            sample_rate: Fraction of slices of the training data split to use.
                Can be set to less than 1.0 for rapid prototyping. If not set,
                it defaults to 1.0. To subsample the dataset either set
                sample_rate (sample by slice) or volume_sample_rate (sample by
                volume), but not both.
            volume_sample_rate: Fraction of volumes of the training data split
                to use. Can be set to less than 1.0 for rapid prototyping. If
                not set, it defaults to 1.0. To subsample the dataset either set
                sample_rate (sample by slice) or volume_sample_rate (sample by
                volume), but not both.
            use_dataset_cache_file: Whether to cache dataset metadata. This is
                very useful for large datasets like the brain data.
            dataset_cache_file: File location for dataset cache.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This
                should be set to True if training with ddp.
            num_espirit_maps: The number of sets of ESPIRiT coil sensitivity
                maps to use (Soft-SENSE). Usually 1 or 2. Ignored if
                coil_sens_path is None.
        """
        super().__init__()

        if data_source is None and (train_source is None or val_source is None or test_source is None):
            raise ValueError('Either data_source or train_source, val_source, and test_source must be set.')

        # Resolve data paths: if YHAO_DATA env var is set, prepend it to relative paths
        yhao_data = os.environ.get('YHAO_DATA', None)
        self.data_path = Path(data_path)
        self.coil_sens_path = Path(coil_sens_path)
        if yhao_data and not self.data_path.is_absolute():
            self.data_path = Path(yhao_data) / self.data_path
        if yhao_data and not self.coil_sens_path.is_absolute():
            self.coil_sens_path = Path(yhao_data) / self.coil_sens_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.train_source = train_source if train_source is not None else data_source
        self.val_source = val_source if val_source is not None else data_source
        self.test_source = test_source if test_source is not None else data_source
        self.test_split = test_split
        self.sample_rate = sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.use_dataset_cache_file = use_dataset_cache_file
        self.dataset_cache_file = dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.num_espirit_maps = num_espirit_maps

    def _normalize_test_splits(self) -> list[str]:
        if isinstance(self.test_split, (list, tuple)):
            return [str(x) for x in self.test_split]
        return [str(self.test_split)]

    def _build_test_dataset(self, data_transform: Callable) -> SliceDataset:
        source = self.test_source
        assert source is not None
        coil_sens_path = self.coil_sens_path / source
        splits = self._normalize_test_splits()
        roots = [self.data_path / f'{source}_{split}' for split in splits]
        return SliceDataset(
            roots,
            coil_sens_path,
            transform=data_transform,
            sample_rate=1.0,
            volume_sample_rate=None,
            use_dataset_cache=self.use_dataset_cache_file,
            dataset_cache_file=self.dataset_cache_file,
            num_espirit_maps=self.num_espirit_maps,
        )

    def _create_data_loader(self, data_transform: Callable, data_partition: str) -> torch.utils.data.DataLoader:
        if data_partition == 'train':
            is_train = True
            sample_rate = self.sample_rate
            volume_sample_rate = self.volume_sample_rate
        else:
            is_train = False
            sample_rate = 1.0
            volume_sample_rate = None  # default case, no subsampling

        if data_partition == 'train':
            source = self.train_source
        elif data_partition == 'val':
            source = self.val_source
        else:
            source = self.test_source
        assert source is not None
        if data_partition == 'test':
            dataset = self._build_test_dataset(data_transform)
        else:
            data_path = self.data_path / f'{source}_{data_partition}'
            coil_sens_path = self.coil_sens_path / source
            dataset = SliceDataset(
                data_path,
                coil_sens_path,
                transform=data_transform,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                use_dataset_cache=self.use_dataset_cache_file,
                dataset_cache_file=self.dataset_cache_file,
                num_espirit_maps=self.num_espirit_maps,
            )

        sampler = None
        if self.batch_size > 1:
            # ensure that batches contain only samples of the same size
            sampler = ClusteredBatchSampler(
                dataset, batch_size=self.batch_size, shuffle=is_train, distributed=self.distributed_sampler,
            )
        elif self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                # ensure that entire volumes go to the same GPU in the ddp setting
                sampler = VolumeSampler(dataset, shuffle=False)

        # Own generator: batch order and the per-worker mask seeds (worker_init_fn reads
        # worker_info.seed, which is derived from the loader's base_seed) become a pure
        # function of the epoch index instead of depending on how much global RNG the model
        # consumed (phase_cycling draws random phases once per VN-block iteration and thus
        # advances the global RNG). Seeded from the configured seed, not from entropy.
        generator = torch.Generator().manual_seed(torch.initial_seed())
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size if not isinstance(sampler, ClusteredBatchSampler) else 1,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler if not isinstance(sampler, ClusteredBatchSampler) else None,
            batch_sampler=sampler if isinstance(sampler, ClusteredBatchSampler) else None,
            shuffle=is_train if sampler is None else False,
            generator=generator,
        )

        return dataloader

    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        if self.use_dataset_cache_file:
            fn = functools.partial(
                SliceDataset, use_dataset_cache=self.use_dataset_cache_file, dataset_cache_file=self.dataset_cache_file,
            )
            fn(root=self.data_path / f'{self.train_source}_train',
               coil_sens_path=self.coil_sens_path / f'{self.train_source}',
               transform=self.train_transform,
               sample_rate=self.sample_rate,
               volume_sample_rate=self.volume_sample_rate)
            fn(root=self.data_path / f'{self.val_source}_val',
               coil_sens_path=self.coil_sens_path / f'{self.val_source}',
               transform=self.val_transform,
               sample_rate=1.0,
               volume_sample_rate=None)
            test_roots = [self.data_path / f'{self.test_source}_{split}' for split in self._normalize_test_splits()]
            fn(root=test_roots,
               coil_sens_path=self.coil_sens_path / f'{self.test_source}',
               transform=self.test_transform,
               sample_rate=1.0,
               volume_sample_rate=None)

    def train_dataloader(self):
        if self.train_transform is None:
            raise ValueError('train_transform must be set.')
        return self._create_data_loader(self.train_transform, data_partition='train')

    def val_dataloader(self):
        if self.val_transform is None:
            raise ValueError('val_transform must be set.')
        return self._create_data_loader(self.val_transform, data_partition='val')

    def test_dataloader(self):
        if self.test_transform is None:
            raise ValueError('test_transform must be set.')
        return self._create_data_loader(self.test_transform, data_partition='test')
