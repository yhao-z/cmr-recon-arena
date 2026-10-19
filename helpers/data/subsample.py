"""
Dynamic (k-t) undersampling mask generators for cine MRI.
Defines the MaskFunc ABC and its random/VISTA/GRO/CAVA/equispaced variants;
the classes are instantiated directly via configs/data/*.yaml
class_path strings.
"""

import contextlib
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

import helpers.sampling_patterns as patterns


@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: float | Sequence[float] | None):
    """A context manager for temporarily adjusting the random seed."""
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc(ABC):
    """Base class for dynamic k-t masks: picks an acceleration, delegates to generate_mask()."""
    def __init__(self, accelerations: Sequence[int]):
        """
        Args:
            accelerations: Amount of under-sampling. If multiple values are
                provided, then one of these is chosen randomly each time.
        """
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(
            self,
            num_frames: int,
            num_cols: int,
            seed: float | Sequence[float] | None = None,
            **kwargs,
    ) -> np.ndarray:
        """
        Sample and return a k-space mask.

        Args:
            num_frames: Number of frames in the temporal dimension.
            num_cols: Number of k-space columns (phase dimension).
            padding_left: Amount of zero-padding on the left. This should not be
                included in `num_cols` but is considered when determining the
                center of k-space. Note that certain mask functions might only
                support certain combinations of `padding_left` and
                `padding_right`.
            padding_right: Amount of zero-padding on the right. This should not
                be included in `num_cols` but is considered when determining the
                center of k-space. Note that certain mask functions might only
                support certain combinations of `padding_left` and
                `padding_right`.
            offset: Offset from 0 to begin mask (for equispaced masks). If -1,
                then one is selected randomly. Defaults to -1.
            augment: If True, augment the mask (for GRO masks). Defaults to
                False.
            seed: Seed for random number generator for reproducibility.

        Returns:
            the k-space mask
        """
        with temp_seed(self.rng, seed):
            acceleration = self.rng.choice(self.accelerations)
            mask = self.generate_mask(num_frames, num_cols, acceleration, **kwargs)

        assert mask.ndim == 2, f'Mask should have 2 dimensions, but has {mask.ndim}'
        assert mask.shape[0] == num_frames, 'Temporal size of mask and target shape do not match'
        assert mask.shape[1] == num_cols, 'Number of columns of mask and target shape do not match'

        return mask.astype(np.uint8)

    @abstractmethod
    def generate_mask(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _is_off_center(num_cols: int, padding_left: int, padding_right: int) -> bool:
        center_idx_padded = (num_cols + padding_left + padding_right) // 2
        center_idx_unpadded = num_cols // 2
        if center_idx_padded - padding_left != center_idx_unpadded:
            return True
        return False


class RandomMaskFunc(MaskFunc):
    """Uniformly random dynamic mask with a randomly chosen acceleration."""
    def generate_mask(self, num_frames: int, num_cols: int, acceleration: int, **kwargs) -> np.ndarray:
        samp = patterns.dynamic.uniform(num_frames, num_cols, acceleration=acceleration, rng=self.rng)
        return samp


class RandomACSMaskFunc(MaskFunc):
    """Uniformly random dynamic mask with a 16-line auto-calibration (ACS) center."""
    def generate_mask(self,
            num_frames: int,
            num_cols: int,
            acceleration: int,
            padding_left: int = 0,
            padding_right: int = 0,
            **kwargs,
    ) -> np.ndarray:
        center = (num_cols + padding_left + padding_right) // 2 - padding_left
        samp = patterns.dynamic.uniform(
            num_frames, num_cols, acceleration, acs_width=16, adjust_acceleration=True, center=center, rng=self.rng,
        )
        return samp


class VISTAMaskFunc(MaskFunc):
    """VISTA (view-sharing-free) variable-density mask; rejects off-center padding."""
    def generate_mask(
            self,
            num_frames: int,
            num_cols: int,
            acceleration: int,
            padding_left: int = 0,
            padding_right: int = 0,
            **kwargs,
    ) -> np.ndarray:
        if self._is_off_center(num_cols, padding_left, padding_right):
            raise NotImplementedError('VISTA does not support off-center masking caused by uneven padding')

        num_samples_per_frame = round(num_cols / acceleration)
        param = patterns.dynamic.VISTAParam(PE=num_cols, FR=num_frames, n=num_samples_per_frame, fc=16/num_cols)
        samp = patterns.dynamic.VISTA(param, rng=self.rng)

        return samp.T


class GROMaskFunc(MaskFunc):
    """Golden-ratio ordering mask; a random offset is applied when augmenting."""
    def generate_mask(
            self,
            num_frames: int,
            num_cols: int,
            acceleration: int,
            padding_left: int = 0,
            padding_right: int = 0,
            augment: bool = False,
            **kwargs,
    ) -> np.ndarray:
        if self._is_off_center(num_cols, padding_left, padding_right):
            raise NotImplementedError('GRO does not support off-center masking caused by uneven padding')

        num_samples_per_frame = round(num_cols / acceleration)
        param = patterns.dynamic.GROParam(PE=num_cols, FR=num_frames, n=num_samples_per_frame, E=1)
        offset = self.rng.random() if augment else 0
        samp = patterns.dynamic.GRO(param, offset=offset)
        return samp.T[0]  # type: ignore


class CAVAMaskFunc(MaskFunc):
    """CAVA variable-density mask; rejects off-center padding."""
    def generate_mask(
            self,
            num_frames: int,
            num_cols: int,
            acceleration: int,
            padding_left: int = 0,
            padding_right: int = 0,
            **kwargs,
    ) -> np.ndarray:
        if self._is_off_center(num_cols, padding_left, padding_right):
            raise NotImplementedError('CAVA does not support off-center masking caused by uneven padding')

        num_samples_per_frame = round(num_cols / acceleration)
        param = patterns.dynamic.CAVAParam(PE=num_cols, FR=num_frames, n=num_samples_per_frame, E=1)
        samp = patterns.dynamic.CAVA(param)
        return samp.T[0]


class EquispacedMaskFunc(MaskFunc):
    """Equidistant dynamic mask without an ACS center."""
    def generate_mask(
            self,
            num_frames: int,
            num_cols: int,
            acceleration: int,
            offset: int = -1,
            **kwargs,
    ) -> np.ndarray:
        samp = patterns.dynamic.equispaced(
            num_frames, num_cols, acceleration, acs_width=0, offset=offset, roll_offset=True, rng=self.rng,
        )
        return samp


class EquispacedACSMaskFunc(MaskFunc):
    """Equidistant dynamic mask with a 16-line ACS center."""
    def generate_mask(
            self,
            num_frames: int,
            num_cols: int,
            acceleration: int,
            padding_left: int = 0,
            padding_right: int = 0,
            offset: int = -1,
            **kwargs,
    ) -> np.ndarray:
        center = (num_cols + padding_left + padding_right) // 2 - padding_left
        samp = patterns.dynamic.equispaced(
            num_frames, num_cols, acceleration, acs_width=16, offset=offset, roll_offset=True, adjust_acceleration=True,
            center=center, rng=self.rng,
        )
        return samp
    
    
class StaticEquispacedACSMaskFunc(MaskFunc):
    '''
    CMRxRecon2023 Cine track uses this type of mask
    '''
    
    def generate_mask(
            self,
            num_frames: int,
            num_cols: int,
            acceleration: int,
            padding_left: int = 0,
            padding_right: int = 0,
            offset: int = -1,
            **kwargs,
    ) -> np.ndarray:
        center = (num_cols + padding_left + padding_right) // 2 - padding_left
        samp = patterns.static.equispaced(
            num_cols, acceleration, acs_width=24, offset=offset, adjust_acceleration=False,
            center=center, rng=self.rng,
        )
        samp = np.tile(samp[None,...], (num_frames, 1))
        return samp

