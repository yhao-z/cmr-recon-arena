"""Convolution wrappers with per-dimension padding modes for the cinevn U-Nets.
Used by models/cinevn/unet.py (the per-cascade regularizers of VarNet), which in turn is
used by pl_modules/varnet_module.py (class_path: pl_modules.VarNetModule).
"""

import torch
from torch import nn


class ConvNdWithFlexiblePadding(nn.Module):
    """
    This class implements a convolution layer where different padding modes can be applied along different dimensions
    """
    def __init__(self, N: int, padding: tuple[int, ...], padding_modes: tuple[str, ...], **kwargs):
        super().__init__()

        if len(padding) != N:
            raise ValueError(f'`padding` must have length {N}, but was {len(padding)}')
        if len(padding_modes) != N:
            raise ValueError(f'`padding_modes` must have length {N}, but was {len(padding_modes)}')

        match N:
            case 1:
                self.conv = nn.Conv1d(**kwargs)
            case 2:
                self.conv = nn.Conv2d(**kwargs)
            case 3:
                self.conv = nn.Conv3d(**kwargs)
            case _:
                raise NotImplementedError(f'ConvNdWithFlexiblePadding only implemented for N={{1, 2, 3}}, but was {N}')

        self.N = N
        self.padding_ = padding
        self.padding_modes = padding_modes

    def pad(self, image: torch.Tensor) -> torch.Tensor:
        """Pad every dimension from right to left with its configured padding mode."""
        # iterate from right to left
        for i in range(self.N):
            p = self.padding_[-i - 1]
            pad = (0, 0) * i + (p, p) + (0, 0) * (self.N - i - 1)
            mode = self.padding_modes[-i - 1]
            image_padded = torch.nn.functional.pad(image, pad, mode=mode)

            # replace
            image = image_padded

        return image

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """[B, C, ...] image -> padded image passed through the convolution (same rank)."""
        image = self.pad(image)
        image = self.conv.forward(image)
        return image


class Conv3dSpatiotemporal(ConvNdWithFlexiblePadding):
    """Conv3d with per-dimension padding: `pad_mode_temp` in time, replicate in readout, circular in phase."""
    def __init__(self, padding: tuple[int, int, int], pad_mode_temp: str, **kwargs):
        # `pad_mode_temp` in temporal dimension, 'replicate' in readout dimension, 'circular' in phase dimension
        super().__init__(N=3, padding=padding, padding_modes=(pad_mode_temp, 'replicate', 'circular'), **kwargs)


class Conv2dSpatial(ConvNdWithFlexiblePadding):
    """Conv2d with replicate padding in readout and circular padding in phase."""
    def __init__(self, padding: tuple[int, int], **kwargs):
        # 'replicate' in readout dimension, 'circular' in phase dimension
        super().__init__(N=2, padding=padding, padding_modes=('replicate', 'circular'), **kwargs)


class Conv1dTemporal(ConvNdWithFlexiblePadding):
    """Conv1d along the temporal dimension only, padded with `pad_mode_temp`."""
    def __init__(self, padding: tuple[int, ], pad_mode_temp: str, **kwargs):
        # `pad_mode_temp` in temporal dimension
        super().__init__(N=1, padding=padding, padding_modes=(pad_mode_temp, ), **kwargs)


class Conv3dSpatial(Conv2dSpatial):
    """Applies the 2D spatial convolution frame-wise: [B, C, T, X, Y] -> [B, C', T, X', Y] (T folded into the batch)."""
    def forward(self, image: torch.Tensor):
        """[B, C, T, X, Y] -> [B, C', T, X', Y]: spatial conv applied independently to every frame."""

        # move temporal dimension to batch dimension
        b, c, t, x, y = image.shape
        image = image.permute(0, 2, 1, 3, 4).reshape(b * t, c, x, y)

        # conv
        image = super().forward(image)

        # reshape back
        bt, c, x, y = image.shape
        assert b * t == bt
        image = image.view(b, t, c, x, y).permute(0, 2, 1, 3, 4).contiguous()

        return image


class Conv3dTemporal(Conv1dTemporal):
    """Applies the 1D temporal convolution per pixel: [B, C, T, X, Y] -> [B, C', T', X, Y] (spatial dims folded into the batch)."""
    def forward(self, image: torch.Tensor):
        """[B, C, T, X, Y] -> [B, C', T', X, Y]: temporal conv applied independently to every pixel."""
        # move spatial dimensions to batch dimension
        b, c, t, x, y = image.shape
        image = image.permute(0, 3, 4, 1, 2).reshape(b * x * y, c, t)

        # conv
        image = super().forward(image)

        # reshape back
        bxy, c, t = image.shape
        assert b * x * y == bxy
        image = image.view(b, x, y, c, t).permute(0, 3, 4, 1, 2).contiguous()

        return image
