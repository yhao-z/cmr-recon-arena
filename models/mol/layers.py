"""
Convolutional layer for MOL — 3D version.

Original: layers.py in Monotone-Operator-Learning
Change: Conv2d -> Conv3d for 2D+T (spatial + temporal).
All other logic preserved exactly.

NOTE: trn_mol.py uses spectral_norm=False (sn=False) by default.
      The spectral_norm option is kept for future experiments.

Part of models/mol; loaded by pl_modules/mol_module.py (from models.mol import MOL).
"""

import torch.nn as nn
from .spectral_norm_3d import spectral_norm


class convlayer(nn.Module):
    """
    Conv3d + ReLU (no ReLU if last layer).
    Identical logic to original convlayer, Conv2d -> Conv3d.
    """

    def __init__(self, input_channels, output_channels, last, sn=False):
        super().__init__()
        if sn:
            # NOTE: sn=False in trn_mol.py default. This branch is unused.
            self.conv = spectral_norm(
                nn.Conv3d(input_channels, output_channels,
                          kernel_size=3, stride=1, padding=1, bias=True)
            )
        else:
            self.conv = nn.Conv3d(input_channels, output_channels,
                                  kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.last = last

    def forward(self, x):
        """x: [B, C_in, T, H, W] -> [B, C_out, T, H, W], followed by ReLU unless this is the last layer."""
        x = self.conv(x)
        if not self.last:
            x = self.relu(x)
        return x
