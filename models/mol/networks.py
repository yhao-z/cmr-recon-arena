"""
Network blocks for MOL — 3D version.

Original: networks.py in Monotone-Operator-Learning
Only dwblock is migrated (trn_mol.py only uses dwblock via fwdbwdBlock).

Part of models/mol; loaded by pl_modules/mol_module.py (from models.mol import MOL).
"""

import torch
import torch.nn as nn
from .layers import convlayer


def complex_to_real(x):
    """
    complex64 [B, T, H, W] -> float32 [B, 2, T, H, W]
    """
    return torch.view_as_real(x).permute(0, 4, 1, 2, 3).float()


def real_to_complex(x):
    """
    float32 [B, 2, T, H, W] -> complex64 [B, T, H, W]
    """
    return torch.view_as_complex(x.permute(0, 2, 3, 4, 1).contiguous())


class dwblock(nn.Module):
    """
    Deep-learning denoiser block — the monotone operator in MOL.
    complex [B, T, H, W] -> real [B, 2, T, H, W] -> Conv3D x N -> complex [B, T, H, W]
    """

    def __init__(self, input_channels, features, output_channels, number_of_layers, spectral_norm=False):
        super().__init__()
        layers = [convlayer(input_channels * 2, features, False, spectral_norm)]
        for _ in range(1, number_of_layers - 1):
            layers.append(convlayer(features, features, False, spectral_norm))
        layers.append(convlayer(features, output_channels * 2, True, spectral_norm))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """complex input (for MOL [B, T, H, W]) -> Conv3d stack in the real/imag layout -> complex output with `output_channels` channels."""
        return real_to_complex(self.net(complex_to_real(x)))
