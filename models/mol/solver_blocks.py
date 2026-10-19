"""
Solver block for MOL — only fwdbwdBlock (MOL-LR).

Original: solver_blocks.py in Monotone-Operator-Learning
Only fwdbwdBlock is migrated — this is the only block used in trn_mol.py
for the DEQ path. modlBlock/gradBlock/molgradBlock/admmBlock are NOT used.

fwdbwdBlock logic (IDENTICAL to original):
    z = dw(x)
    rhs = (1 - alpha) * x + alpha * z + alpha * lam * Atb
    x = A.inv(x, rhs, alpha * lam, csm, mask)

Part of models/mol; loaded by pl_modules/mol_module.py (from models.mol import MOL).
"""

import torch
import torch.nn as nn
from .networks import dwblock


class fwdbwdBlock(nn.Module):
    """
    Forward-backward splitting block (MOL-LR).

    Paper eq: x_{k+1} = prox_{alpha*g}(x_k - alpha * A^H(Ax_k - b))  +  learnable denoiser
    where g is the data fidelity term and the denoiser is the monotone CNN.

    Parameters (same as original):
        A: forward model (sense or MOLForwardModel)
        lam: initial regularization weight (learnable)
        input_channels: number of complex channels (1 for single image)
        features: CNN feature channels
        output_channels: number of complex channels out (1)
        number_of_layers: CNN depth
        spectral_norm: use spectral norm for monotonicity
    """

    def __init__(self, A, lam, input_channels, features, output_channels,
                 number_of_layers, spectral_norm=False):
        super().__init__()
        self.dw = dwblock(input_channels, features, output_channels,
                          number_of_layers, spectral_norm)
        self.lam = nn.Parameter(torch.tensor(lam, dtype=torch.float32))
        self.A = A
        self.alpha = torch.tensor(0.1, dtype=torch.float32)

    def forward(self, x, Atb, csm, mask):
        """
        One forward-backward iteration.

        x:   complex64 [B, C, T, H, W]  current estimate
        Atb: complex64 [B, C, T, H, W]  A^H * b (precomputed)
        csm: [B, maps, coils, T, H, W, 2]
        mask: [B, 1, T, 1, W, 1]

        Returns: complex64 [B, C, T, H, W]
        """
        z = self.dw(x)
        rhs = (1 - self.alpha) * x + self.alpha * z + self.alpha * self.lam * Atb
        x = self.A.inv(x, rhs, self.alpha * self.lam, csm, mask)
        return x
