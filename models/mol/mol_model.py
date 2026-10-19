"""
Top-level MOL model for 2D+T multi-coil dynamic MRI.

trn_mol.py call path:
    A_init = sense(cgIter_init)           -> MOLForwardModel(cg_iter=num_cg_init)
    A = sense(cgIter_itr)                 -> MOLForwardModel(cg_iter=num_cg_iters)
    optBlock = fwdbwdBlock(A, lam, ...)   -> self.optBlock
    model = DEQ(optBlock, K, A_init, lam) -> self.deq

Part of models/mol; loaded by pl_modules/mol_module.py (from models.mol import MOL).
"""

import torch
import torch.nn as nn

from .forward_model import MOLForwardModel
from .solver_blocks import fwdbwdBlock
from .solvers import DEQ


class MOL(nn.Module):
    """
    Monotone Operator Learning — DEQ variant for 2D+T multi-coil MRI.

    Parameters (mapped from trn_mol.py):
        num_iters:      K (nFETarget=500)
        num_cg_init:    cgIter_init (50)
        num_cg_iters:   cgIter_itr (5)
        features:       number_of_feature_filters (64)
        layers:         number_of_layers (5)
        lam_init:       lam_init (100)
        lam:            lam_itr (100, learnable)
        alpha:          alpha (0.01)
        spectral_norm:  False (sn=False in trn_mol.py)
        tol:            err_tol (1e-4)
    """

    def __init__(
        self,
        num_iters: int = 500,
        num_cg_init: int = 50,
        num_cg_iters: int = 5,
        features: int = 64,
        layers: int = 5,
        lam_init: float = 100.0,
        lam: float = 100.0,
        alpha: float = 0.01,
        spectral_norm: bool = False,
        tol: float = 1e-4,
        verbose: bool = False,
    ):
        super().__init__()

        A_init = MOLForwardModel(cg_iter=num_cg_init)
        A = MOLForwardModel(cg_iter=num_cg_iters)

        self.optBlock = fwdbwdBlock(
            A, lam,
            input_channels=1, features=features, output_channels=1,
            number_of_layers=layers, spectral_norm=spectral_norm,
        )
        self.optBlock.alpha = torch.tensor(alpha, dtype=torch.float32)

        self.deq = DEQ(
            self.optBlock, K=num_iters,
            A_init=A_init, lam_init=lam_init,
            tol=tol, verbose=verbose,
        )

    def forward(self, masked_kspace, mask, sens_maps):
        """
        Parameters
        ----------
        masked_kspace : [B, coils, T, H, W, 2]
        mask : [B, 1, T, 1, W, 1]
        sens_maps : [B, maps, coils, T, H, W, 2]

        Returns
        -------
        reconstruction : complex64 [B, T, H, W]
        errforward : float
        nFE : int
        """
        x_deq, sense_out, errforward, nFE = self.deq(masked_kspace, sens_maps, mask)
        return x_deq, errforward, nFE

    @property
    def dw(self):
        """The denoiser (dwblock) inside the forward-backward solver, exposed as a property."""
        return self.optBlock.dw
