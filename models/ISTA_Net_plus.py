"""ISTA-Net+ unrolled reconstruction network for 2D+T multi-coil MRI.

Reference: https://github.com/jianzhangcs/ISTA-Net-PyTorch

Port of ISTA-Net+ (J. Zhang, B. Ghanem, "ISTA-Net: Interpretable Optimization-Inspired Deep
Network for Image Compressive Sensing", CVPR 2018 - the ISTA-Net+ variant) to dynamic multi-coil
MRI: each iteration is one gradient/data-consistency step plus one CNN proximal step.
Referenced by pl_modules/ISTANetplus_module.py (class_path: pl_modules.ISTANetplusModule).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers.mri_torch import sens_expand, sens_reduce


class CONV_OP(nn.Module):
    """Single 3x3 Conv3d over channels-first volumes with optional ReLU (no bias)."""
    def __init__(self, in_ch: int = 2, n_f: int = 32, ifactivate: bool = False):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, n_f, kernel_size=3, stride=1, padding=1, bias=False)
        self.activate = nn.ReLU() if ifactivate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: channels-first [B, C, D, H, W] -> conv (+ optional activation), same shape."""
        y = self.conv(x)
        y = self.activate(y)
        return y


class ISTA_Net_plus(nn.Module):
    """ISTA-Net+ model: `niter` unrolled ISTA cells (ISTACell) sharing one learned layer set."""
    def __init__(self, niter: int = 10, channels: int = 16):
        super().__init__()
        self.niter = niter
        self.channels = channels
        self.celllist = nn.ModuleList()
        for i in range(self.niter):
            self.celllist.append(ISTACell(channels))

    def forward(self,
                img_zf: torch.Tensor,
                masked_kspace: torch.Tensor,
                mask: torch.Tensor,
                sens_maps: torch.Tensor):
        """
        img_zf: zero-filled image [nb, nmaps=1, nc=1, nt, nx, ny, 2];
        masked_kspace / mask / sens_maps: measured multi-coil data.
        Returns (x_rec: complex [nb, nt, nx, ny], symmetry loss: scalar).
        """
        # img_zf: [nb, nmaps=1, nc=1, nt, nx, ny, 2]
        atb = torch.view_as_complex(img_zf)[:, 0, 0]  # [nb, nt, nx, ny]
        
        x = atb
        x_sym = torch.zeros_like(x)
        data = [x, x_sym, masked_kspace, mask, sens_maps]

        X_SYM = []
        for i in range(self.niter):
            data = self.celllist[i](data)
            x_sym = data[1]
            X_SYM.append(x_sym)

        x_rec = data[0]
        loss_sym = 0
        for k in range(len(X_SYM)):
            loss_sym += torch.mean(torch.square(X_SYM[k]))

        return x_rec, loss_sym


class ISTACell(nn.Module):
    """One ISTA iteration: gradient/data-consistency step (`gdecent_step`) then CNN proximal step (`prox_step`)."""
    def __init__(self, channels: int = 16):
        super().__init__()
        self.thres_coef = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.mu = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # follow original TF naming: conv_1..conv_6
        self.conv_1 = CONV_OP(in_ch=2, n_f=channels, ifactivate=True)
        self.conv_2 = CONV_OP(in_ch=channels, n_f=channels, ifactivate=True)
        self.conv_3 = CONV_OP(in_ch=channels, n_f=channels, ifactivate=False)

        self.conv_4 = CONV_OP(in_ch=channels, n_f=channels, ifactivate=True)
        self.conv_5 = CONV_OP(in_ch=channels, n_f=channels, ifactivate=True)
        self.conv_6 = CONV_OP(in_ch=channels, n_f=2, ifactivate=False)

    def forward(self, data):
        """data = [x, x_sym, masked_kspace, mask, sens_maps] -> same list with x and x_sym updated in place."""
        x, x_sym, masked_kspace, mask, sens_maps = data

        r = self.gdecent_step(x, masked_kspace, mask, sens_maps)
        x, x_sym = self.prox_step(r)

        data[0] = x
        data[1] = x_sym
        return data

    def prox_step(self, r: torch.Tensor):
        """Proximal step on complex r [batch, Nt, Nx, Ny] -> (A, x_sym), both complex with the same shape."""
        batch, Nt, Nx, Ny = r.shape
        # build channels-first complex real/imag tensor for Conv3d: [B, 2, Nt, Nx, Ny]
        x_in = torch.stack([r.real, r.imag], dim=1)

        x_1 = self.conv_1(x_in)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)

        x_soft = torch.sign(x_3) * torch.relu(torch.abs(x_3) - torch.relu(self.thres_coef))

        x_4 = self.conv_4(x_soft)
        x_5 = self.conv_5(x_4)
        x_6 = self.conv_6(x_5)

        A = x_6 + x_in
        A = torch.complex(A[:, 0], A[:, 1])

        x_1_sym = self.conv_4(x_3)
        x_1_sym = self.conv_5(x_1_sym)
        x_sym = x_1_sym - x_1

        return A, x_sym

    def gdecent_step(self, x_pre: torch.Tensor, masked_kspace: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """Gradient step x - relu(mu) * E^H (mask * E x - y); complex [nb, nt, nx, ny] in and out."""
        # forward op: apply sensitivities then FFT
        nb, nt, nx, ny = x_pre.shape
        x_real = torch.view_as_real(x_pre).reshape(nb, 1, 1, nt, nx, ny, 2)
        k = sens_expand(x_real, sens_maps)  # [nb, coils, nt, nx, ny, 2]

        # data consistency residual in k-space
        resid = mask * k - masked_kspace

        # adjoint: iFFT then coil-combine with sensitivities
        img = sens_reduce(resid, sens_maps)  # [nb, 1, 1, nt, nx, ny, 2]
        adj = torch.view_as_complex(img).reshape(nb, nt, nx, ny)

        r = x_pre - F.relu(self.mu).to(dtype=torch.complex64) * adj
        return r