"""T2LR-Net unrolled reconstruction network for 2D+T multi-coil MRI.

Reference: https://github.com/yhao-z/T2LR-Net

Implements T2LR-Net from "T2LR-Net: An unrolling network learning transformed tensor low-rank
prior for dynamic MR image reconstruction" (arXiv:2209.03832): per iteration one gradient/data-
consistency step plus a CNN proximal step with a learned singular-value threshold.
Referenced by pl_modules/T2LRNet_module.py (class_path: pl_modules.T2LRNetModule).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers.mri_torch import sens_expand, sens_reduce


class CNNLayer(nn.Module):
    """Stack of three 3x3 Conv3d layers (in_ch -> n_f -> n_f -> out_ch), optional ReLU on the output."""
    def __init__(self, in_ch: int, n_f: int = 32, out_ch: int = 2, act_last: bool = False):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, n_f, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_f, n_f, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_f, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        ]
        if act_last:
            layers.append(nn.ReLU(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """[B, in_ch, T, H, W] -> [B, out_ch, T, H, W] (same temporal/spatial size)."""
        return self.seq(input)


class T2LRNet(nn.Module):
    """T2LR-Net: `niter` unrolled T2LRCell iterations (transformed tensor low-rank prior)."""
    def __init__(self, niter: int = 15, channels: int = 16):
        super().__init__()
        self.niter = niter
        self.channels = channels

        self.celllist = nn.ModuleList()
        for i in range(self.niter):
            self.celllist.append(T2LRCell(self.channels))

    def forward(self,
                img_zf: torch.Tensor,
                masked_kspace: torch.Tensor,
                mask: torch.Tensor,
                sens_maps: torch.Tensor):
        """
        img_zf: zero-filled image [nb, nmaps=1, nc=1, nt, nx, ny, 2];
        masked_kspace / mask / sens_maps: measured multi-coil data.
        Returns the reconstruction as complex [nb, nt, nx, ny].
        """
        # img_zf: [nb, nmaps=1, nc=1, nt, nx, ny, 2]
        atb = torch.view_as_complex(img_zf)[:, 0, 0]  # [nb, nt, nx, ny]

        x_rec = atb
        data = [x_rec, masked_kspace, mask, sens_maps]

        for cell in self.celllist:
            data = cell(data)

        x_rec = data[0]

        return x_rec


class T2LRCell(nn.Module):
    """One unrolled step: gradient/data-consistency step + transformed tensor low-rank proximal (`prox_tlr`)."""
    def __init__(self, channels: int):
        super().__init__()

        self.thres_lr = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.mu = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.lconv_1 = CNNLayer(in_ch=2, n_f=channels, out_ch=2)
        self.lconv_2 = CNNLayer(in_ch=2, n_f=channels, out_ch=2)


    def forward(self, data):
        """data = [x_rec, masked_kspace, mask, sens_maps] -> same list with x_rec updated in place."""
        x_rec, masked_kspace, mask, sens_maps = data

        xg = self.gdecent_step(x_rec, masked_kspace, mask, sens_maps)
        x_rec = self.prox_tlr(xg)

        data[0] = x_rec

        return data


    def prox_tlr(self, x: torch.Tensor) -> torch.Tensor:
        """Learned transform -> batched SVD soft threshold (one [Nx, Ny] matrix per frame) -> learned transform; complex [batch, Nt, Nx, Ny] -> same shape."""
        x_in = torch.stack([x.real, x.imag], dim=1)  # [batch, 2, Nt, Nx, Ny]
        x_conv = self.lconv_1(x_in)  # [batch, 2, Nt, Nx, Ny]
        x_c = torch.complex(x_conv[:, 0], x_conv[:, 1])  # [batch, Nt, Nx, Ny]
        
        U, S, Vh = torch.linalg.svd(x_c, full_matrices=False)
        thres = torch.sigmoid(self.thres_lr) * S[..., 0:1] # [batch, Nt, 1]
        S = F.relu(S - thres)
        S = torch.diag_embed(S).to(dtype=torch.complex64)
        x_soft = U @ S @ Vh  # [batch, Nt, Nx, Ny]
        
        x_soft = torch.stack([x_soft.real, x_soft.imag], dim=1)  # [batch, 2, Nt, Nx, Ny]
        x_out = self.lconv_2(x_soft)  # [batch, 2, Nt, Nx, Ny]
        out = torch.complex(x_out[:, 0], x_out[:, 1])  # [batch, Nt, Nx, Ny]

        return out + x


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