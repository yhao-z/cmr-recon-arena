"""JotlasNet unrolled reconstruction network for 2D+T multi-coil MRI.

Reference: https://github.com/yhao-z/JotlasNet

Implements JotlasNet from "JotlasNet: Joint tensor low-rank and attention-based sparse unrolling
network for dynamic MRI reconstruction" (arXiv:2502.11749, IEEE TMI 2025): per iteration a
gradient step followed by a tensor low-rank proximal and an attention-based sparse proximal.
Referenced by pl_modules/JotlasNet_module.py (class_path: pl_modules.JotlasNetModule).
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


class ST(nn.Module):
    """Channel-attention soft threshold: GAP + MLP predict per-channel thresholds, applied to |x| with weight w."""
    def __init__(self, channels: int):
        super().__init__()
        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.att = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x: complex [B, C, T, H, W] plus scalar weight w -> soft-thresholded x with the same shape."""
        x_abs = torch.abs(x)
        gap = self.GAP(x_abs)  # [batch, channels, 1, 1, 1]
        attcoefs = self.att(gap).view(gap.shape[0], gap.shape[1], 1, 1, 1)
        soft_thres = gap * attcoefs / w
        x_soft = torch.sign(x) * F.relu(x_abs - soft_thres)
        return x_soft
    

class JotlasNet(nn.Module):
    """JotlasNet: `niter` unrolled JotlasCell steps combining a tensor low-rank and an attention-based sparse prior."""
    def __init__(self, niter: int = 15, channels: int = 16):
        super().__init__()
        self.niter = niter
        self.channels = channels

        self.celllist = nn.ModuleList()
        for i in range(self.niter):
            self.celllist.append(JotlasCell(self.channels))

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
        x_done = x_rec
        data = [x_rec, masked_kspace, mask, sens_maps, x_done]

        for cell in self.celllist:
            data = cell(data)

        x_rec = data[0]

        return x_rec


class JotlasCell(nn.Module):
    """One unrolled step: gradient step, TLR proximal and attention sparse proximal mixed by w, plus an x - x_done momentum term."""
    def __init__(self, channels: int):
        super().__init__()

        self.thres_lr = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.mu = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.t = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

        self.sconv_1 = CNNLayer(in_ch=2, n_f=channels, out_ch=channels)
        self.sconv_2 = CNNLayer(in_ch=channels, n_f=channels, out_ch=2)
        self.ST = ST(channels)

        self.lconv_1 = CNNLayer(in_ch=2, n_f=channels, out_ch=2)
        self.lconv_2 = CNNLayer(in_ch=2, n_f=channels, out_ch=2)


    def forward(self, data):
        """data = [x_rec, masked_kspace, mask, sens_maps, x_done] -> same list with x_rec and x_done updated in place."""
        x_rec, masked_kspace, mask, sens_maps, x_done = data
        w = torch.softmax(self.w, dim=0)
        xg = self.gdecent_step(x_rec, masked_kspace, mask, sens_maps)
        Z_tlr = self.prox_tlr(xg, w)
        Z_as = self.prox_as(xg, w)

        x = Z_tlr * w[0].to(dtype=torch.complex64) + Z_as * w[1].to(dtype=torch.complex64)
        x_rec = x + torch.sigmoid(self.t).to(dtype=torch.complex64) * (x - x_done)
        x_done = x
        data[0] = x_rec
        data[-1] = x_done

        return data


    def prox_tlr(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Tensor low-rank proximal: conv -> batched SVD soft threshold (one [Nx, Ny] matrix per frame, scaled by w[0]) -> conv; complex [batch, Nt, Nx, Ny] -> same shape."""
        x_in = torch.stack([x.real, x.imag], dim=1)  # [batch, 2, Nt, Nx, Ny]
        x_conv = self.lconv_1(x_in)  # [batch, 2, Nt, Nx, Ny]
        x_c = torch.complex(x_conv[:, 0], x_conv[:, 1])  # [batch, Nt, Nx, Ny]
        
        U, S, Vh = torch.linalg.svd(x_c, full_matrices=False)
        thres = torch.sigmoid(self.thres_lr) * S[..., 0:1] / w[0] # [batch, Nt, 1]
        S = F.relu(S - thres)
        S = torch.diag_embed(S).to(dtype=torch.complex64)
        x_soft = U @ S @ Vh  # [batch, Nt, Nx, Ny]
        
        x_soft = torch.stack([x_soft.real, x_soft.imag], dim=1)  # [batch, 2, Nt, Nx, Ny]
        x_out = self.lconv_2(x_soft)  # [batch, 2, Nt, Nx, Ny]
        out = torch.complex(x_out[:, 0], x_out[:, 1])  # [batch, Nt, Nx, Ny]

        return out + x

    def prox_as(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Attention-based sparse proximal: conv -> ST soft threshold (weight w[1]) -> conv; complex [batch, Nt, Nx, Ny] -> same shape."""
        x_in = torch.stack([x.real, x.imag], dim=1)  # [batch, 2, Nt, Nx, Ny]
        x_conv1 = self.sconv_1(x_in)  # [batch, channels, Nt, Nx, Ny]
        
        x_st = self.ST(x_conv1, w[1])  # [batch, channels, Nt, Nx, Ny]
        
        x_conv2 = self.sconv_2(x_st)  # [batch, 2, Nt, Nx, Ny]
        out = torch.complex(x_conv2[:, 0], x_conv2[:, 1])  # [batch, Nt, Nx, Ny]
        
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