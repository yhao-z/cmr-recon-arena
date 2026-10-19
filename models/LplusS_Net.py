"""L+S-Net (low-rank + sparse) reconstruction network for 2D+T multi-coil MRI.

Reference: https://github.com/wenqihuang/LS-Net-Dynamic-MRI

Implements L+S-Net from "Deep low-rank plus sparse network for dynamic MR imaging"
(W. Huang et al., arXiv:2010.13677, Medical Image Analysis 2021): unrolled alternating
low-rank (learned SVT) and sparse CNN updates with k-space data consistency.
Referenced by pl_modules/LplusSNet_module.py (class_path: pl_modules.LplusSNetModule).
"""

import torch
import torch.nn as nn
from helpers.mri_torch import sens_expand, sens_reduce
  

class CNNLayer(nn.Module):
    """Sparse-update CNN: three 3x3 Conv3d layers (4 -> n_f -> n_f -> 2) with LeakyReLU."""
    def __init__(self, n_f: int = 32):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(4, n_f, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(n_f, n_f, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(n_f, 2, kernel_size=3, stride=1, padding=1, bias=False),
        )
        
    def norm(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample normalization over flattened dims -> ((x - mean) / std, mean, std). Currently unused (see the commented call sites in forward)."""
        mean = x.flatten(1).mean(dim=1)
        std = x.flatten(1).std(dim=1)
        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Inverse of norm(): x * std + mean. Currently unused (normalization is toggled off in forward)."""
        return x * std + mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: complex [B, 2, T, H, W] -> stack real/imag into 4 channels -> Conv3d stack -> complex [B, 2, T, H, W]."""
        x2c = torch.cat((x.real, x.imag), dim=1) # b,4,t,h,w
        # x2c, mean, std = self.norm(x2c)
        res = self.seq(x2c)
        # res = self.unnorm(res, mean, std)

        return torch.complex(res[:, 0], res[:, 1])


class LplusS_Net(nn.Module):
    """L+S-Net: `niter` unrolled LSCell steps alternating low-rank and sparse updates with data consistency."""
    def __init__(self, niter: int, learnedSVT: bool = True):
        super().__init__()
        self.niter = niter
        self.learnedSVT = learnedSVT

        self.celllist = nn.ModuleList()
        for i in range(self.niter):
            self.celllist.append(
                LSCell_learned_step(
                    is_last=(i == self.niter - 1),
                    learnedSVT=self.learnedSVT,
                )
            )

    def forward(self, 
                img_zf: torch.Tensor,
                masked_kspace: torch.Tensor,
                mask: torch.Tensor,
                sens_maps: torch.Tensor):
        """
        img_zf: zero-filled image [nb, nmaps=1, nc=1, nt, nx, ny, 2];
        masked_kspace / mask / sens_maps: measured multi-coil data.
        Returns the reconstruction (L + S) as complex [nb, nt, nx, ny].
        """
        # img_zf: [nb, nmaps=1, nc=1, nt, nx, ny, 2] 
        atb = torch.view_as_complex(img_zf)[:, 0, 0]  # [nb, nt, nx, ny]
        nb, nt, nx, ny = atb.shape

        # Initial L, S, M
        Lpre = atb.reshape(nb, nt, nx * ny)
        Spre = torch.zeros_like(Lpre)
        Mpre = Lpre
        data = [Lpre, Spre, Mpre, masked_kspace, mask, sens_maps]

        # unrolling iterations
        for cell in self.celllist:
            data = cell(data)

        # final reconstruction, reshape and unnormalization
        L, S = data[0:2]
        recon = (L+S).reshape(nb, nt, nx, ny)

        return recon


class LSCell_learned_step(nn.Module):
    """One L+S iteration: low-rank SVT update, sparse CNN update, data-consistency step with step size gamma."""
    def __init__(self, is_last: bool, learnedSVT: bool = True):
        super().__init__()

        self.learnedSVT = learnedSVT
        if self.learnedSVT:
            self.thres_coef = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.sconv = CNNLayer(n_f=32)
        self.is_last = is_last
        if not is_last:
            self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, data):
        """data = [Lpre, Spre, Mpre, masked_kspace, mask, sens_maps] -> same list with L, S and M updated in place."""

        Lpre, Spre, Mpre, masked_kspace, mask, sens_maps = data
        self.nb, self.nt = Lpre.shape[0], Lpre.shape[1]
        self.nx, self.ny = masked_kspace.shape[-3], masked_kspace.shape[-2]

        L = self.lowrank(Mpre)
        S = self.sparse(Mpre, L)
        dc = self.dataconsis(L + S, masked_kspace, mask, sens_maps)
        if not self.is_last:
            gamma = torch.relu(self.gamma).to(dtype=torch.complex64)
        else:
            gamma = torch.tensor(1.0, dtype=torch.complex64, device=Mpre.device)

        M = L + S - gamma * dc

        data[0] = L
        data[1] = S
        data[2] = M
        return data

    def lowrank(self, L: torch.Tensor) -> torch.Tensor:
        """Soft-thresholded SVD (learned threshold) of L: complex [nb, nt, nx*ny] -> same shape."""
        U, S, Vh = torch.linalg.svd(L, full_matrices=False)
        if self.learnedSVT:
            thres = torch.sigmoid(self.thres_coef) * S[:,0]
            S = torch.relu(S - thres)
        else:
            raise NotImplementedError("Only learned SVT is implemented.")

        S = torch.diag_embed(S).to(dtype=torch.complex64)
        U = U.to(dtype=torch.complex64)
        Vh = Vh.to(dtype=torch.complex64)

        L_recon = U @ S @ Vh
        return L_recon

    def sparse(self, Mpre: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Sparse CNN update: stacks L and M on the channel dim, applies sconv -> [nb, nt, nx*ny] complex."""
        M_L = torch.stack(
            (
                L.reshape(self.nb, self.nt, self.nx, self.ny),
                Mpre.reshape(self.nb, self.nt, self.nx, self.ny),
            ),
            dim=1,
        )
        S = self.sconv(M_L)
        S = S.reshape(self.nb, self.nt, self.nx * self.ny)
        return S

    def dataconsis(self, LS: torch.Tensor, masked_kspace: torch.Tensor, mask:torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """Data-consistency residual E^H (mask * E (L+S) - y) reshaped back to [nb, nt, nx*ny] complex."""
        # LS: [nb, nt, nx*ny]
        x = torch.view_as_real(LS).reshape(self.nb, 1, 1, self.nt, self.nx, self.ny, 2)  # [nb, nt, nx, ny, 2]
        dc = sens_reduce( mask * sens_expand(x, sens_maps) - masked_kspace, sens_maps )  # [nb, nmaps=1, nc=1, nt, nx, ny, 2]
        return torch.view_as_complex(dc).reshape(*LS.shape)
