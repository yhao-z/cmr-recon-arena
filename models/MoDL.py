"""
MoDL: Model-Based Deep Learning for MRI Reconstruction.

Reference: https://github.com/bo-10000/MoDL_PyTorch
Paper: MoDL: Model-Based Deep Learning Architecture for Inverse Problems
       H Aggarwal, M Mani, M Jacob, IEEE TMI, 2019.

Adapted for 2D+T multi-coil cardiac cine MRI:
  - Conv2d -> Conv3d (spatial + temporal)
  - Single-coil CG -> multi-coil CGDC (helpers/mri_torch/cgdc_img.py, the image-domain variant)

Referenced by pl_modules/MoDL_module.py (class_path: pl_modules.MoDLModule, configs/model/modl.yaml).
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from helpers.mri_torch.cgdc_img import CGDC


def complex_to_real(x):
    """complex64 [B, T, H, W] -> float32 [B, 2, T, H, W]"""
    return torch.view_as_real(x).permute(0, 4, 1, 2, 3).float()


def real_to_complex(x):
    """float32 [B, 2, T, H, W] -> complex64 [B, T, H, W]"""
    return torch.view_as_complex(x.permute(0, 2, 3, 4, 1).contiguous())


def conv_block(in_ch, out_ch):
    """Conv3d + ReLU; BatchNorm stays commented out here (reference block: Conv2d + BatchNorm2d + ReLU)."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1),
        # nn.BatchNorm3d(out_ch),
        nn.ReLU(),
    )


class cnn_denoiser(nn.Module):
    """
    CNN denoiser with residual connection.
    complex [B, T, H, W] -> complex [B, T, H, W]
    """

    def __init__(self, n_layers=5):
        super().__init__()
        layers = [conv_block(2, 64)]
        for _ in range(n_layers - 2):
            layers.append(conv_block(64, 64))
        layers.append(nn.Sequential(
            nn.Conv3d(64, 2, 3, padding=1),
            # nn.BatchNorm3d(2),
        ))
        self.nw = nn.Sequential(*layers)

    def forward(self, x):
        """complex [B, T, H, W] -> complex [B, T, H, W] (residual: input + CNN output, real/imag layout)."""
        x_r = complex_to_real(x)
        dw_r = x_r + self.nw(x_r)
        return real_to_complex(dw_r)


class MoDL(nn.Module):
    """
    MoDL: unrolled alternating minimization with optional gradient checkpointing.

    Parameters
    ----------
    n_layers : int  (CNN depth, default 5)
    k_iters : int   (unrolled iterations, default 10)
    cg_iter : int   (CG iterations, default 10)
    autograd : bool (PyTorch autograd for CG backward)
    use_checkpoint : bool (gradient checkpointing to save memory)
    """

    def __init__(self, n_layers=5, k_iters=10, cg_iter=10, autograd=True, use_checkpoint=False):
        super().__init__()
        self.k_iters = k_iters
        self.use_checkpoint = use_checkpoint
        self.dw = cnn_denoiser(n_layers)
        self.dc = CGDC(autograd=autograd, mu=None, n_iter=cg_iter, fft_ndim=2)

    def _iteration(self, x_k_ri, x0_ri, csm, mask):
        """One unrolled iteration: CNN denoise + CGDC."""
        x_k = torch.view_as_complex(x_k_ri.contiguous())
        z_k = self.dw(x_k)
        z_k_ri = torch.view_as_real(z_k)[:, None, None]
        x_k_ri = self.dc(x0_ri[:, None, None], z_k_ri, csm, mask)[:, 0, 0]
        return x_k_ri

    def forward(self, x0, csm, mask):
        """
        Parameters
        ----------
        x0 : complex64 [B, T, H, W]
        csm : [B, maps, coils, T, H, W, 2]
        mask : [B, 1, T, 1, W, 1]

        Returns
        -------
        x_k : complex64 [B, T, H, W]
        """
        x0_ri = torch.view_as_real(x0)
        x_k_ri = x0_ri.clone()

        for k in range(self.k_iters):
            if self.use_checkpoint and self.training:
                x_k_ri = checkpoint(
                    self._iteration, x_k_ri, x0_ri, csm, mask,
                    use_reentrant=False,
                )
            else:
                x_k_ri = self._iteration(x_k_ri, x0_ri, csm, mask)

        return torch.view_as_complex(x_k_ri.contiguous())
