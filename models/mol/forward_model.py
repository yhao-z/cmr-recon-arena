"""
Forward model adapter: bridges MOL's `sense` interface to HGNN's sens_expand/sens_reduce.

Original: Sense.py + cg.py in Monotone-Operator-Learning
Provides MOLForwardModel with same interface as original `sense`:
    - adjoint(kspace, csm) -> image
    - ATA(image, csm, mask) -> image
    - inv(x0, rhs, lam, csm, mask) -> image (CG solve)

Part of models/mol; loaded by pl_modules/mol_module.py (from models.mol import MOL).
"""

import torch
import torch.nn as nn

from helpers.mri_torch import sens_expand, sens_reduce
from helpers.mri_torch.complex_math import complex_conj, complex_mul


def _to_complex(x_ri):
    """[.., 2] real/imag -> complex64."""
    return torch.view_as_complex(x_ri)


def _to_real(x_c):
    """complex -> [.., 2] real/imag."""
    return torch.view_as_real(x_c)


class MOLForwardModel(nn.Module):
    """
    Forward model adapter — same interface as original `sense` class.

    Parameters
    ----------
    cg_iter : int
        Number of conjugate gradient iterations for inv().
    """

    def __init__(self, cg_iter=5):
        super().__init__()
        self.cg_iter = cg_iter

    def forward(self, img, csm, mask):
        """
        image -> undersampled k-space.
        img: complex64 [B, T, H, W]
        Returns: [B, coils, T, H, W, 2]
        """
        img_ri = _to_real(img)[:, None]  # [B, 1, T, H, W, 2]
        kspace = sens_expand(img_ri, csm, fft_ndim=2)
        return kspace * mask

    def adjoint(self, kspace, csm):
        """
        k-space -> coil-combined image.
        kspace: [B, coils, T, H, W, 2]
        Returns: complex64 [B, T, H, W]
        """
        img_ri = sens_reduce(kspace, csm, fft_ndim=2)  # [B, maps, 1, T, H, W, 2]
        img_ri = img_ri[:, 0, 0]  # [B, T, H, W, 2]
        return _to_complex(img_ri)

    def ATA(self, img, csm, mask):
        """A^H A x: image -> k-space (masked) -> image."""
        kspace = self.forward(img, csm, mask)
        return self.adjoint(kspace, csm)

    def inv(self, x0, rhs, lam, csm, mask):
        """
        Solve (lam * A^H A + 1.001 * I) x = rhs via CG.
        Exact same formula as original sense.inv().

        All inputs complex64 [B, T, H, W].
        """
        mu = 1.001
        # lam_val = lam.detach() if isinstance(lam, torch.Tensor) else lam
        lam_val = lam

        x0_ri = _to_real(x0)   # [B, T, H, W, 2]
        rhs_ri = _to_real(rhs)  # [B, T, H, W, 2]

        def _zdot(a, b):
            """
            Batch-wise complex inner product.
            Sum over all dims EXCEPT batch (dim 0).
            Returns [B] shaped real tensor.
            Same as original: fn=lambda a,b: torch.abs(torch.sum(torch.conj(a)*b, axis=[-1,-2,-3]))
            but generalized for 4D or 5D tensors.
            """
            inner = complex_mul(complex_conj(a), b)
            # sum over all dims except batch
            sum_dims = tuple(range(1, inner.ndim))
            return torch.abs(inner.sum(sum_dims))

        def _lhs(p_ri):
            """lam * ATA(p) + mu * p"""
            ATA_p = _to_real(self.ATA(_to_complex(p_ri), csm, mask))
            return lam_val * ATA_p + mu * p_ri

        # CG iteration — identical to original cg.py cg_block.forward
        x = x0_ri
        r = rhs_ri - _lhs(x0_ri)
        p = r
        rTr = _zdot(r, r)  # [B]
        eps = torch.tensor(1e-10, device=r.device)

        # broadcast shape: [B] -> [B, 1, 1, 1] (4D) or [B, 1, 1, 1, 1] (5D)
        expand_shape = (None,) * (x.ndim - 1)

        for i in range(self.cg_iter):
            Ap = _lhs(p)
            alpha = rTr / (_zdot(p, Ap) + eps)  # [B]
            x = x + alpha[expand_shape] * p
            r = r - alpha[expand_shape] * Ap
            rTrNew = _zdot(r, r)  # [B]
            if torch.sum(torch.sqrt(rTrNew + eps)) < 1e-9:
                break
            beta = rTrNew / (rTr + eps)  # [B]
            p = r + beta[expand_shape] * p
            rTr = rTrNew

        return _to_complex(x)
