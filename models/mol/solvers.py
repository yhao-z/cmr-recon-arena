"""
DEQ solver for MOL.

Original: solvers.py in Monotone-Operator-Learning
Only DEQ is migrated — this is the only solver used in trn_mol.py.
Unroll/DEQ_inf/Unroll_admm/Direct_inversion/Sense_block are NOT used.

DEQ logic (IDENTICAL to original):
    1. SENSE initialization: x = A_init.inv(0, lam_init * Atb, lam_init, csm, mask)
    2. Fixed-point iteration (no grad): x = f(x, Atb, csm, mask) until convergence
    3. One more forward pass tracked by PyTorch
    4. Jacobian-free backprop via implicit function theorem

Part of models/mol; loaded by pl_modules/mol_module.py (from models.mol import MOL).
"""

import torch
import torch.nn as nn
import torch.autograd as autograd


class DEQ(nn.Module):
    """
    Deep Equilibrium solver for MOL.

    Memory-efficient: only stores the fixed point, not intermediate activations.
    Backward pass uses implicit differentiation (Jacobian-free).

    Parameters (same as original):
        f: solver block (fwdbwdBlock)
        K: max iterations
        A_init: forward model for SENSE initialization (more CG iters)
        lam_init: regularization weight for SENSE init
        tol: convergence tolerance
        verbose: print convergence info
    """

    def __init__(self, f, K, A_init, lam_init, tol=0.05, verbose=True):
        super().__init__()
        self.K = K
        self.f = f
        self.A_init = A_init
        self.lam_init = lam_init
        self.tol = tol
        self.verbose = verbose

    def forward(self, b, csm, mask):
        """
        DEQ forward pass.

        b:    [B, coils, T, H, W, 2]  (measured k-space)
        csm:  [B, maps, coils, T, H, W, 2]
        mask: [B, 1, T, 1, W, 1]

        Returns: (x, sense_out, errforward, nFE)
            x: complex64 [B, T, H, W]  (reconstructed image)
            sense_out: complex64 [B, T, H, W]  (SENSE initialization)
            errforward: float  (convergence error)
            nFE: int  (number of function evaluations)
        """
        # SENSE initialization
        Atb = self.f.A.adjoint(b, csm)
        zero = torch.zeros_like(Atb).to(Atb.device)
        sense_out = self.A_init.inv(zero, self.lam_init * Atb, self.lam_init, csm, mask)
        x = sense_out

        # Fixed-point iteration (no gradients tracked)
        with torch.no_grad():
            for blk in range(self.K):
                xold = x
                x = self.f(x, Atb, csm, mask)
                errforward = torch.norm(x - xold) / torch.norm(xold)
                if self.verbose:
                    print(errforward)
                    print("diff", torch.norm(x - xold).cpu().numpy(),
                          " xnew ", torch.norm(x).cpu().numpy(),
                          " xold ", torch.norm(xold).cpu().numpy())
                if errforward < self.tol and blk > 2:
                    if self.verbose:
                        print("exiting front prop after ", blk, " iterations with ", errforward)
                    break

        # One more forward pass — tracked by PyTorch for backward
        z = self.f(x, Atb, csm, mask)

        # Jacobian-free backprop via implicit function theorem
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, Atb, csm, mask)

        def backward_hook(grad):
            """Neumann-series solve for the implicit backward: g <- J^T g + grad until the change falls below tol."""
            g = grad
            for i in range(self.K):
                gold = g
                g = autograd.grad(f0, z0, gold, retain_graph=True)[0] + grad
                errback = torch.norm(g - gold) / torch.norm(gold)
                if errback < self.tol:
                    if self.verbose:
                        print("exiting back prop after ", i, " iterations with ", errback)
                    break
            g = autograd.grad(f0, z0, gold)[0] + grad
            return g

        z = z.requires_grad_(True)
        z.register_hook(backward_hook)

        return z, sense_out, errforward, blk
