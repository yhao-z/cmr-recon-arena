"""
Spectral Normalization — NOT USED by default in MOL DEQ.

Original trn_mol.py uses spectral_norm=False (sn=False).
This file is kept as a placeholder for future use when monotonicity
guarantees are needed. Currently convlayer passes sn=False through.

If spectral_norm=True is needed in the future, the implementation below
supports both Conv2d and Conv3d via power iteration.

Part of models/mol; loaded by pl_modules/mol_module.py (from models.mol import MOL).
"""

import torch
from torch.nn.functional import conv2d, conv3d


def normalize(tensor, eps=1e-12):
    """L2-normalize a tensor, clamping the norm at `eps`."""
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, eps)
    return tensor / norm


class SpectralNorm(object):
    """
    Spectral normalization via power iteration.
    Auto-detects Conv2d (4D weight) vs Conv3d (5D weight).

    NOTE: This is NOT used in the default MOL DEQ configuration (sn=False).
    Kept for future experiments with strict monotonicity constraints.
    """

    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive')
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        """One power iteration for 4D (Conv2d) or 5D (Conv3d) weights -> (normalized weight, updated u)."""
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        is_3d = (weight.dim() == 5)

        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                if is_3d:
                    v = normalize(
                        conv3d(u.flip(3, 4), weight.permute(1, 0, 2, 3, 4), padding=2),
                        eps=self.eps
                    ).flip(3, 4)[:, :, :, 1:-1, 1:-1]
                    u = normalize(conv3d(v, weight, padding=1), eps=self.eps)
                else:
                    v = normalize(
                        conv2d(u.flip(2, 3), weight.permute(1, 0, 2, 3), padding=2),
                        eps=self.eps
                    ).flip(2, 3)[:, :, 1:-1, 1:-1]
                    u = normalize(conv2d(v, weight, padding=1), eps=self.eps)
                if self.n_power_iterations > 0:
                    u = u.clone()
                    v = v.clone()

        if is_3d:
            sigma = torch.sum(u * conv3d(v, weight, padding=1))
        else:
            sigma = torch.sum(u * conv2d(v, weight, padding=1))
        weight = weight / sigma
        return weight, u

    def remove(self, module):
        """Restore the plain `weight` parameter on `module` and drop the spectral-norm buffers."""
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        """Forward pre-hook: recompute the weight while training, detach it in eval mode."""
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        """Install spectral norm on `module`: register the _orig parameter and the u buffer, add the forward pre-hook."""
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        C_out = weight.shape[0]
        is_3d = (weight.dim() == 5)
        if is_3d:
            u = normalize(weight.new_empty(1, C_out, 1, 40, 40).normal_(0, 1), eps=fn.eps)
        else:
            u = normalize(weight.new_empty(1, C_out, 40, 40).normal_(0, 1), eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    """Apply spectral normalization. Auto-detects Conv2d vs Conv3d."""
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module, name='weight'):
    """Remove the SpectralNorm pre-hook named `name` from `module` (raises if it is absent)."""
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))
