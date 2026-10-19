"""
Lightning module for Monotone Operator Learning (MOL) DEQ.

All loss, optimizer, and hyperparameter settings match trn_mol.py exactly.
Normalization strategy follows ISTA-Net-Plus: input / max_value -> network -> output * max_value.
Loaded by main.py through the class_path pl_modules.MOLModule in configs/model/mol.yaml.
"""

from typing import Iterable

import torch
import torch.nn as nn

from helpers.mri_torch import center_crop_to_smallest, sens_reduce
from models.mol import MOL

from .mri_module import MriModule


# ---------- trn_mol.py helpers (IDENTICAL) ----------

def contdifflog(x, T, e):
    """Continuously differentiable log barrier — from trn_mol.py line 34-42."""
    if not isinstance(T, torch.Tensor):
        T = torch.tensor(T, device=x.device)
    if not isinstance(e, torch.Tensor):
        e = torch.tensor(e, device=x.device)
    condition = (x < (T - e)) * torch.tensor(1.0)
    y1 = -torch.log((T - x) * condition + (1.0 - condition))
    y2 = -torch.log(e) + (1 / e) * (x - T + e)
    return y1 * condition + y2 * (1.0 - condition)


class LipschitzEstimator:
    """Adversarial Lipschitz estimation — from lipschitz_cnn.py."""

    def __init__(self, u, eps, shape, model, lr=1e0, device='cuda'):
        self.model = model
        self.lr = lr
        self.eps = eps
        self.u = u.to(device)
        self.eps_tensor = torch.tensor(eps, dtype=torch.float32).to(device)
        self.v = torch.complex(
            torch.rand(shape, dtype=torch.float32),
            torch.rand(shape, dtype=torch.float32)
        ).to(device)
        self.v = self.u + self.eps_tensor * self.v
        self.v = self.v.requires_grad_(True)
        self.device = device

    def _l2_norm(self, x):
        return torch.norm(x.view(x.shape[0], -1), p=2, dim=1)

    def compute_ratio(self):
        u_out = self.model(self.u)
        v_out = self.model(self.v)
        return self._l2_norm(u_out - v_out) / self._l2_norm(self.u - self.v)

    def adverserial_update(self, iters=1):
        """Projected gradient ascent maximizing ||f(u)-f(v)||/||u-v||; returns the ratio."""
        self.v = self.v.requires_grad_(True)
        for i in range(iters):
            loss = self.compute_ratio()
            loss_sum = torch.sum(loss)
            loss_sum.backward()
            v_grad = self.v.grad.detach()
            v_tmp = self.v.data + self.lr * v_grad
            v_tmp = (v_tmp / torch.norm(v_tmp)) * torch.norm(self.u)
            self.v.grad.zero_()
            self.v.data = v_tmp
        self.v = self.v.requires_grad_(False)
        return self.compute_ratio()


# ---------- Lightning Module ----------

class MOLModule(MriModule):
    """
    MOL DEQ training module.

    Normalization: input / max_value -> MOL -> output * max_value (ISTA-Net-Plus style).
    Loss: MSE on |prediction| vs |target| + Lipschitz clipping (trn_mol.py exact).
    """

    def __init__(
            self,
            # MOL architecture — trn_mol.py defaults
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
            # Training — trn_mol.py defaults
            learning_rate: float = 1e-3,
            learning_rate_lam: float = 1.0,
            lr_step_size: int = 1,
            lr_gamma: float = 0.3,
            # Lipschitz — trn_mol.py defaults
            learning_rate_lipschitz: float = 1e0,
            eps: float = 0.01,
            T: float = 0.98,
            e: float = 1e-3,
            clip_wt: float = 1e-6,
            lipschitz_iters: int = 10,
            # Other
            normalize_for_loss: bool = True,
            loss_fns: Iterable[str] = ('mse',),
            ssim_mode: str = '3d',
            compile_model: bool = False,
            ckpt_path: str | None = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.normalize_for_loss = normalize_for_loss
        self.learning_rate = learning_rate
        self.learning_rate_lam = learning_rate_lam
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.ssim_mode = ssim_mode
        self.learning_rate_lipschitz = learning_rate_lipschitz
        self.eps = eps
        self.T = T
        self.e = e
        self.clip_wt = clip_wt
        self.lipschitz_iters = lipschitz_iters

        self.mol = MOL(
            num_iters=num_iters, num_cg_init=num_cg_init, num_cg_iters=num_cg_iters,
            features=features, layers=layers, lam_init=lam_init, lam=lam,
            alpha=alpha, spectral_norm=spectral_norm, tol=tol, verbose=verbose,
        )

        if compile_model:
            try:
                self.mol = torch.compile(self.mol)
            except RuntimeError:
                pass

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
            state_dict = {name[4:]: value for name, value in state_dict.items() if name.startswith("mol.")}
            self.mol.load_state_dict(state_dict, strict=True)
            print(f'Loaded MOL model weights from {ckpt_path}')

        self.loss_fn = nn.MSELoss()

    def forward(self, masked_kspace, mask, sens_maps):
        return self.mol(masked_kspace, mask, sens_maps)

    def _run_step(self, batch):
        """Shared train/val logic: normalize -> forward -> denormalize -> loss."""
        max_value = batch.max_value.type_as(batch.masked_kspace)

        # Normalize input (ISTA-Net-Plus style)
        masked_kspace_norm = batch.masked_kspace / max_value
        target_norm = batch.target / max_value

        # Forward
        reconstruction, errforward, nFE = self(masked_kspace_norm, batch.mask, batch.sens_maps)

        # Crop phase oversampling
        target_crop, output_crop = center_crop_to_smallest(target_norm, reconstruction, ndim=2)

        # Denormalize
        target = target_crop * max_value
        output = output_crop * max_value

        # MSE loss on magnitude (trn_mol.py exact)
        if self.normalize_for_loss:
            norm_val = target.abs().flatten(1).max(dim=1).values
            target_for_loss = target / norm_val[:, None, None, None]
            output_for_loss = output / norm_val[:, None, None, None]
        else:
            target_for_loss = target
            output_for_loss = output
        mse_loss = self.loss_fn(torch.abs(output_for_loss), torch.abs(target_for_loss))

        return mse_loss, output, target, errforward, nFE

    def training_step(self, batch, batch_idx):
        if torch.any(batch.max_value == 0) or torch.any(batch.max_value.isnan()):
            raise ValueError('max_value must not be zero or nan during training')

        mse_loss, output, target, errforward, nFE = self._run_step(batch)

        # Lipschitz estimation (trn_mol.py exact)
        u = torch.clone(output).detach()
        lip_pred = LipschitzEstimator(
            u, self.eps, u.shape, self.mol.optBlock.dw,
            lr=self.learning_rate_lipschitz, device=output.device,
        )
        lip_pred_est = lip_pred.adverserial_update(iters=self.lipschitz_iters)

        # Total loss
        clip_loss = self.clip_wt * contdifflog(lip_pred_est, self.T, self.e)
        loss = mse_loss + clip_loss

        self.log('train_loss', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        self.log('train/mse_loss', mse_loss, on_step=True, on_epoch=False)
        self.log('train/clip_loss', clip_loss, on_step=True, on_epoch=False)
        self.log('params/lam', self.mol.optBlock.lam.data, on_step=True, on_epoch=False)
        self.log('params/lip_pred', lip_pred_est.detach(), on_step=True, on_epoch=False)
        self.log('params/nFE', nFE, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if torch.any(batch.max_value == 0) or torch.any(batch.max_value.isnan()):
            raise ValueError('max_value must not be zero or nan during validation')

        mse_loss, output, target, errforward, nFE = self._run_step(batch)

        # ZF image for logging
        img_zf = sens_reduce(batch.masked_kspace, batch.sens_maps, fft_ndim=2)
        img_zf = torch.view_as_complex(img_zf)[:, 0, 0]
        img_zf = center_crop_to_smallest(target, img_zf, ndim=2)[1]
        mask = batch.mask.squeeze_(-1).squeeze_(1).tile(1, 1, img_zf.shape[-2], 1)
        sens_maps = torch.view_as_complex(batch.sens_maps)[:, 0]

        return {
            "output": output, "target": target, "val_loss": mse_loss,
            "img_zf": img_zf, "mask": mask, "sens_maps": sens_maps,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        mse_loss, output, target, errforward, nFE = self._run_step(batch)
        return {'output': output, 'fname': batch.fname, 'slice': batch.slice_num}

    def configure_optimizers(self):
        # Exact trn_mol.py: Adam lr=1e-3 for dw, lr=1 for lam; StepLR(1, 0.3)
        optimizer = torch.optim.Adam([
            {'params': self.mol.dw.parameters(), 'lr': self.learning_rate},
            {'params': self.mol.optBlock.lam, 'lr': self.learning_rate_lam},
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )
        return [optimizer], [scheduler]
