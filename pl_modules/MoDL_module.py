"""
Lightning module for MoDL.

Reference: https://github.com/bo-10000/MoDL_PyTorch
Settings match reference exactly:
  Loss: MSE (nn.MSELoss)
  Optimizer: Adam, lr=0.001
  Gradient clipping: clip_value=1.0
  Normalization: input / max_value -> network -> output * max_value (ISTA-Net-Plus style)
Loaded by main.py through the class_path pl_modules.MoDLModule in configs/model/modl.yaml.
"""

import warnings
from typing import Iterable

import torch
import torch.nn as nn

from helpers import losses
from helpers.mri_torch import center_crop_to_smallest, sens_reduce
from models.MoDL import MoDL

from .mri_module import MriModule


class MoDLModule(MriModule):
    """Lightning module for MoDL reconstruction (models.MoDL.MoDL)."""
    def __init__(
        self,
        # MoDL architecture (reference defaults)
        n_layers: int = 5,
        k_iters: int = 10,
        cg_iter: int = 10,
        autograd: bool = True,
        use_checkpoint: bool = False,
        # Training (reference: Adam lr=0.001, no scheduler, grad clip=1.0)
        lr: float = 0.001,
        weight_decay: float = 0.0,
        gradient_clip: float = 1.0,
        # Loss (reference: MSE)
        normalize_for_loss: bool = True,
        loss_fns: Iterable[str] = ('mse',),
        ssim_mode: str = '3d',
        ckpt_path: str = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.normalize_for_loss = normalize_for_loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.ssim_mode = ssim_mode

        # model
        self.net = MoDL(n_layers=n_layers, k_iters=k_iters, cg_iter=cg_iter, autograd=autograd, use_checkpoint=use_checkpoint)

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
            state_dict = {name[4:]: value for name, value in state_dict.items() if name.startswith("net.")}
            self.net.load_state_dict(state_dict)
            print(f"Loaded model from {ckpt_path}")

        # loss functions (reference: nn.MSELoss)
        losses_list = []
        for loss_fn_str in loss_fns:
            match loss_fn_str.lower():
                case 'ssim':
                    loss_fn = losses.SSIMLoss(mode=self.ssim_mode)
                case 'perp' | 'perpendicular':
                    loss_fn = losses.PerpendicularLoss()
                case 'mse':
                    loss_fn = losses.ComplexMSELoss()
                case 'hfen':
                    loss_fn = losses.HFENLoss()
                case _:
                    raise RuntimeError(f'Unknown loss type: {loss_fn_str}')

            if any([isinstance(loss, type(loss_fn)) for loss in losses_list]):
                warnings.warn(
                    f'Loss function of type {type(loss_fn)} already exists.',
                    category=UserWarning,
                )
            losses_list.append(loss_fn)
        self.loss = losses.MultiTaskLoss(losses_list)

    def forward(self, img_zf, csm, mask):
        return self.net(img_zf, csm, mask)

    def train_val_forward(self, batch):
        if torch.any(batch.max_value == 0) or torch.any(batch.max_value.isnan()):
            raise ValueError('max_value must not be zero or nan')

        max_value = batch.max_value.type_as(batch.masked_kspace)

        # Normalize input (ISTA-Net-Plus style)
        target = batch.target / max_value
        masked_kspace = batch.masked_kspace / max_value

        # Zero-filled image (complex64 [B, T, H, W])
        img_zf = sens_reduce(masked_kspace, batch.sens_maps, fft_ndim=2)
        img_zf = torch.view_as_complex(img_zf)[:, 0, 0]

        # Forward
        output = self(img_zf, batch.sens_maps, batch.mask)

        # Crop phase oversampling
        target, output = center_crop_to_smallest(target, output, ndim=2)

        # Denormalize
        target = target * max_value
        output = output * max_value

        # Compute loss
        if self.normalize_for_loss:
            norm_val = target.abs().flatten(1).max(dim=1).values
            target_for_loss = target / norm_val[:, None, None, None]
            output_for_loss = output / norm_val[:, None, None, None]
            data_range = batch.max_value / norm_val
        else:
            target_for_loss = target
            output_for_loss = output
            data_range = batch.max_value

        loss = self.loss(pred=output_for_loss[:, None], targ=target_for_loss[:, None], data_range=data_range)
        return target, output, loss

    def training_step(self, batch, batch_idx):
        _, _, loss = self.train_val_forward(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        return loss

    def on_after_backward(self):
        # Gradient clipping (reference: clip_value=1.0)
        if self.gradient_clip > 0:
            nn.utils.clip_grad_value_(self.parameters(), clip_value=self.gradient_clip)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        target, output, loss = self.train_val_forward(batch)

        img_zf = sens_reduce(batch.masked_kspace, batch.sens_maps, fft_ndim=2)
        img_zf = torch.view_as_complex(img_zf)[:, 0, 0]
        img_zf = center_crop_to_smallest(target, img_zf, ndim=2)[1]
        mask = batch.mask.squeeze_(-1).squeeze_(1).tile(1, 1, img_zf.shape[-2], 1)
        sens_maps = torch.view_as_complex(batch.sens_maps)[:, 0]

        return {
            "output": output,
            "target": target,
            "val_loss": loss,
            "img_zf": img_zf,
            "mask": mask,
            "sens_maps": sens_maps,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        max_value = batch.max_value.type_as(batch.masked_kspace)
        masked_kspace = batch.masked_kspace / max_value
        img_zf = sens_reduce(masked_kspace, batch.sens_maps, fft_ndim=2)
        img_zf = torch.view_as_complex(img_zf)[:, 0, 0]
        output = self(img_zf, batch.sens_maps, batch.mask)
        target, output = center_crop_to_smallest(batch.target / max_value, output, ndim=2)
        output = output * max_value
        return {'output': output, 'fname': batch.fname, 'slice': batch.slice_num}

    def configure_optimizers(self):
        # Reference: Adam lr=0.001, no scheduler
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.1)
        return [optim], [scheduler]
