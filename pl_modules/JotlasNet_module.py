"""
Lightning module wrapping models.JotlasNet.JotlasNet for 2D+T cine reconstruction.
Defines JotlasNetModule (niter iterations, L1 loss on magnitude);
loaded by main.py through the class_path pl_modules.JotlasNetModule in configs/model/jotlasnet.yaml.
"""

import warnings
from typing import Iterable

import torch

from helpers import losses
from helpers.mri_torch import center_crop_to_smallest
from models.JotlasNet import JotlasNet

from .mri_module import MriModule
from helpers.mri_torch import sens_reduce


class JotlasNetModule(MriModule):
    """Lightning module for JotlasNet (models.JotlasNet.JotlasNet)."""
    def __init__(
        self,
        niter: int = 15,
        chans: int = 16,
        normalize_for_loss: bool = True,
        lr: float = 0.005,
        lr_step_size: int = 50,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        loss_fns: Iterable[str] = ('ssim', 'perp'),
        ssim_mode: str = '3d',
        ckpt_path: str = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)


        self.normalize_for_loss = normalize_for_loss
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.ssim_mode = ssim_mode

        # model
        # the mask argument is passed to forward(), not stored on the module
        self.niter = niter
        self.chans = chans
        self.net = JotlasNet(niter, chans)

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
            state_dict = {name[4:]: value for name, value in state_dict.items() if name.startswith("net.")}
            if self.net is not None:
                self.net.load_state_dict(state_dict)
            print(f"Loaded model from {ckpt_path}")

        # loss functions
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
                    raise RuntimeError('Unknown loss type')

            if any([isinstance(loss, type(loss_fn)) for loss in losses_list]):
                warnings.warn(
                    f'Loss function of type {type(loss_fn)} already exists. Are you sure you want to add it twice?',
                    category=UserWarning,
                )
            losses_list.append(loss_fn)
        self.loss = losses.MultiTaskLoss(losses_list)

    def forward(self, img_zf: torch.Tensor, masked_kspace:torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        output = self.net(img_zf, masked_kspace, mask, sens_maps)
        return output

    def train_val_forward(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.any(batch.max_value == 0) or torch.any(batch.max_value.isnan()):
            raise ValueError('max_value must not be zero or nan during training or validation')

        # forward through network
        max_value = batch.max_value.type_as(batch.masked_kspace)
        target = batch.target / max_value
        masked_kspace = batch.masked_kspace / max_value
        img_zf = sens_reduce(masked_kspace, batch.sens_maps, fft_ndim=2)
        output = self(img_zf, masked_kspace, batch.mask, batch.sens_maps)

        # crop phase oversampling
        target, output = center_crop_to_smallest(target, output, ndim=2)
        target = target * max_value
        output = output * max_value

        # compute loss
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

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        _, _, loss = self.train_val_forward(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> dict[str, torch.Tensor]:

        target, output, loss = self.train_val_forward(batch)
        
        # processing other outputs for logging / visualization
        img_zf = sens_reduce(batch.masked_kspace, batch.sens_maps, fft_ndim=2)
        img_zf = torch.view_as_complex(img_zf)
        img_zf = img_zf[:, 0, 0]
        img_zf = center_crop_to_smallest(target, img_zf, ndim=2)[1]  # [B, t, H, W]
        mask = batch.mask.squeeze_(-1).squeeze_(1).tile(1, 1, img_zf.shape[-2], 1)  # [B, t, H, W]
        sens_maps = torch.view_as_complex(batch.sens_maps)[:, 0]  # [B, C, 1(t), H, W]
        
        return {
            "output": output,
            "target": target,
            "val_loss": loss,
            "img_zf": img_zf,
            "mask": mask, 
            "sens_maps": sens_maps
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> dict[str, torch.Tensor]:
        # forward through network
        max_value = batch.max_value.type_as(batch.masked_kspace)
        target = batch.target / max_value
        masked_kspace = batch.masked_kspace / max_value
        img_zf = sens_reduce(masked_kspace, batch.sens_maps, fft_ndim=2)
        output = self(img_zf, masked_kspace, batch.mask, batch.sens_maps)
        
        # crop phase oversampling
        if target.numel() > 1:  # no reference crop when the split has no GT (empty(2) marker)
            target, output = center_crop_to_smallest(target, output, ndim=2)
        target = target * max_value
        output = output * max_value
        
        return {'output': output, 'fname': batch.fname, 'slice': batch.slice_num}

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.StepLR]]:
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.lr_step_size, self.lr_gamma)

        return [optim], [scheduler]

