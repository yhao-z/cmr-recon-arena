"""
Lightning module wrapping models.CST_Net.CST for 2D+T cine reconstruction.
Defines CSTNetModule (16-frame windows, image + k-space L1 loss);
loaded by main.py through the class_path pl_modules.CSTNetModule in configs/model/CSTNet.yaml.
"""

import warnings
import torch
import torch.nn.functional as F

from helpers.mri_torch import center_crop_to_smallest
from models.CST_Net import CST

from .mri_module import MriModule
from helpers.mri_torch import sens_reduce, fftnc


class CSTNetModule(MriModule):
    """Lightning module for CST-Net (models.CST_Net.CST): L1 image + k-space loss, 16-frame windows."""
    def __init__(
        self,
        num_stages: list[int] = [2,2,2],
        inchans: int = 2,
        normalize_for_loss: bool = True,
        lr: float = 0.005,
        lr_step_size: int = 50,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
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

        # model
        self.net = CST(num_stages=num_stages, inchans = inchans)

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
            state_dict = {name[4:]: value for name, value in state_dict.items() if name.startswith("net.")}
            if self.net is not None:
                self.net.load_state_dict(state_dict)
            print(f"Loaded model from {ckpt_path}")


    def forward(self, img_zf: torch.Tensor, kspace: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        output = self.net(img_zf, kspace, mask, sens_maps) # [B, t, H, W, 2]
        output = output.contiguous() # [B, t, H, W, 2]
        return torch.view_as_complex(output)
    

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        if torch.any(batch.max_value == 0) or torch.any(batch.max_value.isnan()):
            raise ValueError('max_value must not be zero or nan during training or validation')

        # forward through network
        max_value = batch.max_value.type_as(batch.masked_kspace)
        target = batch.target / max_value
        masked_kspace = batch.masked_kspace / max_value
        mask = batch.mask
        img_zf = sens_reduce(masked_kspace, batch.sens_maps, fft_ndim=2)[:, 0, 0] # [B, t, H, W, 2]
        nt = img_zf.shape[1]
        
        # randomly crop 16 frames for training if nt > 16
        if nt > 16:
            t_rand = torch.randint(0, nt - 16, (1,)).item()
            img_zf = img_zf[:, t_rand:t_rand+16]
            target = target[:, t_rand:t_rand+16]
            masked_kspace = masked_kspace[:,:,t_rand:t_rand+16]
            mask = mask[:,:,t_rand:t_rand+16]
        # else, padding to 16 frames for training 
        else:
            pad_sz = 16 - nt
            img_zf = torch.cat([img_zf, img_zf[:, :pad_sz]], dim=1)
            assert img_zf.shape[1] == 16
            target = torch.cat([target, target[:, :pad_sz]], dim=1)
            masked_kspace = torch.cat([masked_kspace, masked_kspace[:,:,:pad_sz]], dim=2)
            mask = torch.cat([mask, mask[:,:,:pad_sz]], dim=2)
        
        output = self(img_zf, masked_kspace, mask, batch.sens_maps)

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

        loss_img = F.l1_loss(output_for_loss.abs(), target_for_loss.abs())
        k_rec = fftnc( torch.view_as_real(output_for_loss) )
        k_tar = fftnc( torch.view_as_real(target_for_loss) )
        loss_k = F.l1_loss(k_rec, k_tar)
        loss = loss_img + 0.5 * loss_k
        
        if torch.isnan(loss):
            warnings.warn('Skipping batch because training loss is NaN')
            return None
        
        self.log('train_loss', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> dict[str, torch.Tensor]:
        if torch.any(batch.max_value == 0) or torch.any(batch.max_value.isnan()):
            raise ValueError('max_value must not be zero or nan during training or validation')

        # forward through network
        max_value = batch.max_value.type_as(batch.masked_kspace)
        target = batch.target / max_value
        masked_kspace = batch.masked_kspace / max_value
        mask = batch.mask
        img_zf = sens_reduce(masked_kspace, batch.sens_maps, fft_ndim=2)[:, 0, 0] # [B, t, H, W, 2]
        nt = img_zf.shape[1]
        
        if nt > 16:
            output = torch.zeros_like(torch.view_as_complex(img_zf))
            output[:, :16] = self(img_zf[:, :16], masked_kspace[:, :, :16], mask[:, :, :16], batch.sens_maps)
            output[:, -16:] = self(img_zf[:, -16:], masked_kspace[:, :, -16:], mask[:, :, -16:], batch.sens_maps)
        else:
            pad_sz = 16 - nt
            img_zfp = torch.cat([img_zf, img_zf[:, :pad_sz]], dim=1)
            assert img_zfp.shape[1] == 16
            masked_kspacep = torch.cat([masked_kspace, masked_kspace[:,:,:pad_sz]], dim=2)
            maskp = torch.cat([mask, mask[:,:,:pad_sz]], dim=2)
            output = self(img_zfp, masked_kspacep, maskp, batch.sens_maps)[:,:nt]
            assert output.shape[1] == target.shape[1]

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

        loss_img = F.l1_loss(output_for_loss.abs(), target_for_loss.abs())
        k_rec = fftnc( torch.view_as_real(output_for_loss) )
        k_tar = fftnc( torch.view_as_real(target_for_loss) )
        loss_k = F.l1_loss(k_rec, k_tar)
        loss = loss_img + loss_k

        # processing other outputs for logging / visualization
        img_zf = torch.view_as_complex(img_zf)
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
        mask = batch.mask
        img_zf = sens_reduce(masked_kspace, batch.sens_maps, fft_ndim=2)[:, 0, 0] # [B, t, H, W, 2]
        nt = img_zf.shape[1]
        
        if nt > 16:
            output = torch.zeros_like(torch.view_as_complex(img_zf))
            output[:, :16] = self(img_zf[:, :16], masked_kspace[:, :, :16], mask[:, :, :16], batch.sens_maps)
            output[:, -16:] = self(img_zf[:, -16:], masked_kspace[:, :, -16:], mask[:, :, -16:], batch.sens_maps)
        else:
            pad_sz = 16 - nt
            img_zfp = torch.cat([img_zf, img_zf[:, :pad_sz]], dim=1)
            assert img_zfp.shape[1] == 16
            masked_kspacep = torch.cat([masked_kspace, masked_kspace[:,:,:pad_sz]], dim=2)
            maskp = torch.cat([mask, mask[:,:,:pad_sz]], dim=2)
            output = self(img_zfp, masked_kspacep, maskp, batch.sens_maps)[:,:nt]
            assert output.shape[1] == target.shape[1]

        # crop phase oversampling
        target, output = center_crop_to_smallest(target, output, ndim=2)
        target = target * max_value
        output = output * max_value
        
        return {'output': output, 'fname': batch.fname, 'slice': batch.slice_num}

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.StepLR]]:
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.lr_step_size, self.lr_gamma)

        return [optim], [scheduler]

