"""
Abstract base LightningModule shared by every reconstruction model in this repo.
Defines MriModule (shared validation metrics/logging and incremental test h5 saving);
subclassed by all other pl_modules/*_module.py and loaded via configs/model/*.yaml.
"""
import os
import pathlib
import csv
import h5py

from typing import Optional
import numpy as np
import lightning as L
import torch

from helpers import metrics

import torchmetrics


class MriModule(L.LightningModule):
    """
    Abstract super class for deep learning reconstruction models.

    This is a subclass of the LightningModule class from Lightning
    (`lightning.pytorch`), with some additional functionality specific to fastMRI:
        - Evaluating reconstructions
        - Visualization

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, num_log_images: int = 16, save_val_images: bool = False, model_name: Optional[str] = None, test_savedir: Optional[str] = None, *args, **kwargs):
        """
        Args:
            num_log_images: Number of images to log. Defaults to 16.
        """
        super().__init__()

        # Collect hparams once for the whole inheritance tree: this runs inside
        # the child's __init__ chain, so the child's architecture arguments are
        # gathered here as well - subclasses must NOT call
        # save_hyperparameters() themselves (a second call would overwrite this
        # one and un-ignore the runtime options below).
        #
        # Why the ignore list matters: jsonargparse lets checkpoint hparams
        # override CLI values, so a stored `test_savedir: None` silently undoes
        # --model.init_args.test_savedir at test time (lost output locations,
        # tests writing to the wrong directory). Add any new test-only switch
        # to this list as well.
        self.save_hyperparameters(ignore=['model_name', 'save_val_images', 'test_savedir', 'ckpt_path'])

        self.num_log_images = num_log_images
        self.save_val_images = save_val_images
        self.model_name = model_name
        print(f"Initialized MriModule with model_name={model_name}, num_log_images={num_log_images}, save_val_images={save_val_images}, test_savedir={test_savedir}")
        self.val_log_indices = None
        self.validation_step_outputs = []
        self.test_logs = []
        self.test_savedir = test_savedir
        csv_name = f"{self.model_name}_val_batch_metrics.csv" if self.model_name else "val_batch_metrics.csv"
        self.val_batch_metrics_csv = pathlib.Path('val_output') / csv_name
        
        # evaluation metrics
        self.val_loss = torchmetrics.MeanMetric()
        self.nmse = metrics.NMSEMetric()
        self.ssim = metrics.SSIMMetric()
        self.psnr = metrics.PSNRMetric()
        self.hfen = metrics.HFENMetric()
        # self.ssim_xt = metrics.SSIMXTMetric() # NOTE: NOT SUPPORTED YET
        

    def on_train_epoch_end(self):
        pass

    def on_validation_batch_end(self, val_logs, batch, batch_idx, dataloader_idx=0):
        # check inputs
        for k in (
            "output",
            "target",
            "val_loss",
            'img_zf',
            "mask",
            'sens_maps'
        ):
            if k not in val_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            # Determine the number of batches to sample from
            limit_val_batches = self.trainer.limit_val_batches
            if isinstance(limit_val_batches, float) and limit_val_batches <= 1.0:
                num_val_batches = int(
                    limit_val_batches
                    * len(self.trainer.val_dataloaders) #.dataset)
                )
            else:
                num_val_batches = int(limit_val_batches)

            # Randomly sample indices, just running once, each log will be same images
            self.val_log_indices = list(
                np.random.permutation(num_val_batches)[: self.num_log_images]
            )

        if batch_idx in self.val_log_indices:
            key = f"val_images_idx_{batch_idx}"
            # only show the first frame for multi-frame dynamic data
            target_ = val_logs["target"][0,0].abs().unsqueeze(0) # [1, h, w]
            output_ = val_logs["output"][0,0].abs().unsqueeze(0) # [1, h, w]
            mask = val_logs["mask"][0,0].abs().unsqueeze(0).type_as(target_) # [1, h, w]
            img_zf = val_logs["img_zf"][0,0].abs().unsqueeze(0) # [1, h, w]
            sens_maps = val_logs["sens_maps"][0,0,0].abs().unsqueeze(0) # [1, h, w]
            error = torch.abs(target_ - output_)

            # Normalize the logged images
            # mask = mask / mask.max() # looks betetr if not normalized
            img_zf = img_zf / img_zf.max()
            sens_maps = sens_maps / sens_maps.max()
            output_ = output_ / output_.max()
            target_ = target_ / target_.max()
            error = error / error.max()
            ##* adjust contrast, make it bright
            alpha = 0.2
            self.log_image(key, [ output_**alpha, target_**alpha,error, mask, sens_maps, img_zf**alpha,], captions=[ 'reconstruction', 'target','error','mask','sens_maps','zf'])

        mask = val_logs['mask']
        acc_values = self._compute_batch_true_acc(mask)
        acc = acc_values.mean()
        maxval = [v.reshape(-1) for v in batch.max_value]
        target = val_logs['target'].abs()
        output = val_logs['output'].abs()
        self.val_loss.update(val_logs['val_loss'])
        self.nmse.update(batch.fname, batch.slice_num, target, output)
        self.ssim.update(batch.fname, batch.slice_num, target, output, maxvals=maxval)
        self.psnr.update(batch.fname, batch.slice_num, target, output, maxvals=maxval)
        self.hfen.update(batch.fname, batch.slice_num, target, output)

        # Compute per-sample PSNR for incremental CSV logging.
        batch_psnr_vals = []
        for i in range(target.shape[0]):
            maxval_i = torch.as_tensor(batch.max_value[i], device=target.device).reshape(-1)[0]
            batch_psnr_vals.append(metrics.psnr(target[i], output[i], maxval=maxval_i))
        batch_psnr_tensor = torch.stack(batch_psnr_vals)
        fnames = batch.fname
        if isinstance(fnames, str):
            fnames = [fnames]
        self._append_val_batch_metrics_csv(fnames, acc_values, batch_psnr_tensor)
        
        
        if self.save_val_images:
            self.validation_step_outputs.append({
                "fname": batch.fname,
                "slice": batch.slice_num,
                "output": val_logs["output"],
                "target": val_logs["target"],
                'mask': val_logs['mask'],
                'acc': acc
            })

    def log_image(self, key, images, captions):
        self.logger.log_image(key, images, caption=captions, step=self.global_step)

    def _compute_sample_true_acc(self, mask_sample: torch.Tensor) -> torch.Tensor:
        """Compute acceleration from mask after removing contiguous fully-sampled padding columns."""
        mask_sample = mask_sample.to(torch.float32)

        if mask_sample.ndim == 3:
            # [T, H, W] from validation logs; H is repeated by tile, so use one readout row.
            mask_phase = mask_sample[:, 0, :]
        elif mask_sample.ndim == 2:
            # [T, W]
            mask_phase = mask_sample
        else:
            mask_phase = mask_sample.reshape(mask_sample.shape[0], -1)

        col_all_one = torch.all(mask_phase > 0.5, dim=0)
        non_padding_cols = torch.where(~col_all_one)[0]

        if non_padding_cols.numel() > 0:
            left = int(non_padding_cols[0].item())
            right = int(non_padding_cols[-1].item()) + 1
            mask_phase = mask_phase[:, left:right]

        denom = torch.sum(mask_phase)
        if denom.item() == 0:
            return torch.as_tensor(float('nan'), device=mask_sample.device)

        return torch.as_tensor(mask_phase.numel(), device=mask_sample.device, dtype=torch.float32) / denom

    def _compute_batch_true_acc(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim < 2:
            raise RuntimeError(f"Expected mask with batch dimension, got shape={tuple(mask.shape)}")

        acc_vals = [self._compute_sample_true_acc(mask[i]) for i in range(mask.shape[0])]
        return torch.stack(acc_vals)

    def _append_val_batch_metrics_csv(self, fnames, acc_tensor: torch.Tensor, psnr_tensor: torch.Tensor):
        self.val_batch_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.val_batch_metrics_csv.exists()

        with self.val_batch_metrics_csv.open('a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['fname', 'acc', 'psnr'])

            psnr_list = psnr_tensor.detach().cpu().tolist()
            acc_list = acc_tensor.detach().cpu().tolist()
            n = min(len(fnames), len(acc_list), len(psnr_list))
            for i in range(n):
                writer.writerow([fnames[i], acc_list[i], psnr_list[i]])
                
    def on_validation_epoch_end(self):        
        # logging
        self.log('validation_loss', self.val_loss, prog_bar=True)
        self.log('val_metrics/nmse', self.nmse)
        self.log('val_metrics/ssim', self.ssim, prog_bar=True)
        self.log('val_metrics/psnr', self.psnr)
        self.log('val_metrics/hfen', self.hfen)
        
        if self.save_val_images:
            # save validation_step_outputs to a file for later analysis
            torch.save(self.validation_step_outputs,
                       os.path.join('val_output', f'{self.model_name}_val.pt'))
            self.validation_step_outputs.clear()

    def on_test_batch_end(self, test_log, batch, batch_idx, dataloader_idx=0):
        outputs = test_log["output"]
        fnames = test_log["fname"]
        source_splits = getattr(batch, "source_split", None)

        if isinstance(fnames, str):
            fnames = [fnames]
        if source_splits is None:
            source_splits = [''] * len(fnames)
        elif isinstance(source_splits, str):
            source_splits = [source_splits]

        n = min(len(fnames), len(outputs), len(source_splits))
        for i in range(n):
            split_name = source_splits[i] or 'unknown_split'
            save_path = self._resolve_test_save_dir(split_name)
            self._append_single_reconstruction_h5(save_path, str(fnames[i]), outputs[i])
        
    def on_test_epoch_end(self):
        self.print("Done! Reconstructions saved incrementally during test batches.")

    def _resolve_test_save_dir(self, split_name: str) -> pathlib.Path:
        if self.test_savedir is not None:
            base_path = pathlib.Path(self.test_savedir)
        else:
            base_path = pathlib.Path.cwd() / "reconstructions"
        save_path = base_path / split_name
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    def _append_single_reconstruction_h5(self, save_path: pathlib.Path, fname: str, output: torch.Tensor):
        file_name = f"{self.model_name}_reconstructions.h5" if self.model_name else "reconstructions.h5"
        file_path = save_path / file_name
        data = output.detach().cpu().numpy()

        with h5py.File(file_path, 'a') as hf:
            if fname in hf:
                del hf[fname]
            hf.create_dataset(fname, data=data)