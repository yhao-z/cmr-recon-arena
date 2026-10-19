"""Reconstruction export helpers for cmr-recon-arena.

``save_movie`` writes a frame stack to an animated GIF/APNG. It is called
from preprocessing/core/ocmr.py, preprocessing/core/cmrxrecon23.py and
helpers/mri_basic/pics_impl.py.
"""

import warnings
from pathlib import Path

import numpy as np
import skimage.exposure
import torch
from PIL import Image


def save_movie(
        image: np.ndarray | torch.Tensor, fname: Path | str, clip: bool = False, equalize_histogram: bool = False,
        tres: float = 50, vmin: float | None = None, vmax: float | None = None, gif: bool = True, apng: bool = True,
):
    """Save a stack of 2D frames as animated GIF and/or APNG images.

    ``image`` is a torch.Tensor or numpy array of shape (N, H, W); complex
    input (complex dtype or a trailing size-2 dimension) is converted to
    magnitude, then clipped/normalized to uint8 frames and written with PIL.
    """
    if clip and equalize_histogram:
        warnings.warn('Both clip and equalize_histogram are set to True. This is not recommended.')

    # to numpy array
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    else:
        image_np = image

    # to real-valued
    if image_np.shape[-1] == 2:
        image_np = image_np[..., 0] + 1j * image_np[..., 1]
    if np.iscomplexobj(image_np):
        image_np = np.abs(image_np)

    # clip, normalize, and equalize histogram
    if clip:
        image_np = np.clip(image_np, np.percentile(image_np, 3), np.percentile(image_np, 97))
    vmin_ = vmin or np.min(image_np)
    vmax_ = vmax or np.max(image_np)
    image_np = (image_np - vmin_) / (vmax_ - vmin_)
    if equalize_histogram:
        image_np = skimage.exposure.equalize_adapthist(image_np, clip_limit=0.02)
    image_np = (image_np * 255).astype(np.uint8)

    fname = Path(fname)
    images_pil = [Image.fromarray(img) for img in image_np]
    if gif:
        images_pil[0].save(fname.with_suffix('.gif'), save_all=True, append_images=images_pil[1:], duration=tres, loop=0)
    if apng:
        images_pil[0].save(fname.with_suffix('.apng'), save_all=True, append_images=images_pil[1:], duration=tres, loop=0)

