"""CineVN package: VarNet with U-Net regularizers for 2D+T cine MRI reconstruction.

Reference: https://github.com/marcvornehm/CineVN

Implements the variational network of Sriram et al. (MICCAI 2020), after Hammernik et al.
(MRM 2018), adapted to 2D+T cine data. Referenced by pl_modules/varnet_module.py
(class_path: pl_modules.VarNetModule, configs/model/cinevn.yaml).
"""

from .unet import Unet2d, Unet2dPlusTime, Unet3d
from .varnet import NormUnet, VarNet
