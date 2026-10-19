"""U-Net regularizers of the cinevn VarNet (Unet2d / Unet2dPlusTime / Unet3d, NormUnet).
Standard U-Net with residual blocks and flexible per-dimension padding; used as the
per-cascade regularizer by models/cinevn/varnet.py, which is used by
pl_modules/varnet_module.py (class_path: pl_modules.VarNetModule, configs/model/cinevn.yaml).
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as func

from . import conv


class _Unet(nn.Module, ABC):
    """Abstract U-Net: encoder, bottleneck and decoder assembled by _setup() from subclass hooks; operates on complex tensors with the real/imag axis last ([b, c, (t), h, w, 2])."""
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            chans: int = 32,
            num_pool_layers: int = 4,
            drop_prob: float = 0.0,
            conv_size: int = 3,
            residual_blocks: bool = True,
            two_convs_per_block: bool = True,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.conv_size = conv_size
        self.residual_blocks = residual_blocks
        self.two_convs_per_block = two_convs_per_block

        self.padder: nn.Module
        self.enc_conv_blocks: nn.ModuleList
        self.bottleneck: nn.Module
        self.dec_transpose_convs: nn.ModuleList
        self.dec_conv_blocks: nn.ModuleList
        self.out: nn.Module

    def _setup(self) -> None:
        """Build the padder, encoder, bottleneck, decoder and output conv from the subclass _setup_* hooks."""
        # padder
        self.padder = self._setup_padder()

        # encoder
        self.enc_conv_blocks = nn.ModuleList([self._setup_conv_block(self.in_chans, self.chans)])
        ch = self.chans
        for _ in range(self.num_pool_layers - 1):
            self.enc_conv_blocks.append(self._setup_conv_block(ch, ch * 2))
            ch = ch * 2

        # bottleneck
        self.bottleneck = self._setup_conv_block(ch, ch * 2)
        ch = ch * 2

        # decoder
        self.dec_transpose_convs = nn.ModuleList()
        self.dec_conv_blocks = nn.ModuleList()
        for _ in range(self.num_pool_layers):
            self.dec_transpose_convs.append(self._setup_transpose_conv_block(ch, ch // 2))
            self.dec_conv_blocks.append(self._setup_conv_block(ch, ch // 2))
            ch = ch // 2

        # out conv
        self.out = self._setup_out_conv(ch)

    @abstractmethod
    def _setup_padder(self) -> nn.Module:
        """Build the Padder that pads the input to a multiple of 2**(num_pool_layers + 1)."""
        raise NotImplementedError

    @abstractmethod
    def _setup_conv_block(self, in_chans: int, out_chans: int) -> nn.Module:
        """Build one encoder/decoder conv block mapping `in_chans` -> `out_chans` channels."""
        raise NotImplementedError

    @abstractmethod
    def _setup_transpose_conv_block(self, in_chans: int, out_chans: int) -> nn.Module:
        """Build the upsampling block mapping `in_chans` -> `out_chans` channels."""
        raise NotImplementedError

    @abstractmethod
    def _setup_out_conv(self, in_chans: int) -> nn.Module:
        """Build the final 1x1 convolution producing `out_chans` channels."""
        raise NotImplementedError

    @abstractmethod
    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Subclass hook: downsample the feature map by 2 (abstract; implemented by the concrete U-Nets)."""
        raise NotImplementedError

    @staticmethod
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        """[b, c, (t), h, w, 2] -> [b, 2*c, (t), h, w]: move the complex axis into the channel axis."""
        x = x.movedim(-1, 1).flatten(1, 2)  # [b, 2*c, (t), h, w]
        return x

    @staticmethod
    def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
        """[b, 2*c, (t), h, w] -> [b, c, (t), h, w, 2] (inverse of complex_to_chan_dim)."""
        target_shape = (x.shape[0], 2, x.shape[1] // 2, *x.shape[2:])
        x = x.reshape(target_shape)  # [b, 2, c, (t), h, w]
        x = x.movedim(1, -1)  # [b, c, (t), h, w, 2]
        return x.contiguous()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: complex [b, c, (t), h, w, 2] -> same layout (spatial pad -> encode -> decode -> unpad)."""
        enc_features_stack = []
        output = image

        # complex as channel dimension
        output = self.complex_to_chan_dim(output)

        # pad
        output = self.padder(output)

        # encoder
        for conv_block in self.enc_conv_blocks:
            output = conv_block(output)
            enc_features_stack.append(output)
            output = self._pool(output)

        # bottleneck
        output = self.bottleneck(output)

        # decoder
        for transpose_conv, conv_block in zip(self.dec_transpose_convs, self.dec_conv_blocks):
            # upsample
            output = transpose_conv(output)

            # get encoder features through skip connection
            enc_features = enc_features_stack.pop()

            # reflect pad on the right/bottom/(back) if needed to handle odd input dimensions
            padding = (
                0, enc_features.shape[-1] - output.shape[-1],  # padding right
                0, enc_features.shape[-2] - output.shape[-2],  # padding bottom
            )
            if output.ndim == 5:
                padding += (0, enc_features.shape[-3] - output.shape[-3])  # padding back
            output = func.pad(output, padding, 'reflect')

            # concatenate
            output = torch.cat([output, enc_features], dim=1)

            # conv block
            output = conv_block(output)

        # out conv
        output = self.out(output)

        # unpad
        output = self.padder(output, unpad=True)

        # channel as complex dimension
        output = self.chan_complex_to_last_dim(output)

        return output


class Unet2d(_Unet):
    """2D U-Net; a temporal axis (if present) is folded into the batch so convolutions run per frame."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._setup()

    def _setup_padder(self) -> nn.Module:
        """Build the Padder that pads the input to a multiple of 2**(num_pool_layers + 1)."""
        size_multiple_of = 2 ** (self.num_pool_layers + 1)  # tensor size should be at least two at the bottleneck
        return Padder(size_multiple_of=size_multiple_of, pad_temporal=False)

    def _setup_conv_block(self, in_chans: int, out_chans: int) -> nn.Module:
        """Build one encoder/decoder conv block mapping `in_chans` -> `out_chans` channels."""
        return ConvBlock2d(in_chans, out_chans, self.drop_prob, conv_size=self.conv_size,
                           residual_blocks=self.residual_blocks, two_convs_per_block=self.two_convs_per_block)

    def _setup_transpose_conv_block(self, in_chans: int, out_chans: int) -> nn.Module:
        """Build the upsampling block mapping `in_chans` -> `out_chans` channels."""
        return TransposeConvBlock2d(in_chans, out_chans)

    def _setup_out_conv(self, in_chans: int) -> nn.Module:
        """Build the final 1x1 convolution producing `out_chans` channels."""
        return nn.Conv2d(in_chans, self.out_chans, kernel_size=(1, 1), stride=(1, 1), padding_mode='replicate')

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """2x2 average pooling with stride 2 (any temporal axis is already folded into the batch)."""
        return func.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """[b, c, t, h, w, 2] (6D: frames folded into the batch) or [b, c, h, w, 2] -> same layout as the input."""
        # move temporal dimension to batch dimension
        in_shape = image.shape
        if len(in_shape) == 6:
            image = image.movedim(2, 1)  # [b, t, c, h, w, two]
            image = image.reshape((-1, in_shape[1], *in_shape[-3:]))  # [b*t, c, h, w, two]

        # forward
        image = super().forward(image)

        # move temporal dimension back
        if len(in_shape) == 6:
            image = image.reshape((-1, in_shape[2], in_shape[1], *in_shape[-3:]))  # [b, t, c, h, w, two]
            image = image.movedim(1, 2)  # [b, c, t, h, w, two]

        return image


class _UnetSpatiotemporal(_Unet, ABC):
    """U-Net variant for 2D+t/3D convolutions: adds `pool_temporal`, `conv_size_temp` and `pad_mode_temp` options."""
    def __init__(
            self,
            *args,
            pool_temporal: bool = True,
            conv_size_temp: int = 3,
            pad_mode_temp: str = 'circular',
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.pool_temporal = pool_temporal
        self.conv_size_temp = conv_size_temp
        self.pad_mode_temp = pad_mode_temp

        self._setup()

    def _setup_padder(self) -> nn.Module:
        """Build the Padder that pads the input to a multiple of 2**(num_pool_layers + 1)."""
        size_multiple_of = 2 ** (self.num_pool_layers + 1)  # tensor size should be at least two at the bottleneck
        return Padder(size_multiple_of=size_multiple_of, pad_temporal=self.pool_temporal)

    def _setup_out_conv(self, in_chans: int) -> nn.Module:
        """Build the final 1x1 convolution producing `out_chans` channels."""
        return nn.Conv3d(in_chans, self.out_chans, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate')

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Average pooling by 2: spatial only when `pool_temporal` is False, spatial and temporal otherwise."""
        if self.pool_temporal:
            return func.avg_pool3d(x, kernel_size=2, stride=2, padding=0)
        else:
            return func.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)


class Unet2dPlusTime(_UnetSpatiotemporal):
    """2D+t U-Net: spatial convolutions per frame plus separate temporal convolutions (ConvBlock2dPlusTime)."""
    def _setup_conv_block(self, in_chans: int, out_chans: int) -> nn.Module:
        """Build one encoder/decoder conv block mapping `in_chans` -> `out_chans` channels."""
        return ConvBlock2dPlusTime(
            in_chans, out_chans, self.drop_prob, conv_size=self.conv_size, conv_size_temp=self.conv_size_temp,
            pad_mode_temp=self.pad_mode_temp, residual_blocks=self.residual_blocks,
            two_convs_per_block=self.two_convs_per_block,
        )

    def _setup_transpose_conv_block(self, in_chans: int, out_chans: int) -> nn.Module:
        """Build the upsampling block mapping `in_chans` -> `out_chans` channels."""
        return TransposeConvBlock2dPlusTime(
            in_chans, out_chans, upsample_temporal=self.pool_temporal, pad_mode_temp=self.pad_mode_temp,
        )


class Unet3d(_UnetSpatiotemporal):
    """3D U-Net: fully spatiotemporal convolutions (ConvBlock3d)."""
    def _setup_conv_block(self, in_chans: int, out_chans: int) -> nn.Module:
        """Build one encoder/decoder conv block mapping `in_chans` -> `out_chans` channels."""
        return ConvBlock3d(
            in_chans, out_chans, self.drop_prob, conv_size=self.conv_size, conv_size_temp=self.conv_size_temp,
            pad_mode_temp=self.pad_mode_temp, residual_blocks=self.residual_blocks,
            two_convs_per_block=self.two_convs_per_block,
        )

    def _setup_transpose_conv_block(self, in_chans: int, out_chans: int) -> nn.Module:
        """Build the upsampling block mapping `in_chans` -> `out_chans` channels."""
        return TransposeConvBlock3d(
            in_chans, out_chans, upsample_temporal=self.pool_temporal, pad_mode_temp=self.pad_mode_temp,
        )


class _ConvBlock(nn.Module, ABC):
    """Abstract conv block: `layers` (+ optional residual identity) followed by the output activation/dropout. Input/output: feature maps with `in_chans`/`out_chans` channels."""
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            drop_prob: float,
            residual_blocks: bool = True,
            two_convs_per_block: bool = True,
    ):
        super().__init__()

        self.in_channels = in_chans
        self.out_channels = out_chans
        self.drop_prob = drop_prob
        self.residual_blocks = residual_blocks
        self.two_convs_per_block = two_convs_per_block
        self.conv_args_common = {
            'out_channels': self.out_channels,
            'bias': False,
        }
        self.act_args = {
            'negative_slope': 0.2,
            'inplace': True,
        }

        self.layers: nn.Module
        self.identity: nn.Module | None
        self.out: nn.Module

    @abstractmethod
    def _setup_layers(self) -> None:
        """Fill `self.layers` with this block's convolution stack."""
        raise NotImplementedError

    @abstractmethod
    def _setup_identity(self) -> None:
        """Build the 1x1 projection of the residual connection (None when residual_blocks is False)."""
        raise NotImplementedError

    @abstractmethod
    def _setup_out_conv(self) -> None:
        """Build the output activation + dropout applied after the residual add."""
        raise NotImplementedError

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: [B, in_chans, ...] -> [B, out_chans, ...] (layers, optional residual add, output conv)."""
        output = self.layers(image)
        if self.residual_blocks:
            assert self.identity is not None, 'Identity and out must be set for residual connection.'
            identity = self.identity(image)
            output = output + identity
        output = self.out(output)
        return output


class ConvBlock2d(_ConvBlock):
    """2D conv block: InstanceNorm + LeakyReLU, one or two convs per block (`two_convs_per_block`)."""
    def __init__(self, *args, conv_size: int = 3, **kwargs):
        super().__init__(*args, **kwargs)

        if conv_size % 2 != 1 or conv_size <= 0:
            raise ValueError(f'`conv_size` must be a positive odd integer, but was {conv_size}')

        self.conv_args_spatial = {
            **self.conv_args_common,
            'kernel_size': (conv_size, conv_size),
            'padding': (conv_size // 2, conv_size // 2),
        }

        self._setup_layers()
        self._setup_identity()
        self._setup_out_conv()

    def _setup_layers(self) -> None:
        """Fill `self.layers` with this block's convolution stack."""
        layers = [
            conv.Conv2dSpatial(in_channels=self.in_channels, **self.conv_args_spatial),
            nn.InstanceNorm2d(self.out_channels),
        ]
        if self.two_convs_per_block:
            layers.extend([
                nn.LeakyReLU(**self.act_args),
                nn.Dropout2d(self.drop_prob),
                conv.Conv2dSpatial(in_channels=self.out_channels, **self.conv_args_spatial),
                nn.InstanceNorm2d(self.out_channels),
            ])

        self.layers = nn.Sequential(*layers)

    def _setup_identity(self) -> None:
        """Build the 1x1 projection of the residual connection (None when residual_blocks is False)."""
        if self.residual_blocks:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, kernel_size=(1, 1), **self.conv_args_common),
                nn.InstanceNorm2d(self.out_channels),
            )
        else:
            self.identity = None

    def _setup_out_conv(self) -> None:
        """Build the output activation + dropout applied after the residual add."""
        self.out = nn.Sequential(
            nn.LeakyReLU(**self.act_args),
            nn.Dropout2d(self.drop_prob),
        )


class _ConvBlockSpatiotemporal(_ConvBlock, ABC):
    """Conv block with separate kernel/padding configs for spatial, temporal and spatiotemporal convolutions."""
    def __init__(self, *args, conv_size: int = 3, conv_size_temp: int = 3, pad_mode_temp: str = 'circular', **kwargs):
        super().__init__(*args, **kwargs)

        if conv_size % 2 != 1 or conv_size <= 0:
            raise ValueError(f'`conv_size` must be a positive odd integer, but was {conv_size}')
        if conv_size_temp % 2 != 1 or conv_size_temp <= 0:
            raise ValueError(f'`conv_size_temp` must be a positive odd integer, but was {conv_size_temp}')

        self.conv_args_spatial = {
            **self.conv_args_common,
            'kernel_size': (conv_size, conv_size),
            'padding': (conv_size // 2, conv_size // 2),
        }
        self.conv_args_temporal = {
            **self.conv_args_common,
            'kernel_size': (conv_size_temp,),
            'padding': (conv_size_temp // 2,),
            'pad_mode_temp': pad_mode_temp,
        }
        self.conv_args_spatiotemporal = {
            **self.conv_args_common,
            'kernel_size': (conv_size_temp, conv_size, conv_size),
            'padding': (conv_size_temp // 2, conv_size // 2, conv_size // 2),
            'pad_mode_temp': pad_mode_temp,
        }

        self._setup_layers()
        self._setup_identity()
        self._setup_out_conv()

    def _setup_identity(self) -> None:
        """Build the 1x1 projection of the residual connection (None when residual_blocks is False)."""
        if self.residual_blocks:
            self.identity = nn.Sequential(
                nn.Conv3d(in_channels=self.in_channels, kernel_size=(1, 1, 1), **self.conv_args_common),
                nn.InstanceNorm3d(self.out_channels),
            )
        else:
            self.identity = None

    def _setup_out_conv(self) -> None:
        """Build the output activation + dropout applied after the residual add."""
        self.out = nn.Sequential(
            nn.LeakyReLU(**self.act_args),
            nn.Dropout3d(self.drop_prob),
        )


class ConvBlock2dPlusTime(_ConvBlockSpatiotemporal):
    """2D+t block: each half-block is a spatial conv followed by a temporal conv (with InstanceNorm)."""
    def _setup_layers(self) -> None:
        """Fill `self.layers` with this block's convolution stack."""
        conv1 = nn.Sequential(
            conv.Conv3dSpatial(in_channels=self.in_channels, **self.conv_args_spatial),
            nn.InstanceNorm3d(self.out_channels),
            nn.LeakyReLU(**self.act_args),
            conv.Conv3dTemporal(in_channels=self.out_channels, **self.conv_args_temporal),
        )
        layers = [
            conv1,
            nn.InstanceNorm3d(self.out_channels),
        ]
        if self.two_convs_per_block:
            conv2 = nn.Sequential(
                conv.Conv3dSpatial(in_channels=self.out_channels, **self.conv_args_spatial),
                nn.InstanceNorm3d(self.out_channels),
                nn.LeakyReLU(**self.act_args),
                conv.Conv3dTemporal(in_channels=self.out_channels, **self.conv_args_temporal),
            )
            layers.extend([
                nn.LeakyReLU(**self.act_args),
                nn.Dropout3d(self.drop_prob),
                conv2,
                nn.InstanceNorm3d(self.out_channels),
            ])

        self.layers = nn.Sequential(*layers)


class ConvBlock3d(_ConvBlockSpatiotemporal):
    """3D block: spatiotemporal convolutions (combined spatial + temporal kernel) with InstanceNorm."""
    def _setup_layers(self) -> None:
        """Fill `self.layers` with this block's convolution stack."""
        layers = [
            conv.Conv3dSpatiotemporal(in_channels=self.in_channels, **self.conv_args_spatiotemporal),
            nn.InstanceNorm3d(self.out_channels),
        ]
        if self.two_convs_per_block:
            layers.extend([
                nn.LeakyReLU(**self.act_args),
                nn.Dropout3d(self.drop_prob),
                conv.Conv3dSpatiotemporal(in_channels=self.out_channels, **self.conv_args_spatiotemporal),
                nn.InstanceNorm3d(self.out_channels),
            ])

        self.layers = nn.Sequential(*layers)


class _TransposeConvBlock(nn.Module, ABC):
    """Abstract upsampling block: subclasses fill `self.layers`, forward just runs them."""
    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()

        self.in_channels = in_chans
        self.out_channels = out_chans
        self.conv_args_common = {
            'out_channels': self.out_channels,
            'bias': False,
        }
        self.act_args = {
            'negative_slope': 0.2,
            'inplace': True,
        }

        self.layers: nn.Module

    @abstractmethod
    def _setup_layers(self) -> None:
        """Fill `self.layers` with this block's convolution stack."""
        raise NotImplementedError

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image -> self.layers(image): upsampled output with `out_chans` channels."""
        output = self.layers(image)
        return output


class TransposeConvBlock2d(_TransposeConvBlock):
    """2D upsampling block: 2x bilinear upsample + 3x3 conv (avoids the checkerboard artifacts of ConvTranspose2d)."""
    def __init__(self, *args):
        super().__init__(*args)

        self.conv_args_spatial = {
            **self.conv_args_common,
            'kernel_size': (3, 3),
            'padding': (1, 1),
        }

        self._setup_layers()

    def _setup_layers(self) -> None:
        """Fill `self.layers` with this block's convolution stack."""
        self.layers = nn.Sequential(
            # Upsample + Conv{2,3}d reduces checkerboard artifacts compared to ConvTranspose{2,3}d
            # https://distill.pub/2016/deconv-checkerboard/
            nn.Upsample(scale_factor=2),
            conv.Conv2dSpatial(in_channels=self.in_channels, **self.conv_args_spatial),
            nn.InstanceNorm2d(self.out_channels),
            nn.LeakyReLU(**self.act_args),
        )


class TransposeConvBlock2dPlusTime(_TransposeConvBlock):
    """2D+t upsampling block: (optional temporal) 2x upsample, spatial conv, temporal conv."""
    def __init__(self, *args, upsample_temporal: bool = True, pad_mode_temp: str = 'circular'):
        super().__init__(*args)

        self.upsample_temporal = upsample_temporal
        self.conv_args_spatial = {
            **self.conv_args_common,
            'kernel_size': (3, 3),
            'padding': (1, 1),
        }
        self.conv_args_temporal = {
            **self.conv_args_common,
            'kernel_size': (3,),
            'padding': (1,),
            'pad_mode_temp': pad_mode_temp,
        }

        self._setup_layers()

    def _setup_layers(self) -> None:
        """Fill `self.layers` with this block's convolution stack."""
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=(2 if self.upsample_temporal else 1, 2, 2)),
            conv.Conv3dSpatial(in_channels=self.in_channels, **self.conv_args_spatial),
            nn.InstanceNorm3d(self.out_channels),
            nn.LeakyReLU(**self.act_args),
            conv.Conv3dTemporal(in_channels=self.out_channels, **self.conv_args_temporal),
            nn.InstanceNorm3d(self.out_channels),
            nn.LeakyReLU(**self.act_args),
        )


class TransposeConvBlock3d(_TransposeConvBlock):
    """3D upsampling block: (optional temporal) 2x upsample followed by a 3x3x3 conv."""
    def __init__(self, *args, upsample_temporal: bool = True, pad_mode_temp: str = 'circular'):
        super().__init__(*args)

        self.upsample_temporal = upsample_temporal
        self.conv_args_spatiotemporal = {
            **self.conv_args_common,
            'kernel_size': (3, 3, 3),
            'padding': (1, 1, 1),
            'pad_mode_temp': pad_mode_temp,
        }

        self._setup_layers()

    def _setup_layers(self) -> None:
        """Fill `self.layers` with this block's convolution stack."""
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=(2 if self.upsample_temporal else 1, 2, 2)),
            conv.Conv3dSpatiotemporal(in_channels=self.in_channels, **self.conv_args_spatiotemporal),
            nn.InstanceNorm3d(self.out_channels),
            nn.LeakyReLU(**self.act_args),
        )


class Padder(nn.Module):
    """Pads the last spatial (and optionally temporal) dims to a multiple of `size_multiple_of` (circular phase, replicate readout) and can undo that padding."""
    def __init__(self, size_multiple_of: int,  pad_temporal: bool) -> None:
        super().__init__()
        self.size_multiple_of = size_multiple_of
        self.pad_temporal = pad_temporal

        self.padding_front: torch.Tensor | None = None
        self.padding_back: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, unpad: bool = False) -> torch.Tensor:
        """x -> padded x; with unpad=True the padding recorded by the last pad() call is removed instead."""
        if unpad:
            return self.unpad(x)
        else:
            return self.pad(x)

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the front/back padding of the last 2 (or 3) dims, apply it and store it for unpad()."""
        # determine amount of padding
        unpadded_size = torch.tensor(x.shape, dtype=torch.int64)
        padded_size = torch.tensor(x.shape, dtype=torch.int64)
        for dim in range(3 if self.pad_temporal else 2):
            padded_size[-dim-1] = ((x.shape[-dim-1] - 1) // self.size_multiple_of + 1) * self.size_multiple_of
        padding_front = torch.floor((padded_size - unpadded_size) / 2).to(torch.int32)
        padding_back = padded_size - unpadded_size - padding_front

        # phase padding (circular)
        x = func.pad(x, (padding_front[-1], padding_back[-1]) + (0, 0) * (x.ndim - 3), mode='circular')  # type: ignore

        # readout padding (replicate)
        x = func.pad(x, (0, 0, padding_front[-2], padding_back[-2]) + (0, 0) * (x.ndim - 4), mode='replicate')  # type: ignore

        # temporal padding (circular)
        if self.pad_temporal:
            x = func.pad(x, (0, 0, 0, 0, padding_front[-3], padding_back[-3]), mode='circular')  # type: ignore

        self.padding_front = padding_front
        self.padding_back = padding_back

        return x

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the padding stored by pad() (raises if pad() has not been applied yet)."""
        if self.padding_front is None or self.padding_back is None:
            raise ValueError('Padding has not been applied yet.')

        x = x[
            ...,
            self.padding_front[-3]:x.shape[-3] - self.padding_back[-3],
            self.padding_front[-2]:x.shape[-2] - self.padding_back[-2],
            self.padding_front[-1]:x.shape[-1] - self.padding_back[-1],
        ]
        self.padding_front = None
        self.padding_back = None
        return x


class NormUnet(nn.Module):
    """
    Wrapper for U-Net model that applies normalization to complex-valued
    input and output. This keeps the values more numerically stable
    during training. The normalization is applied separately to the real
    and imaginary parts, which are expected in the last dimension.
    """

    def __init__(self, unet: _Unet):
        """
        Args:
            unet: The U-Net model.
        """
        super().__init__()
        self.unet = unet

    def norm(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize the real and imaginary parts separately -> (x_norm, mean, std)."""
        mean = x.flatten(1, -2).mean(dim=1).view((-1,) + (1,) * (x.ndim - 2) + (2,))
        std = x.flatten(1, -2).std(dim=1).view((-1,) + (1,) * (x.ndim - 2) + (2,))
        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Inverse of norm(): x * std + mean."""
        return x * std + mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: complex [b, c, (t), h, w, 2] -> same shape (normalize -> U-Net -> unnormalize)."""
        if x.shape[-1] != 2:
            raise ValueError('Last dimension must be 2 for complex.')

        # normalize
        x, mean, std = self.norm(x)

        # apply U-Net
        x = self.unet(x)

        # unnormalize
        x = self.unnorm(x, mean, std)

        return x
