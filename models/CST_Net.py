"""CST: CRNN-refined spatiotemporal transformer for dynamic (cine) MRI reconstruction.

Reference: https://github.com/XWangBin/CST-Net

Implements CST from "CRNN-Refined Spatiotemporal Transformer for Dynamic MRI Reconstruction"
(B. Wang et al., Computers in Biology and Medicine, 2024): space-attention (InceptionNeXt)
and time-attention stages interleaved with CRNN blocks, each update followed by k-space data
consistency. Referenced by pl_modules/CSTNet_module.py (class_path: pl_modules.CSTNetModule).
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import trunc_normal_, DropPath
from timm.layers.helpers import to_2tuple
from einops import rearrange
import numbers
from torch.autograd import Variable
from helpers.mri_torch import sens_expand, sens_reduce

class DataConsistencyInKspace(nn.Module):
    """K-space data consistency: measured samples (mask == 1) come from k0, unmeasured ones from the network prediction."""
    def __init__(self):
        super(DataConsistencyInKspace, self).__init__()

    def dc(self, x, k0, mask, sens_maps):
        """x: [B, t, H, W, 2] image -> blend in k-space (keep measured k0, use the prediction elsewhere) -> [B, t, H, W, 2]."""
        x = x[:,None,None].contiguous() # [B, 1, 1, t, H, W, 2]
        k_pred = sens_expand(x, sens_maps)
        k_pred = (1-mask) * k_pred + k0
        x = sens_reduce(k_pred, sens_maps, fft_ndim=2)[:,0,0].contiguous() # [B, t, H, W, 2]
        return x

    def forward(self, x, k0, mask, sens_maps):
        """4D input [B, 2*t, H, W] (2*t == 32 is asserted) or 5D input [B, 2, t, H, W] -> the same layout after data consistency with k0/mask/sens_maps."""
        if len(x.shape) == 4:
            assert x.shape[1] == 32, "Expected input with 32 channels, got {}".format(x.shape[1])
            x = rearrange(x, 'b (two t) h w -> b t h w two', two=2)
            x = self.dc(x, k0, mask, sens_maps)
            x = rearrange(x, 'b t h w two -> b (two t) h w', two=2)
            
            return x
        
        elif len(x.shape) == 5:
            assert x.shape[1] == 2, "Expected input with 2 channels, got {}".format(x.shape[1])
            x = rearrange(x, 'b two t h w -> b t h w two', two=2)
            x = self.dc(x, k0, mask, sens_maps)
            x = rearrange(x, 'b t h w two -> b two t h w', two=2)
            
            return x
        
        else:
            raise ValueError("Expected input with 4 or 5 dimensions, got {}".format(x.shape))


# --------------------------------------------------------------------------- #
#                             Space dimension reconstruction
# ----------------------------------------------------------------------------#

class InceptionDWConv2d(nn.Module):
    """Inception depthwise convolution: splits the channels into an identity branch and three depthwise branches (square, horizontal, vertical)."""

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        """[B, C, H, W] -> [B, C, H, W]: identity branch plus square/horizontal/vertical depthwise branches."""

        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        """[B, C, H, W] -> [B, out_features, H, W] (1x1 convs keep the spatial dims)."""
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MetaNeXtBlock(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=nn.Identity,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,

    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """[B, C, H, W] -> [B, C, H, W] residual block: token mixer, norm, MLP, layer scale, drop path."""
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class MetaNeXtStage(nn.Module):
    """One MetaNeXt stage: `depth` residual MetaNeXtBlocks; the downsample path is nn.Identity in this port. Input/output: [B, C, H, W]."""
    def __init__(
            self,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=nn.Identity,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        """[B, C, H, W] -> [B, C, H, W] (downsample, then the stage blocks)."""
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class Space_Attention(nn.Module):
    r"""InceptionNeXt-style space-attention stack used by STTstage.

    Token mixer: Inception depthwise convolution (see "InceptionNeXt: When Inception Meets
    ConvNeXt", https://arxiv.org/abs/2303.16900). Input/output: [B, C, H, W] feature maps;
    this repo instantiates it with depths=(1, 1), dims=(32, 32) and in_chans=32.
    """

    def __init__(
            self,
            in_chans=32,

            depths=(1, 1),
            dims=(32, 32),
            token_mixers=InceptionDWConv2d,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU,
            mlp_ratios=(4, 4, 4, 3),
            drop_rate=0.,
            drop_path_rate=0.,
            ls_init_value=1e-6,
            **kwargs,
    ):
        super().__init__()

        num_stage = len(depths)
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage

        self.drop_rate = drop_rate

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
        # feature resolution stages, each consisting of multiple residual blocks
        for i in range(num_stage):
            out_chs = dims[i]
            stages.append(MetaNeXtStage(
                prev_chs,
                out_chs,
                ds_stride=2 if i > 0 else 1,
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                token_mixer=token_mixers[i],
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratios[i],
            ))
            prev_chs = out_chs
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs
        self.apply(self._init_weights)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Propagate the grad-checkpointing flag to every stage (not used by the current training code)."""
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        """Parameter names excluded from weight decay ({'norm'}, as in the MetaNeXt reference)."""
        return {'norm'}

    def forward(self, x):
        """[B, C, H, W] -> [B, C, H, W] (channel count after the stages is dims[-1])."""
        x = self.stages(x)
        return x

    def _init_weights(self, m):
        """Trunc-normal (std=.02) weights and zero biases for Conv2d/Linear modules."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# --------------------------------------------------------------------------- #
#                             Time dimension reconstruction
# ----------------------------------------------------------------------------#

class Residual(nn.Module):
    """Residual wrapper around a module: forward returns x + module(x)."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        """x -> x + module(x), same shape."""
        return x + self.module(x)

ACT_FN = {'gelu': nn.GELU(),'relu' : nn.ReLU(),'lrelu' : nn.LeakyReLU()}

def to_3d(x):
    """[b, c, h, w] -> [b, h*w, c] (fold the spatial dims into one sequence dim)."""
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    """[b, h*w, c] -> [b, c, h, w] (inverse of to_3d)."""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    """Layer normalization over the last dim without mean subtraction or bias (scale only)."""
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """Normalize over the last dimension: x / sqrt(var + 1e-5) * weight (scale only, no shift)."""
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """Layer normalization over the last dim with learned scale and shift."""
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """Normalize over the last dimension: (x - mean) / sqrt(var + 1e-5) * weight + bias."""
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    """LayerNorm over channels that flattens/restores the spatial dims via to_3d/to_4d. Input/output: [b, c, h, w]."""
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        """[b, c, h, w] -> [b, c, h, w]; normalization is applied per pixel over channels."""
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SpectralBranch(nn.Module):
    """Multi-head self-attention branch with 1x1 + depthwise conv QKV and a learned temperature. Input/output: [B, C, H, W]."""
    def __init__(self,

                 dim,
                 num_heads,
                 bias=False,
                 LayerNorm_type="WithBias"
                 ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, spatial_interaction=None):
        """x: [B, C, H, W] -> [B, C, H, W] multi-head self-attention over spatial tokens; optional spatial_interaction (never passed here) gates q/k/v."""
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        if spatial_interaction is not None:
            q = q * spatial_interaction
            k = k * spatial_interaction
            v = v * spatial_interaction

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)  # (b, c, h, w) -> (b, head, c_, h * w)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h,
                        w=w)  # (b, head, c_, h*w) -> (b, c, h, w)

        out = self.project_out(out)  # (b, c, h, w)
        return out

class Gated_Dconv_FeedForward(nn.Module):
    """Gated depthwise-convolution FFN (channels expanded by ffn_expansion_factor, gated in two halves). Input/output: [B, C, H, W]."""
    def __init__(self,
                 dim,
                 ffn_expansion_factor = 2.66,
                 bias = False,
                 LayerNorm_type = "WithBias",
                 act_fn_name = "gelu"
    ):
        super(Gated_Dconv_FeedForward, self).__init__()
        self.norm = LayerNorm(dim, LayerNorm_type = LayerNorm_type)

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.act_fn = ACT_FN[act_fn_name]

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """[B, C, H, W] -> [B, C, H, W] (norm -> 1x1 expand -> depthwise conv -> gate -> 1x1 project)."""
        x = self.norm(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x

def FFN_FN(ffn_name,dim,ffn_expansion_factor=2.66,bias=False,LayerNorm_type="WithBias",act_fn_name = "gelu"):
    """Factory returning the FFN block named `ffn_name`; only "Gated_Dconv_FeedForward" is implemented (other names return None)."""
    if ffn_name == "Gated_Dconv_FeedForward":
        return Gated_Dconv_FeedForward(
                dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                act_fn_name = act_fn_name
            )

class MixS2_Block(nn.Module):
    """MixS2 block: SpectralBranch (attention) plus a gated depthwise FFN, both with residual connections. Input/output: [B, C, H, W]."""
    def __init__(self,
                 dim,
                 num_heads,
                 ):
        super().__init__()


        self.spectral_branch = SpectralBranch(
            dim,
            num_heads=num_heads,
            bias=False,
            LayerNorm_type="BiasFree"
        )


        self.ffn = Residual(
            FFN_FN(
                dim=dim,
                ffn_name='Gated_Dconv_FeedForward',
                ffn_expansion_factor=2.66,
                bias=False,
                LayerNorm_type="BiasFree"
            )
        )

    def forward(self, x):
        """[B, C, H, W] -> [B, C, H, W]: residual spectral attention followed by a residual gated FFN."""
        spectral_fea = self.spectral_branch(x)
        spectral_fea = x + spectral_fea
        out = self.ffn(spectral_fea)
        return out

class Time_Attention(nn.Module):
    """Stack of `num` MixS2 blocks used as the temporal branch of STTstage."""
    def __init__(self,
                 num=2,
                 dim=32
                 ):
        super().__init__()

        blocks = []
        for i in range(num):
            blocks.append(MixS2_Block(dim=dim,num_heads=1))
        self.stages = nn.Sequential(*blocks)

    def forward(self, x):
        """Applies every stage block to the same input x and returns the last block's output (the blocks are not chained). [B, C, H, W] -> [B, C, H, W]."""
        for stage in self.stages:
            out = stage(x)
        return out

# --------------------------------------------------------------------------- #
#                             Space-Time Transformer reconstruction
# ----------------------------------------------------------------------------#

class DownSample(nn.Module):
    """Bilinear resampling by scale_factor followed by a 3x3 conv adjusting the channel count (unused by CST; kept from the reference code)."""
    def __init__(self, in_channels,scale_factor, bias=False):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels*(int(1/scale_factor)), 3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x):
        """x -> self.down(x): resample by scale_factor, then the 3x3 conv."""
        x = self.down(x)
        return x

class UpSample(nn.Module):
    """Bilinear upsampling by scale_factor followed by a 3x3 conv that reduces channels by //scale_factor (unused by CST; kept from the reference code)."""
    def __init__(self, in_channels,scale_factor, bias=False):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            # blinear interpolate may make results not deterministic
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels//scale_factor, 3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x):
        """x -> self.up(x): resample by scale_factor, then the 3x3 conv."""
        x = self.up(x)
        return x

class STTstage(nn.Module):
    """Space-time transformer stage: Space_Attention and Time_Attention run in parallel (both under gradient checkpointing) and are fused by a 1x1 conv. Input/output: [B, dim, H, W]."""
    def __init__(self,dim=32,dims=(128,128)):
        super().__init__()
        self.spaceattn = Space_Attention(dims=dims)
        self.timeattn = Time_Attention(dim=dim)
        self.end = nn.Conv2d(dim*2,dim,1,bias=False)

    def forward(self, x):
        """[B, dim, H, W] -> [B, dim, H, W]: parallel space/time attention, concatenated and fused by `end`."""

        spa = checkpoint.checkpoint(self.spaceattn, x, use_reentrant=False)
        tim = checkpoint.checkpoint(self.timeattn, x, use_reentrant=False)
        out = self.end(torch.cat([spa,tim],dim=1))
        return out



# --------------------------------------------------------------------------- #
#                             CRNN  reconstruction
# ----------------------------------------------------------------------------#

class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations

    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    iteration: True or False, to use iteration recurrence or not; if iteration=False: hidden_iteration=None

    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size, dilation, iteration=False):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.iteration = iteration
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=dilation, dilation=dilation)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=dilation, dilation=dilation)
        # add iteration hidden connection
        if self.iteration:
            self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, input, hidden, hidden_iteration=None):
        """input [B, C_in, H, W] + hidden [B, C_hid, H, W] (+ optional hidden_iteration) -> next hidden state [B, C_hid, H, W]."""
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        if hidden_iteration is not None:
            ih_to_ih = self.ih2ih(hidden_iteration)
            hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)
        else:
            hidden = self.relu(in_to_hid + hid_to_hid)

        return hidden

class CRNN_MRI(nn.Module):
    """
    CRNN-MRI block in image domain
    RNN evolves over temporal dimension only
    """
    def __init__(self, n_ch, nf=64, ks=3, dilation=2):
        super(CRNN_MRI, self).__init__()
        self.nf = nf
        self.ks = ks

        self.bcrnn_1 = BCRNNlayer(n_ch, nf, ks, dilation=1)
        self.bcrnn_2 = BCRNNlayer(nf, nf, ks, dilation)
        self.bcrnn_3 = BCRNNlayer(nf, nf, ks, dilation)
        self.bcrnn_4 = BCRNNlayer(nf, nf, ks, dilation)

        self.conv4_x = nn.Conv2d(nf, 2, ks, padding=ks//2)

    def forward(self, x, test=False):
        """x: [n_batch, n_ch, width, length, n_seq] -> [n_batch, 2, width, length, n_seq].

        The last axis (sequence/time) is moved to the front, processed by the
        bidirectional BCRNN layers, mapped to 2 channels by conv4_x, and moved back.
        """

        n_batch, n_ch, width, length, n_seq = x.size()

        x = x.permute(4, 0, 1, 2, 3)

        out = self.bcrnn_1(x, None, test)
        out = self.bcrnn_2(out, None, test)
        out = self.bcrnn_3(out, None, test)
        out = self.bcrnn_4(out, None, test)
        out = out.view(-1, self.nf, width, length)
        out = self.conv4_x(out)

        out = out.view(-1, n_batch, 2, width, length)
        out = out.permute(1, 2, 3, 4, 0)

        return out

class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer

    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode
               iteration: True if use iteration recurrence and input_iteration is not None; False if input_iteration=None

    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size, dilation, iteration=False):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.iteration = iteration
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size, dilation, iteration=self.iteration)

    def forward(self, input, input_iteration=None, test=False):
        """input [n_seq, n_batch, C, nx, ny] -> [n_seq, n_batch, hidden, nx, ny] (bidirectional recurrence over n_seq)."""
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h), requires_grad=False).to(torch.device('cuda'))
        else:
            hid_init = Variable(torch.zeros(size_h), requires_grad=False).to(torch.device('cuda'))

        output_f = []
        output_b = []
        if input_iteration is not None:
            # forward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[i], hidden, input_iteration[i])
                output_f.append(hidden)
            # backward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[nt - i - 1], hidden, input_iteration[nt - i -1])
                output_b.append(hidden)
        else:
            # forward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[i], hidden)
                output_f.append(hidden)
            # backward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[nt - i - 1], hidden)
                output_b.append(hidden)

        output_f = torch.cat(output_f)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b


        output = output.view(nt, nb, self.hidden_size, nx, ny)

        return output

# --------------------------------------------------------------------------- #
#                  CRNN Refined  Space-Time Transformer reconstruction
# ----------------------------------------------------------------------------#
class CST(nn.Module):
    """CST: CRNN-refined space-time transformer for dynamic MRI.

    Three stages of space/time attention (STTstage) interleaved with CRNN blocks,
    each update followed by k-space data consistency; called by CSTNetModule as
    CST(x, k0, mask, sens_maps) -> [B, t, H, W, 2]."""
    def __init__(self,
                 num_stages=[2,2,2],
                 inchans = 32
                 ):
        super().__init__()
        self.dc = DataConsistencyInKspace()
        self.embeding = nn.Conv2d(inchans,32,3,1,1,bias=False)

        self.stages1 = nn.ModuleList([])
        for i in range(num_stages[0]):
            self.stages1.append(
                STTstage(dim=32, dims=(32, 32)),
            )

        self.stages2 = nn.ModuleList([])
        for i in range(num_stages[1]):
            self.stages2.append(
                STTstage(dim=32, dims=(32, 32)),
            )

        self.stages3 = nn.ModuleList([])
        for i in range(num_stages[2]):
            self.stages3.append(
                STTstage(dim=32, dims=(32, 32)),
            )

        self.end = nn.Conv3d(6,2,3,1,1,bias=False)


        xt_conv_blocks1 = []
        for i in range(3):
            xt_conv_blocks1.append(CRNN_MRI(2, 64, dilation=3))
        self.xt_conv_blocks1 = nn.ModuleList(xt_conv_blocks1)
        xt_conv_blocks2 = []
        for i in range(3):
            xt_conv_blocks2.append(CRNN_MRI(2, 64, dilation=3))
        self.xt_conv_blocks2 = nn.ModuleList(xt_conv_blocks2)
        xt_conv_blocks3 = []
        for i in range(2):
            xt_conv_blocks3.append(CRNN_MRI(2, 64, dilation=3))
        self.xt_conv_blocks3 = nn.ModuleList(xt_conv_blocks3)


    def forward(self, x, k0, mask, sens_maps):
        """x: zero-filled image [B, t, H, W, 2]; k0: measured k-space, mask, sens_maps.
        Returns the reconstruction as real/imag [B, t, H, W, 2]."""
        x0 = rearrange(x, 'b t h w two -> b (two t) h w')
        x1 = self.embeding(x0)
        for (stage1) in self.stages1:
            x11 = stage1(x1)+x1
            x1 = self.dc(x11,k0,mask,sens_maps)
        x1 = rearrange(x1, 'b (two t) h w -> b two t h w', two=2)
        for i in range(3):
            out = self.xt_conv_blocks1[i](x1)
            x1 = x1 + out # residual add; layout stays [B, 2, t, h, w]
            x1 = self.dc(x1,k0,mask,sens_maps)

        x2 = rearrange(x1, 'b two t h w -> b (two t) h w')
        for (stage2) in self.stages2:
            x22 = stage2(x2)+x2
            x2 = self.dc(x22,k0,mask,sens_maps)
        x2 = rearrange(x2, 'b (two t) h w -> b two t h w', two=2)
        for i in range(3):
            out = self.xt_conv_blocks2[i](x2)
            x2 = x2 + out
            x2 = self.dc(x2, k0, mask,sens_maps)

        x3 = rearrange(x2, 'b two t h w -> b (two t) h w')
        for (stage3) in self.stages3:
            x33 = stage3(x3)+x3
            x3 = self.dc(x33,k0,mask,sens_maps)
        x3 = rearrange(x3, 'b (two t) h w -> b two t h w', two=2)
        for i in range(2):
            out = self.xt_conv_blocks3[i](x3)
            x3 = x3 + out
            x3 = self.dc(x3, k0, mask,sens_maps)

        out = self.end(torch.cat([x1,x2,x3],dim=1))
        out = self.dc(out,k0,mask,sens_maps).permute(0,2,3,4,1) # [B, t, H, W, 2]
        return out # [B, t, H, W, 2]
