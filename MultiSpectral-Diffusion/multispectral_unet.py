from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, ll, lh, hl, emb):
        """
        Apply the module to subbands given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, ll, lh, hl, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                ll, lh, hl = layer(ll, lh, hl, emb)
            else:
                ll, lh, hl = layer(ll, lh, hl)
        return ll, lh, hl


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.ll_in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.lh_in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.hl_in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.ll_out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        self.lh_out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        self.hl_out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.ll_skip_connection = nn.Identity()
            self.lh_skip_connection = nn.Identity()
            self.hl_skip_connection = nn.Identity()

        elif use_conv:
            self.ll_skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
            self.lh_skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
            self.hl_skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)

        else:
            self.ll_skip_connection = conv_nd(dims, channels, self.out_channels, 1)
            self.lh_skip_connection = conv_nd(dims, channels, self.out_channels, 1)
            self.hl_skip_connection = conv_nd(dims, channels, self.out_channels, 1)


    def forward(self, ll, lh,hl, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (ll,lh,hl, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, ll, lh,hl, emb):
        if self.updown:
            ll_in_rest, ll_in_conv = self.ll_in_layers[:-1], self.ll_in_layers[-1]
            ll_h = ll_in_rest(ll)
            ll_h = self.h_upd(ll_h)
            ll = self.x_upd(ll)
            ll_h = ll_in_conv(ll_h)

            lh_in_rest, lh_in_conv = self.lh_in_layers[:-1], self.lh_in_layers[-1]
            lh_h = lh_in_rest(lh)
            lh_h = self.h_upd(lh_h)
            lh = self.x_upd(lh)
            lh_h = lh_in_conv(lh_h)

            hl_in_rest, hl_in_conv = self.hl_in_layers[:-1], self.hl_in_layers[-1]
            hl_h = hl_in_rest(hl)
            hl_h = self.h_upd(hl_h)
            hl = self.x_upd(hl)
            hl_h = hl_in_conv(hl_h)

        else:
            ll_h = self.ll_in_layers(ll)
            lh_h = self.lh_in_layers(lh)
            hl_h = self.hl_in_layers(hl)

        emb_out = self.emb_layers(emb).type(ll_h.dtype)
        while len(emb_out.shape) < len(ll_h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            ll_out_norm, ll_out_rest = self.ll_out_layers[0], self.ll_out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            ll_h = ll_out_norm(ll_h) * (1 + scale) + shift
            ll_h = ll_out_rest(ll_h)

            lh_out_norm, lh_out_rest = self.lh_out_layers[0], self.lh_out_layers[1:]
            lh_h = lh_out_norm(lh_h) * (1 + scale) + shift
            lh_h = lh_out_rest(lh_h)

            hl_out_norm, hl_out_rest = self.hl_out_layers[0], self.hl_out_layers[1:]
            hl_h = hl_out_norm(hl_h) * (1 + scale) + shift
            hl_h = hl_out_rest(hl_h)
        else:
            ll_h = ll_h + emb_out
            ll_h = self.ll_out_layers(ll_h)

            lh_h = lh_h + emb_out
            lh_h = self.lh_out_layers(lh_h)

            hl_h = hl_h + emb_out
            hl_h = self.hl_out_layers(hl_h)

        return self.ll_skip_connection(ll) + ll_h, self.lh_skip_connection(lh) + lh_h, self.hl_skip_connection(hl) + hl_h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=True,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint

        self.ll_norm = normalization(channels)
        self.lh_norm = normalization(channels)
        self.hl_norm = normalization(channels)

        
        self.ll_qkv = conv_nd(1, channels, channels * 3, 1)
        self.lh_qkv = conv_nd(1, channels, channels * 3, 1)
        self.hl_qkv = conv_nd(1, channels, channels * 3, 1)

        
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.ll_proj_out = zero_module(conv_nd(1, channels, channels, 1))
        self.lh_proj_out = zero_module(conv_nd(1, channels, channels, 1))
        self.hl_proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, ll, lh, hl):
        return checkpoint(self._forward, (ll,lh,hl), self.parameters(), True)

    def _forward(self, ll, lh, hl):
        b, c, *spatial = ll.shape

        ll = ll.reshape(b, c, -1)
        lh = lh.reshape(b, c, -1)
        hl = hl.reshape(b, c, -1)
        
        ll_qkv = self.ll_qkv(self.ll_norm(ll))
        lh_qkv = self.lh_qkv(self.lh_norm(lh))
        hl_qkv = self.hl_qkv(self.hl_norm(hl))

        ll_h, lh_h, hl_h = self.attention(ll_qkv, lh_qkv, hl_qkv)
        ll_h = self.ll_proj_out(ll_h)
        lh_h = self.lh_proj_out(lh_h)
        hl_h = self.hl_proj_out(hl_h)

        return (ll + ll_h).reshape(b, c, *spatial), (lh + lh_h).reshape(b, c, *spatial), (hl + hl_h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c * 3
    model.total_ops += th.DoubleTensor([matmul_ops])

def count_flops_cross_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors. 
    matmul_ops =  b * (num_spatial * 3 * num_spatial) * c * 3  # for attention weights, 3 stands for each subband
    matmul_ops += b * ((3*num_spatial) ** 2) * c * 3           # for aggregating value, 3 stands for each subband
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, ll_qkv, lh_qkv, hl_qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = ll_qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        ll_q, ll_k, ll_v = ll_qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        lh_q, lh_k, lh_v = lh_qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        hl_q, hl_k, hl_v = hl_qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

        scale = 1 / math.sqrt(math.sqrt(ch))
        ll_weight = th.einsum(
            "bct,bcs->bts", ll_q * scale, ll_k * scale
        )  # More stable with f16 than dividing afterwards
        ll_weight = th.softmax(ll_weight.float(), dim=-1).type(ll_weight.dtype)
        ll_a = th.einsum("bts,bcs->bct", ll_weight, ll_v)

        lh_weight = th.einsum(
            "bct,bcs->bts", lh_q * scale, lh_k * scale
        )  # More stable with f16 than dividing afterwards
        lh_weight = th.softmax(lh_weight.float(), dim=-1).type(lh_weight.dtype)
        lh_a = th.einsum("bts,bcs->bct", lh_weight, lh_v)
        
        hl_weight = th.einsum(
            "bct,bcs->bts", hl_q * scale, hl_k * scale
        )  # More stable with f16 than dividing afterwards
        hl_weight = th.softmax(hl_weight.float(), dim=-1).type(hl_weight.dtype)
        hl_a = th.einsum("bts,bcs->bct", hl_weight, hl_v)
        return ll_a.reshape(bs, -1, length), lh_a.reshape(bs, -1, length), hl_a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, ll_qkv, lh_qkv, hl_qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = ll_qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        ll_q, ll_k, ll_v = ll_qkv.chunk(3, dim=1)
        lh_q, lh_k, lh_v = lh_qkv.chunk(3, dim=1)
        hl_q, hl_k, hl_v = hl_qkv.chunk(3, dim=1)

        scale = 1 / math.sqrt(math.sqrt(ch))
        ll_weight = th.einsum(
            "bct,bcs->bts",
            (ll_q * scale).view(bs * self.n_heads, ch, length),
            (ll_k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        ll_weight = th.softmax(ll_weight.float(), dim=-1).type(ll_weight.dtype)
        ll_a = th.einsum("bts,bcs->bct", ll_weight, ll_v.reshape(bs * self.n_heads, ch, length))

        lh_weight = th.einsum(
            "bct,bcs->bts",
            (lh_q * scale).view(bs * self.n_heads, ch, length),
            (lh_k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        lh_weight = th.softmax(lh_weight.float(), dim=-1).type(lh_weight.dtype)
        lh_a = th.einsum("bts,bcs->bct", lh_weight, lh_v.reshape(bs * self.n_heads, ch, length))

        hl_weight = th.einsum(
            "bct,bcs->bts",
            (hl_q * scale).view(bs * self.n_heads, ch, length),
            (hl_k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        hl_weight = th.softmax(hl_weight.float(), dim=-1).type(hl_weight.dtype)
        hl_a = th.einsum("bts,bcs->bct", hl_weight, hl_v.reshape(bs * self.n_heads, ch, length))
        return ll_a.reshape(bs, -1, length), lh_a.reshape(bs, -1, length), hl_a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class CrossAttention(nn.Module):
    """ qkv ordering is based on qkvattention not qkvattention lagacy! 
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self,qkv, l):

        bs, width, _ = qkv.shape # [b, 3c, 3L], L is the length of each subband

        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1) # q,k,v [bs, c, 3L]
        scale = 1 / math.sqrt(math.sqrt(ch))

        ll_q = q[:, : , :l]      # [bs, c, L]
        lh_q = q[:, : , l:2*l]   # [bs, c, L]
        hl_q = q[:, : , 2*l:]    # [bs, c, L]

        ll_attn = th.einsum("bct,bcs->bts", (ll_q * scale).view(bs * self.n_heads, ch, -1),
                                                (k * scale).view(bs * self.n_heads, ch, -1)) # [bs, L, 3L]
        
        ll_to_all_weights = th.softmax(ll_attn.float(), dim = -1).type(ll_attn.dtype)
        # (* 1) for removing .view error. 
        agg_ll = th.einsum("bts,bcs->bct", ll_to_all_weights, v.reshape(bs * self.n_heads, ch, -1)) # [b, c, L]

        lh_attn = th.einsum("bct,bcs->bts", (lh_q * scale).view(bs * self.n_heads, ch, -1),
                                                (k * scale).view(bs * self.n_heads, ch, -1)) # [bs, L, 3L]
        
        lh_to_all_weights = th.softmax(lh_attn.float(), dim = -1).type(lh_attn.dtype)
        agg_lh = th.einsum("bts,bcs->bct", lh_to_all_weights, v.reshape(bs * self.n_heads, ch, -1)) # [b, c, L]

        hl_attn = th.einsum("bct,bcs->bts", (hl_q * scale).view(bs * self.n_heads, ch, -1),
                                                (k * scale).view(bs * self.n_heads, ch, -1)) # [bs, L, 3L]
        
        hl_to_all_weights = th.softmax(hl_attn.float(), dim = -1).type(hl_attn.dtype)
        agg_hl = th.einsum("bts,bcs->bct", hl_to_all_weights, v.reshape(bs * self.n_heads, ch, -1)) # [b, c, L]

        return agg_ll.reshape(bs, -1, l), agg_lh.reshape(bs, -1, l), agg_hl.reshape(bs, -1, l) # each [b,c,L] 
    
    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_cross_attn(model, _x, y)

class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (self.channels % num_head_channels == 0), f"q,k,v channels {channels} is not divisable by num_head_channels {num_head_channels}"
            self.num_heads = self.channels // num_head_channels
        self.use_checkpoint = use_checkpoint

        self.ll_norm = normalization(channels)
        self.lh_norm = normalization(channels)
        self.hl_norm = normalization(channels)

        self.ll_qkv = conv_nd(1, channels, channels * 3, 1)
        self.lh_qkv = conv_nd(1, channels, channels * 3, 1)
        self.hl_qkv = conv_nd(1, channels, channels * 3, 1)

        self.cross_attention = CrossAttention(self.num_heads)

        self.ll_proj_out = zero_module(conv_nd(1, channels, channels, 1))
        self.lh_proj_out = zero_module(conv_nd(1, channels, channels, 1))
        self.hl_proj_out = zero_module(conv_nd(1, channels, channels, 1))


    def forward(self, ll, lh, hl):
        return checkpoint(self._forward, (ll,lh,hl), self.parameters(), True)

    def _forward(self, ll, lh, hl):
        assert ll.shape == lh.shape == hl.shape
        b, c, h,w = ll.shape 

        ll_token = ll.reshape(b,c,-1)
        lh_token = lh.reshape(b,c,-1)
        hl_token = hl.reshape(b,c,-1)


        ll_qkv = self.ll_qkv(self.ll_norm(ll_token))
        lh_qkv = self.lh_qkv(self.lh_norm(lh_token))
        hl_qkv = self.hl_qkv(self.hl_norm(hl_token))

        qkv = th.concat([ll_qkv, lh_qkv, hl_qkv], dim=2)

        ll_h, lh_h, hl_h = self.cross_attention(qkv, l=h*w)

        ll_h = self.ll_proj_out(ll_h)
        ll_out = (ll_token + ll_h).reshape(b, c, h, w)

        lh_h = self.lh_proj_out(lh_h)
        lh_out = (lh_token + lh_h).reshape(b, c, h, w)

        hl_h = self.hl_proj_out(hl_h)
        hl_out = (hl_token + hl_h).reshape(b, c, h, w)

        return ll_out, lh_out, hl_out



class InitialBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dims=2,
        kernel_size = 3 
    ):
        super().__init__()
        self.ll_conv = conv_nd(dims, in_channels, out_channels, kernel_size, padding=1)
        self.lh_conv = conv_nd(dims, in_channels, out_channels, kernel_size, padding=1)
        self.hl_conv = conv_nd(dims, in_channels, out_channels, kernel_size, padding=1)

    def forward(self, ll, lh, hl): 
        return self.ll_conv(ll), self.lh_conv(lh), self.hl_conv(hl)

class MultiSpectralUNet(nn.Module):
    """
    The full MultiSpectralUNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        cross_attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=True,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.cross_attention_resolutions = cross_attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        
        
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(InitialBlock(in_channels, ch, dims, 3))])

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if ds in cross_attention_resolutions:
                    layers.append(
                        CrossAttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if ds in cross_attention_resolutions:
                    layers.append(
                        CrossAttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.ll_out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        self.lh_out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        self.hl_out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)


    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, ll, lh, hl, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        ll_hs = []
        lh_hs = []
        hl_hs = []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (ll.shape[0],)
            emb = emb + self.label_emb(y)

        ll_h = ll.type(self.dtype)
        lh_h = lh.type(self.dtype)
        hl_h = hl.type(self.dtype)

        for module in self.input_blocks:
            ll_h, lh_h, hl_h = module(ll_h, lh_h, hl_h, emb)
            ll_hs.append(ll_h)
            lh_hs.append(lh_h)
            hl_hs.append(hl_h)

        ll_h, lh_h, hl_h = self.middle_block(ll_h, lh_h, hl_h, emb)
        for module in self.output_blocks:
            ll_h = th.cat([ll_h, ll_hs.pop()], dim=1)
            lh_h = th.cat([lh_h, lh_hs.pop()], dim=1)
            hl_h = th.cat([hl_h, hl_hs.pop()], dim=1)

            ll_h, lh_h, hl_h = module(ll_h, lh_h, hl_h, emb)
        ll_h = ll_h.type(ll.dtype)
        lh_h = lh_h.type(lh.dtype)
        hl_h = hl_h.type(hl.dtype)

        return self.ll_out(ll_h), self.lh_out(lh_h), self.hl_out(hl_h)





if __name__ == "__main__":

    import torch
    from thop import profile
    from thop import clever_format
    
    bs = 1
    resblock_updown = True 
    in_channels= 3
    out_channels =3

    # 64
    model_channels = 128
    img_size = 64
    num_res_block = 3
    attention_resolutions = [2,4] # means at resolution 16,8
    cross_attention_resolutions = [2,4] # means at resolution 8, 16
    channel_mult = (1,2,2,2)
    dropout = 0.1
    num_heads = 4
    

    # 128
    # model_channels = 128
    # img_size = 128
    # num_res_block = 2
    # self_attention_resolutions = [4]
    # cross_attention_resolutions = [4,8] # 102, 108
    # channel_mult = (1,2,3,4)
    # channel_mult = (1,2,2,4,4)

    num_heads = 4
    num_head_channels =64
    dropout = 0.0


    # 256
    model_channels = 128
    img_size = 256
    num_res_block = 2
    self_attention_resolutions = [4,8]
    cross_attention_resolutions = [4,8]
    channel_mult = (1,1,2,2,4,4)
    num_heads = 4
    dropout = 0.0

    # SFUnet on LSUN -> 291(M) params, 669(G) FLOPs

    ll =torch.randn(bs, 3, img_size//2, img_size//2) # low pass 
    lh =torch.randn(bs, 3, img_size//2, img_size//2) # high pass
    hl =torch.randn(bs, 3, img_size//2, img_size//2) # high pass

    
    t = torch.randint(0, 1000, (bs,))

    model = MultiSpectralUNet(image_size = img_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    model_channels=model_channels,
                    num_res_blocks = num_res_block,
                    cross_attention_resolutions=cross_attention_resolutions,
                    attention_resolutions=attention_resolutions,
                    resblock_updown= resblock_updown,
                    num_head_channels=num_head_channels,
                    num_heads = num_heads,
                    channel_mult=channel_mult,
                    dropout=dropout,
                    use_checkpoint=True
                    )
    

    # ll, lh, hl = model(ll, lh, hl, t)
    # macs, params = profile(model, inputs=(ll, lh, hl, t), custom_ops={QKVAttention: QKVAttention.count_flops, LinearQKVAttention: LinearQKVAttention.count_flops})
    # macs, params = profile(model, inputs=(ll, lh, hl, t), custom_ops={QKVAttention: QKVAttention.count_flops})
    macs, params = profile(model, inputs=(ll, lh, hl, t,), custom_ops={QKVAttention: QKVAttention.count_flops, CrossAttention: CrossAttention.count_flops})



    # macs, params = profile(model, inputs=(ll, lh, hl, t))

    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(sum([p.numel() for p in model.parameters()]))