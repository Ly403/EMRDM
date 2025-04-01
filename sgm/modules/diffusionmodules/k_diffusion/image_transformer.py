"""k-diffusion transformer diffusion models, version 2."""
# code modified from Hourglass diffusion transformer https://.com/crogithubwsonkb/k-diffusion/
from dataclasses import dataclass
from functools import lru_cache, reduce
import math
from typing import Union

from einops import rearrange
import torch
from torch import nn

from torch.nn import functional as F
import torch._dynamo
from torchvision.transforms import v2
from sgm.util import tools_scale, tools_scale2

from sgm.modules.diffusionmodules.k_diffusion import flags, flops, layers
from sgm.modules.diffusionmodules.k_diffusion.axial_rope import make_axial_pos
from sgm.modules.diffusionmodules.k_diffusion.layers import PositionalEncoder
try:
    import natten
except ImportError:
    natten = None

try:
    import flash_attn
except ImportError:
    flash_attn = None

# Helpers

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def checkpoint(function, *args, **kwargs):
    if flags.get_checkpointing():
        kwargs.setdefault("use_reentrant", True)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)


def downscale_pos(pos):
    pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=2, nw=2)
    return torch.mean(pos, dim=-2)


# Param tags

def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param


def tag_module(module, tag):
    for param in module.parameters():
        tag_param(param, tag)
    return module


def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module


def filter_params(function, module):
    for param in module.parameters():
        tags = getattr(param, "_tags", set())
        if function(tags):
            yield param


# Kernels

@flags.compile_wrap
def linear_geglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)


@flags.compile_wrap
def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


@flags.compile_wrap
def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype)**2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype)**2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


@flags.compile_wrap
def scale_for_cosine_sim_qkv(qkv, scale, eps):
    q, k, v = qkv.unbind(2)
    q, k = scale_for_cosine_sim(q, k, scale[:, None], eps)
    return torch.stack((q, k, v), dim=2)


# Layers
class Linear(nn.Linear):
    def forward(self, x):
        # flops.op(flops.op_linear, x.shape, self.weight.shape)
        return super().forward(x)


class LinearGEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        # flops.op(flops.op_linear, x.shape, self.weight.shape)
        return linear_geglu(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = apply_wd(zero_init(Linear(cond_features, features, bias=False)))
        tag_module(self.linear, "mapping")

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, None, :] + 1, self.eps)


# Rotary position embeddings

@flags.compile_wrap
def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)


@flags.compile_wrap
def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)


class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // 4, n_heads).T.contiguous())

    def extra_repr(self):
        return f"dim={self.freqs.shape[1] * 4}, n_heads={self.freqs.shape[0]}"

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs.to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs.to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)


# Shifted window attention

def window(window_size, x):
    *b, h, w, c = x.shape
    x = torch.reshape(
        x,
        (*b, h // window_size, window_size, w // window_size, window_size, c),
    )
    x = torch.permute(
        x,
        (*range(len(b)), -5, -3, -4, -2, -1),
    )
    return x


def unwindow(x):
    *b, h, w, wh, ww, c = x.shape
    x = torch.permute(x, (*range(len(b)), -5, -3, -4, -2, -1))
    x = torch.reshape(x, (*b, h * wh, w * ww, c))
    return x


def shifted_window(window_size, window_shift, x):
    x = torch.roll(x, shifts=(window_shift, window_shift), dims=(-2, -3))
    windows = window(window_size, x)
    return windows


def shifted_unwindow(window_shift, x):
    x = unwindow(x)
    x = torch.roll(x, shifts=(-window_shift, -window_shift), dims=(-2, -3))
    return x


@lru_cache
def make_shifted_window_masks(n_h_w, n_w_w, w_h, w_w, shift, device=None):
    ph_coords = torch.arange(n_h_w, device=device)
    pw_coords = torch.arange(n_w_w, device=device)
    h_coords = torch.arange(w_h, device=device)
    w_coords = torch.arange(w_w, device=device)
    patch_h, patch_w, q_h, q_w, k_h, k_w = torch.meshgrid(
        ph_coords,
        pw_coords,
        h_coords,
        w_coords,
        h_coords,
        w_coords,
        indexing="ij",
    )
    is_top_patch = patch_h == 0
    is_left_patch = patch_w == 0
    q_above_shift = q_h < shift
    k_above_shift = k_h < shift
    q_left_of_shift = q_w < shift
    k_left_of_shift = k_w < shift
    m_corner = (
        is_left_patch
        & is_top_patch
        & (q_left_of_shift == k_left_of_shift)
        & (q_above_shift == k_above_shift)
    )
    m_left = is_left_patch & ~is_top_patch & (q_left_of_shift == k_left_of_shift)
    m_top = ~is_left_patch & is_top_patch & (q_above_shift == k_above_shift)
    m_rest = ~is_left_patch & ~is_top_patch
    m = m_corner | m_left | m_top | m_rest
    return m


def apply_window_attention(window_size, window_shift, q, k, v, scale=None):
    # prep windows and masks
    q_windows = shifted_window(window_size, window_shift, q)
    k_windows = shifted_window(window_size, window_shift, k)
    v_windows = shifted_window(window_size, window_shift, v)
    b, heads, h, w, wh, ww, d_head = q_windows.shape
    mask = make_shifted_window_masks(h, w, wh, ww, window_shift, device=q.device)
    q_seqs = torch.reshape(q_windows, (b, heads, h, w, wh * ww, d_head))
    k_seqs = torch.reshape(k_windows, (b, heads, h, w, wh * ww, d_head))
    v_seqs = torch.reshape(v_windows, (b, heads, h, w, wh * ww, d_head))
    mask = torch.reshape(mask, (h, w, wh * ww, wh * ww))

    # do the attention here
    # flops.op(flops.op_attention, q_seqs.shape, k_seqs.shape, v_seqs.shape)
    qkv = F.scaled_dot_product_attention(q_seqs, k_seqs, v_seqs, mask, scale=scale)

    # unwindow
    qkv = torch.reshape(qkv, (b, heads, h, w, wh, ww, d_head))
    return shifted_unwindow(window_shift, qkv)


# Transformer layers


def use_flash_2(x):
    if not flags.get_use_flash_attention_2():
        return False
    if flash_attn is None:
        return False
    if x.device.type != "cuda":
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        pos = rearrange(pos, "... h w e -> ... (h w) e").to(qkv.dtype)
        theta = self.pos_emb(pos)
        if use_flash_2(qkv):
            qkv = rearrange(qkv, "n h w (t nh e) -> n (h w) t nh e", t=3, e=self.d_head)
            qkv = scale_for_cosine_sim_qkv(qkv, self.scale, 1e-6)
            theta = torch.stack((theta, theta, torch.zeros_like(theta)), dim=-3)
            qkv = apply_rotary_emb_(qkv, theta)
            # flops_shape = qkv.shape[-5], qkv.shape[-2], qkv.shape[-4], qkv.shape[-1]
            # flops.op(flops.op_attention, flops_shape, flops_shape, flops_shape)
            x = flash_attn.flash_attn_qkvpacked_func(qkv, softmax_scale=1.0)
            x = rearrange(x, "n (h w) nh e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2])
        else:
            q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh (h w) e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
            theta = theta.movedim(-2, -3)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            # flops.op(flops.op_attention, q.shape, k.shape, v.shape)
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            x = rearrange(x, "n nh (h w) e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class NeighborhoodSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.kernel_size = kernel_size
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head}, kernel_size={self.kernel_size}"

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        if natten is None:
            raise ModuleNotFoundError("natten is required for neighborhood attention")
        if natten.has_fused_na():
            q, k, v = rearrange(qkv, "n h w (t nh e) -> t n h w nh e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None], 1e-6)
            theta = self.pos_emb(pos)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            # flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
            x = natten.functional.na2d(q, k, v, self.kernel_size, scale=1.0)
            x = rearrange(x, "n h w nh e -> n h w (nh e)")
        else:
            q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-6)
            theta = self.pos_emb(pos).movedim(-2, -4)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            # flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
            qk = natten.functional.na2d_qk(q, k, self.kernel_size)
            a = torch.softmax(qk, dim=-1).to(v.dtype)
            x = natten.functional.na2d_av(a, v, self.kernel_size)
            x = rearrange(x, "n nh h w e -> n h w (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class ShiftedWindowSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, window_size, window_shift, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.window_size = window_size
        self.window_shift = window_shift
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head}, window_size={self.window_size}, window_shift={self.window_shift}"

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-6)
        theta = self.pos_emb(pos).movedim(-2, -4)
        q = apply_rotary_emb_(q, theta)
        k = apply_rotary_emb_(k, theta)
        x = apply_window_attention(self.window_size, self.window_shift, q, k, v, scale=1.0)
        x = rearrange(x, "n nh h w e -> n h w (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, cond_features, dropout=0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x, cond):
        skip = x
        x = self.norm(x, cond)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class GlobalTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, d_head, cond_features, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class NeighborhoodTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.self_attn = NeighborhoodSelfAttentionBlock(d_model, d_head, cond_features, kernel_size, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class ShiftedWindowTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, window_size, index, dropout=0.0):
        super().__init__()
        window_shift = window_size // 2 if index % 2 == 1 else 0
        self.self_attn = ShiftedWindowSelfAttentionBlock(d_model, d_head, cond_features, window_size, window_shift, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class NoAttentionTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, cond_features, dropout=0.0):
        super().__init__()
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.ff, x, cond)
        return x


class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x

class TemporalLevel(nn.ModuleList):
    def __init__(self, pad_value=None, modules=None):
        super().__init__(modules)
        self.pad_value = pad_value
    
    def smart_forward(self, layer, input, pos, cond, *args, **kwargs):
        if len(input.shape) == 4:
            return layer(input, *args, **kwargs)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                dummy_cond = torch.zeros(cond.shape, device=input.device).float().repeat(t, 1)
                self.out_shape = layer(dummy.view(b * t, c, h, w), pos, dummy_cond, *args, **kwargs).shape

            out = input.view(b * t, c, h, w)
            out_cond = cond.repeat(t, 1)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(
                            self.out_shape, device=input.device, requires_grad=False
                        )
                        * self.pad_value
                    )
                    temp[~pad_mask] = layer(out[~pad_mask], pos, out_cond, *args, **kwargs)
                    out = temp
                else:
                    out = layer(out, pos, out_cond, *args, **kwargs)
            else:
                out = layer(out, pos, out_cond, *args, **kwargs)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out
    
    def forward(self, x, pos, cond, *args, **kwargs):
        for layer in self:
            x = self.smart_forward(layer, x, pos, cond, *args, **kwargs)
        return x
            


# Mapping network

class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


# Token merging and splitting

class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2), pad_value=None):
        super().__init__()
        self.out_shape = None
        self.pad_value = pad_value
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features * self.h * self.w, out_features, bias=False))

    def forward(self, x):
        x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.h, nw=self.w)
        return self.proj(x)
    
    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(
                            self.out_shape, device=input.device, requires_grad=False
                        )
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out


class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)


class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        x = self.proj(x)
        x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        return torch.lerp(skip.to(x.dtype), x, self.fac.to(x.dtype))

class TokenSplitWithControl(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj1 = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))
        self.proj2 = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        x = self.proj1(x)
        x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        skip = self.proj2(skip)
        skip = rearrange(skip, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        return torch.lerp(skip, x, self.fac.to(x.dtype))

class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super().__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)

class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=False,
        positional_encoding=True,
        use_dropout=True
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout on the MLP-processed values
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
            use_dropout (bool): dropout on the attention masks.
        """
        class MultiHeadAttention(nn.Module):
            """Multi-Head Attention module
            Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
            """
            def __init__(self, n_head, d_k, d_in, use_dropout=True):
                super().__init__()
                self.n_head = n_head
                self.d_k = d_k
                self.d_in = d_in # e.g. self.d_model in LTAE2d
                # define H x k queries, they are input-independent in LTAE
                self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
                nn.init.normal_(self.Q, mean=0, std=(2.0 / (d_k)) ** 0.5)
                self.fc1_k = apply_wd(Linear(d_in, n_head * d_k))
                nn.init.normal_(self.fc1_k.weight, mean=0, std=(2.0 / (d_k)) ** 0.5)
                attn_dropout = 0.1 if use_dropout else 0.0
                self.dropout = nn.Dropout(attn_dropout)

            def forward(self, v, pad_mask=None):
                d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
                # values v are of shapes [B*H*W, T, self.d_in=self.d_model], e.g. [2*32*32=2048 x 4 x 256] (see: sz_b * h * w, seq_len, d)
                # where self.d_in=self.d_model is the output dimension of the FC-projected features  
                sz_b, seq_len, _ = v.size()
                q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k
                k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
                k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

                if pad_mask is not None:
                    pad_mask = pad_mask.repeat(
                        (n_head, 1)
                    ) # replicate pad_mask for each head (nxb) x lk
                # attn   is of shape [B*H*W*h, 1, T], e.g. [2*32*32*16=32768 x 1 x 4], e.g. Size([32768, 1, 4])
                # v      is of shape [B*H*W*h, T, self.d_in/h], e.g. [2*32*32*16=32768 x 4 x 256/16=16], e.g. Size([32768, 4, 16])
                # output is of shape [B*H*W*h, 1, h], e.g. [2*32*32*16=32768 x 1 x 16], e.g. Size([32768, 1, 16])
                v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
                # TODO support the flash attention 2
                attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / (d_k ** 0.5)
                attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3) if pad_mask is not None else attn
                # attn   is of shape [B*H*W*h, 1, T], e.g. [2*32*32*16=32768 x 1 x 4]
                # v      is of shape [B*H*W*h, T, self.d_in/h], e.g. [2*32*32*16=32768 x 4 x 256/16=16]
                # output is of shape [B*H*W*h, 1, h], e.g. [2*32*32*16=32768 x 1 x 16], e.g. Size([32768, 1, 16])
                attn = F.softmax(attn, dim=-1)
                attn = self.dropout(attn)
                output = torch.matmul(attn, v)
                
                attn = attn.view(n_head, sz_b, 1, seq_len)
                attn = attn.squeeze(dim=2)
                output = output.view(n_head, sz_b, 1, d_in // n_head)
                output = output.squeeze(dim=2)
                return output, attn
        
        super().__init__()
        self.in_channels = in_channels
        self.mlp = mlp
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.in_block = Linear(in_channels, d_model)
        else:
            self.d_model = in_channels
            self.in_block = nn.Identity()
        assert self.mlp[0] is None or self.mlp[0] == self.d_model 
        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None
        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model, use_dropout=use_dropout
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )
        self.out_norm = zero_init(nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        ))
        if mlp[0] is not None:
            layers = []
            for i in range(len(self.mlp) - 1):
                layers.extend(
                    [
                        apply_wd(Linear(self.mlp[i], self.mlp[i + 1])),
                        RMSNorm(self.mlp[i + 1]),
                        # nn.BatchNorm1d(self.mlp[i + 1]),
                        nn.ReLU(),
                    ]
                )
                # if i != len(self.mlp) - 2:
                #     layers.extend(
                #         [
                #             apply_wd(Linear(self.mlp[i], self.mlp[i + 1])),
                #             RMSNorm(self.mlp[i + 1]),
                #             nn.GELU(),
                #         ]
                #     )
                # else:
                #     layers.append(apply_wd(zero_init(Linear(self.mlp[i], self.mlp[i + 1]))))
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.in_block(out)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)
        # re-shaped attn to   [h x B*H*W x T], e.g. torch.Size([16, 2048, 4])
        #   in utae.py this is torch.Size([h, B, T, 32, 32])
        # re-shaped output to [h x B*H*W x d_in/h], e.g. torch.Size([16, 2048, 16])
        #   in utae.py this is torch.Size([B, 128, 32, 32])
        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        )  # Concatenate heads, out is now [B*H*W x d_in/h * h], e.g. [2048 x 256]
        # out is of shape [head x b x t x h x w]
        # out = self.dropout(out)
        # out = self.mlp(out)
        out = self.dropout(self.mlp(out))
        # after MLP, out is of shape [B*H*W x outputLayerOfMLP], e.g. [2048 x 128]
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)
        out = self.out_norm(out) if self.out_norm is not None else out
        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )
        # out  is of shape [B x outputLayerOfMLP x h x w], e.g. [2, 128, 32, 32]
        # attn is of shape [h x B x T x H x W], e.g. [16, 2, 4, 32, 32]
        if self.return_att:
            return out, attn
        else:
            return out

class LTAELayer(nn.Module):
    def __init__(
        self,
        d_model,
        cond_features,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        T=1000,
        positional_encoding=True,
        *args, **kwargs
    ):
        class MultiHeadAttention(nn.Module):
            def __init__(self, n_head, d_k, d_in, cond_features, dropout=0.0):
                super().__init__()
                self.d_k = d_k
                self.d_in = d_in
                self.d_head = d_in // n_head
                self.n_head = n_head
                self.norm = AdaRMSNorm(d_in, cond_features)
                self.fc1_k = apply_wd(Linear(d_in, d_k * n_head))
                self.dropout = nn.Dropout(dropout)
                self.out_proj = apply_wd(zero_init(Linear(d_in, d_in, bias=False)))
                self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
                nn.init.normal_(self.Q, mean=0, std=(2.0 / (d_k)) ** 0.5)

            def forward(self, v, cond, pad_mask=None):
                d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
                sz_b, h, w ,seq_len, d = v.shape
                v = v.view(sz_b, -1, seq_len, d)
                v = self.norm(v, cond)
                v = v.view(sz_b * h * w, seq_len, d)
                k = self.fc1_k(v).view(sz_b * h * w, seq_len, n_head, d_k)
                k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)
                q = torch.stack([self.Q for _ in range(sz_b * h * w)], dim=1).view(-1, d_k)
                
                if pad_mask is not None:
                    pad_mask = pad_mask.repeat(
                        (n_head, 1)
                    )
                # flops.op(flops.op_attention, q.shape, k.shape, v.shape)
                v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1), dim=0).view(n_head * sz_b * h * w, seq_len, -1)
                attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / (d_k ** 0.5)
                attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e5) if pad_mask is not None else attn
                attn = F.softmax(attn, dim=-1)
                output = torch.matmul(attn, v)
                attn = attn.view(n_head, sz_b * h * w, 1, seq_len)
                attn = attn.squeeze(dim=2)
                output = output.view(n_head, sz_b * h * w, 1, d_in // n_head)
                output = output.squeeze(dim=2)
                output = output.permute(1, 0, 2).contiguous().view(sz_b * h * w, d_in)
                output = self.dropout(output)
                output = self.out_proj(output)
                return output, attn
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert mlp[0] == d_model
        self.mlp = mlp
        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None
        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model, cond_features=cond_features, dropout=dropout
        )
        self.ff = FeedForwardBlock(d_model, self.mlp[1], cond_features, dropout=dropout)
        

    def forward(self, x, cond, batch_positions=None, pad_mask=None):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)
        out, attn = self.attention_heads(out.view(sz_b, h, w, seq_len, d), cond, pad_mask=pad_mask)
        out = out.view(sz_b * h * w, -1)
        out = self.ff(out.view(sz_b, h, w, -1), cond)
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)
        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )
        return out, attn
        
# Configuration

@dataclass
class GlobalAttentionSpec:
    d_head: int


@dataclass
class NeighborhoodAttentionSpec:
    d_head: int
    kernel_size: int


@dataclass
class ShiftedWindowAttentionSpec:
    d_head: int
    window_size: int


@dataclass
class NoAttentionSpec:
    pass


@dataclass
class LevelSpec:
    depth: int
    width: int
    d_ff: int
    self_attn: Union[GlobalAttentionSpec, NeighborhoodAttentionSpec, ShiftedWindowAttentionSpec, NoAttentionSpec]
    dropout: float


@dataclass
class MappingSpec:
    depth: int
    width: int
    d_ff: int
    dropout: float

@dataclass
class TemporalSpec:
    n_heads : int
    d_model : int
    d_k     : int
    positional_encoding : bool
    agg_mode: str 
    dropout: int
    use_dropout: bool
    mlp: list
       
# Model class

class ImageTransformerDenoiserModel(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, levels, mapping, tanh=False, control_mode=None):
        super(ImageTransformerDenoiserModel, self).__init__()
        assert control_mode in ['sum', 'conv', 'lerp', None], "control_mode must be in ['sum','conv','lerp',None]"
        self.control_mode = control_mode
        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size)

        self.time_emb = layers.FourierFeatures(1, mapping.width)
        self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.mapping = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), "mapping")

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        if control_mode == "conv":
            self.control_convs = nn.ModuleList()
        elif control_mode == "lerp":
            self.control_lerps = nn.ModuleList()
        for i, spec in enumerate(levels):
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.kernel_size, dropout=spec.dropout)
            elif isinstance(spec.self_attn, ShiftedWindowAttentionSpec):
                layer_factory = lambda i: ShiftedWindowTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.window_size, i, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NoAttentionSpec):
                layer_factory = lambda _: NoAttentionTransformerLayer(spec.width, spec.d_ff, mapping.width, dropout=spec.dropout)
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")

            if i < len(levels) - 1:
                self.down_levels.append(Level([layer_factory(i) for i in range(spec.depth)]))
                self.up_levels.append(Level([layer_factory(i + spec.depth) for i in range(spec.depth)]))
            else:
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])
            
            if control_mode == "conv":
                self.control_convs.append(nn.Conv2d(2 * spec.width, spec.width, 1, 1))
            elif control_mode == "lerp":
                self.control_lerps.append(TokenSplitWithControl(spec.width, spec.width))

        self.merges = nn.ModuleList([TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.splits = nn.ModuleList([TokenSplit(spec_2.width, spec_1.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])

        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(levels[0].width, out_channels, patch_size)
        nn.init.zeros_(self.patch_out.proj.weight)
        self.tanh = nn.Tanh() if tanh else nn.Identity()

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0}
        ]
        return groups

    def forward(self, x, timesteps, control = None):
        # Patching
        if control is not None:
            assert isinstance(control, list), "control must be a list!"
        x = x.movedim(-3, -1)
        x = self.patch_in(x)
        # TODO: pixel aspect ratio for nonsquare patches
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(x.shape[-3], x.shape[-2], 2)


        c_noise = timesteps # 外面处理过了
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        cond = self.mapping(time_emb)

        # Hourglass transformer
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)

        if self.control_mode == "sum":
            index = len(control) - 1
            x = x + control[index]
        elif self.control_mode == "conv":
            index = len(control) - 1
            x = self.control_convs[index](torch.cat([x, control[index]], dim=-1).permute(0,3,1,2)).permute(0,2,3,1)        
        elif self.control_mode == "lerp":
            index = len(control) - 1
            x = self.control_lerps[index](x, control[index])
        elif self.control_mode is None:
            pass
        else:
            raise NotImplementedError(f"control mode `{self.control_mode}` is not implemented!")
        x = self.mid_level(x, pos, cond)
        
        for i, (up_level, split, skip, pos) in enumerate(reversed(list(zip(self.up_levels, self.splits, skips, poses)))):
            x = split(x, skip)
            if self.control_mode == "sum":
                index = len(control) - i - 2
                x = x + control[index]
            elif self.control_mode == "conv":
                index = len(control) - i - 2
                x = self.control_convs[index](torch.cat([x, control[index]], dim=-1).permute(0,3,1,2)).permute(0,2,3,1)              
            elif self.control_mode == "lerp":
                index = len(control) - i - 2
                x = self.control_lerps[index](x, control[index])   
            elif self.control_mode is None:
                pass
            else:
                raise NotImplementedError(f"control mode `{self.control_mode}` is not implemented!")
            x = up_level(x, pos, cond)

        # Unpatching
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = self.tanh(x)
        x = x.movedim(-1, -3)

        return x
    
class ImageTemporalTransformerDenoiserModel(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        patch_size,
        levels,
        mapping,
        temporal,
        pad_value=None,
        tanh=False,
    ):
        super(ImageTemporalTransformerDenoiserModel, self).__init__()
        self.pad_value = pad_value
        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size, pad_value)

        self.time_emb = layers.FourierFeatures(1, mapping.width)
        self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.mapping = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), "mapping")

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        for i, spec in enumerate(levels):
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.kernel_size, dropout=spec.dropout)
            elif isinstance(spec.self_attn, ShiftedWindowAttentionSpec):
                layer_factory = lambda i: ShiftedWindowTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.window_size, i, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NoAttentionSpec):
                layer_factory = lambda _: NoAttentionTransformerLayer(spec.width, spec.d_ff, mapping.width, dropout=spec.dropout)
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")

            if i < len(levels) - 1:
                self.down_levels.append(TemporalLevel(pad_value=pad_value, modules=[layer_factory(i) for i in range(spec.depth)]))
                self.up_levels.append(Level([layer_factory(i + spec.depth) for i in range(spec.depth)]))
            else:
                self.mid_level = TemporalLevel(pad_value=pad_value, modules=[layer_factory(i) for i in range(spec.depth)])
                self.temporal_encoder = LTAELayer( # LTAE2d(
                    in_channels=spec.width,
                    d_model=temporal.d_model,
                    n_head=temporal.n_heads,
                    mlp=temporal.mlp,
                    return_att=True,
                    d_k=temporal.d_k,
                    positional_encoding=temporal.positional_encoding,
                    dropout=temporal.dropout,
                    use_dropout=temporal.use_dropout,
                    # add new param
                    cond_features=mapping.width
                )
                self.temporal_aggregator = Temporal_Aggregator(mode=temporal.agg_mode)

        self.merges = nn.ModuleList([TokenMerge(spec_1.width, spec_2.width, pad_value=pad_value) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.splits = nn.ModuleList([TokenSplit(spec_2.width, spec_1.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])

        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(levels[0].width, out_channels, patch_size)
        nn.init.zeros_(self.patch_out.proj.weight)
        # TODO 把时序部分的0初始化也做好
        self.tanh = nn.Tanh() if tanh else nn.Identity()

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0}
        ]
        return groups

    def forward(self, x, timesteps, dates=None, return_attn=False):
        # Patching
        pad_mask = (
            (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  if self.pad_value is not None else None # BxT pad mask
        x = x.movedim(-3, -1)
        x = self.patch_in.smart_forward(x)
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(x.shape[-3], x.shape[-2], 2)

        c_noise = timesteps
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        cond = self.mapping(time_emb)

        # Hourglass transformer
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond)
            skips.append(x)
            poses.append(pos)
            x = merge.smart_forward(x)
            pos = downscale_pos(pos)

        x = self.mid_level(x, pos, cond)
        # x = self.mid_split(x)
        x = x.movedim(-1,-3)
        # x, att = self.temporal_encoder(
        #     x, batch_positions=dates, pad_mask=pad_mask
        # )
        x, att = self.temporal_encoder(
            x, cond, batch_positions=dates, pad_mask=pad_mask
        )
        x = x.movedim(-3,-1)
        # x = self.mid_merge(x)
        
        for up_level, split, skip, pos in reversed(list(zip(self.up_levels, self.splits, skips, poses))):
            # skip = split_wo_skip(skip)
            skip = skip.movedim(-1,-3)
            skip = self.temporal_aggregator(
                skip, pad_mask=pad_mask, attn_mask=att
            )
            skip = skip.movedim(-3,-1)
            x = split(x, skip)    
            x = up_level(x, pos, cond)

        # Unpatching
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = self.tanh(x)
        x = x.movedim(-1, -3)
        if return_attn:
            return x, att
        else:
            return x
    
class ImageTransformerDenoiserModelInterface(ImageTransformerDenoiserModel):
    def __init__(
        self,
        in_channels=13,
        out_channels=13,
        patch_size=(4,4),
        widths=[48,96,192,384],
        depths=[4,4,6,8],
        d_ffs=[96,192,384,768],
        self_attns=[
            {"type": "neighborhood", "d_head": 48, "kernel_size": 7},
            {"type": "neighborhood", "d_head": 48, "kernel_size": 7},
            {"type": "global", "d_head": 48},
            {"type": "global", "d_head": 48}
        ],
        dropout_rate=[0.0,0.0,0.1,0.1],
        mapping_depth=2,
        mapping_width=256,
        mapping_d_ff=512,
        mapping_dropout_rate=0.0,
        tanh=False,
        control_mode=None
    ):
        assert len(widths) == len(depths)
        assert len(widths) == len(d_ffs)
        assert len(widths) == len(self_attns)
        assert len(widths) == len(dropout_rate)
        levels = []
        for depth, width, d_ff, self_attn, dropout in \
            zip(depths, widths, d_ffs, self_attns, dropout_rate):
                if self_attn['type'] == 'global':
                    self_attn = GlobalAttentionSpec(self_attn.get('d_head', 64))
                elif self_attn['type'] == 'neighborhood':
                    self_attn = NeighborhoodAttentionSpec(self_attn.get('d_head', 64), self_attn.get('kernel_size', 7))
                elif self_attn['type'] == 'shifted-window':
                    self_attn = ShiftedWindowAttentionSpec(self_attn.get('d_head', 64), self_attn['window_size'])
                elif self_attn['type'] == 'none':
                    self_attn = NoAttentionSpec()
                else:
                    raise ValueError(f'unsupported self attention type {self_attn["type"]}')
                levels.append(LevelSpec(depth, width, d_ff, self_attn, dropout))
        mapping = MappingSpec(mapping_depth, mapping_width, mapping_d_ff, mapping_dropout_rate)
        super().__init__(in_channels, out_channels, patch_size, levels, mapping, tanh, control_mode)

class ImageTemporalTransformerDenoiserInterface(ImageTemporalTransformerDenoiserModel):
    def __init__(
        self,
        in_channels=13,
        out_channels=13,
        patch_size=(4,4),
        widths=[48,96,192,384],
        depths=[4,4,6,8],
        d_ffs=[96,192,384,768],
        self_attns=[
            {"type": "neighborhood", "d_head": 48, "kernel_size": 7},
            {"type": "neighborhood", "d_head": 48, "kernel_size": 7},
            {"type": "global", "d_head": 48},
            {"type": "global", "d_head": 48}
        ],
        dropout_rate=[0.0,0.0,0.1,0.1],
        mapping_depth=2,
        mapping_width=256,
        mapping_d_ff=512,
        mapping_dropout_rate=0.0,
        temporal_n_heads=16,
        temporal_d_model=768,
        temporal_d_k=4,
        temporal_positional_encoding=True,
        temporal_agg_mode="att_group",
        temporal_dropout=0.0,
        temporal_use_drouput=False,
        temporal_mlp=[768,1536],
        pad_value=0,
        tanh=False,
    ):
        assert len(widths) == len(depths)
        assert len(widths) == len(d_ffs)
        assert len(widths) == len(self_attns)
        assert len(widths) == len(dropout_rate)
        levels = []
        for depth, width, d_ff, self_attn, dropout in \
            zip(depths, widths, d_ffs, self_attns, dropout_rate):
                if self_attn['type'] == 'global':
                    self_attn = GlobalAttentionSpec(self_attn.get('d_head', 64))
                elif self_attn['type'] == 'neighborhood':
                    self_attn = NeighborhoodAttentionSpec(self_attn.get('d_head', 64), self_attn.get('kernel_size', 7))
                elif self_attn['type'] == 'shifted-window':
                    self_attn = ShiftedWindowAttentionSpec(self_attn.get('d_head', 64), self_attn['window_size'])
                elif self_attn['type'] == 'none':
                    self_attn = NoAttentionSpec()
                else:
                    raise ValueError(f'unsupported self attention type {self_attn["type"]}')
                levels.append(LevelSpec(depth, width, d_ff, self_attn, dropout))
        mapping = MappingSpec(mapping_depth, mapping_width, mapping_d_ff, mapping_dropout_rate)
        temporal = TemporalSpec(temporal_n_heads, temporal_d_model, temporal_d_k, temporal_positional_encoding, temporal_agg_mode, temporal_dropout, temporal_use_drouput, temporal_mlp)
        super().__init__(in_channels, out_channels, patch_size, levels, mapping, temporal, pad_value, tanh)
        
