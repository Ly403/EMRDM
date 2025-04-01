from torch import nn
from .modules import AbstractEmbModel
from ..diffusionmodules.k_diffusion.image_transformer import *
from ..diffusionmodules.k_diffusion import layers

# 下面是不带条件控制的Attention Block
class SelfAttentionBlockWithoutCond(nn.Module):
    def __init__(self, d_model, d_head, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = RMSNorm(d_model)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(self, x, pos):
        skip = x
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        pos = rearrange(pos, "... h w e -> ... (h w) e").to(qkv.dtype)
        theta = self.pos_emb(pos)
        if use_flash_2(qkv):
            qkv = rearrange(qkv, "n h w (t nh e) -> n (h w) t nh e", t=3, e=self.d_head)
            qkv = scale_for_cosine_sim_qkv(qkv, self.scale, 1e-6)
            theta = torch.stack((theta, theta, torch.zeros_like(theta)), dim=-3)
            qkv = apply_rotary_emb_(qkv, theta)
            flops_shape = qkv.shape[-5], qkv.shape[-2], qkv.shape[-4], qkv.shape[-1]
            flops.op(flops.op_attention, flops_shape, flops_shape, flops_shape)
            x = flash_attn.flash_attn_qkvpacked_func(qkv, softmax_scale=1.0)
            x = rearrange(x, "n (h w) nh e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2])
        else:
            q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh (h w) e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
            theta = theta.movedim(-2, -3)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_attention, q.shape, k.shape, v.shape)
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            x = rearrange(x, "n nh (h w) e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class NeighborhoodSelfAttentionBlockWithoutCond(nn.Module):
    def __init__(self, d_model, d_head, kernel_size, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.kernel_size = kernel_size
        self.norm = RMSNorm(d_model)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head}, kernel_size={self.kernel_size}"

    def forward(self, x, pos):
        skip = x
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        if natten is None:
            raise ModuleNotFoundError("natten is required for neighborhood attention")
        if natten.has_fused_na():
            q, k, v = rearrange(qkv, "n h w (t nh e) -> t n h w nh e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None], 1e-6)
            theta = self.pos_emb(pos)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
            x = natten.functional.na2d(q, k, v, self.kernel_size, scale=1.0)
            x = rearrange(x, "n h w nh e -> n h w (nh e)")
        else:
            q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-6)
            theta = self.pos_emb(pos).movedim(-2, -4)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
            qk = natten.functional.na2d_qk(q, k, self.kernel_size)
            a = torch.softmax(qk, dim=-1).to(v.dtype)
            x = natten.functional.na2d_av(a, v, self.kernel_size)
            x = rearrange(x, "n nh h w e -> n h w (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class ShiftedWindowSelfAttentionBlockWithoutCond(nn.Module):
    def __init__(self, d_model, d_head, window_size, window_shift, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.window_size = window_size
        self.window_shift = window_shift
        self.norm = RMSNorm(d_model)
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


class FeedForwardBlockWithoutCond(nn.Module):
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


class GlobalTransformerLayerWithoutCond(nn.Module):
    def __init__(self, d_model, d_ff, d_head, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttentionBlockWithoutCond(d_model, d_head, dropout=dropout)
        self.ff = FeedForwardBlockWithoutCond(d_model, d_ff, dropout=dropout)

    def forward(self, x, pos):
        x = checkpoint(self.self_attn, x, pos)
        x = checkpoint(self.ff, x)
        return x


class NeighborhoodTransformerLayerWithoutCond(nn.Module):
    def __init__(self, d_model, d_ff, d_head, kernel_size, dropout=0.0):
        super().__init__()
        self.self_attn = NeighborhoodSelfAttentionBlockWithoutCond(d_model, d_head, kernel_size, dropout=dropout)
        self.ff = FeedForwardBlockWithoutCond(d_model, d_ff, dropout=dropout)

    def forward(self, x, pos):
        x = checkpoint(self.self_attn, x, pos)
        x = checkpoint(self.ff, x)
        return x


class ShiftedWindowTransformerLayerWithoutCond(nn.Module):
    def __init__(self, d_model, d_ff, d_head, window_size, index, dropout=0.0):
        super().__init__()
        window_shift = window_size // 2 if index % 2 == 1 else 0
        self.self_attn = ShiftedWindowSelfAttentionBlockWithoutCond(d_model, d_head, window_size, window_shift, dropout=dropout)
        self.ff = FeedForwardBlockWithoutCond(d_model, d_ff, dropout=dropout)

    def forward(self, x, pos):
        x = checkpoint(self.self_attn, x, pos)
        x = checkpoint(self.ff, x)
        return x


class NoAttentionTransformerLayerWithoutCond(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.ff = FeedForwardBlockWithoutCond(d_model, d_ff, dropout=dropout)

    def forward(self, x, pos):
        x = checkpoint(self.ff, x)
        return x

class ImageTransformerEncoder(AbstractEmbModel):
    def __init__(self, in_channels, patch_size, levels):
        super(ImageTransformerEncoder, self).__init__()
        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size)

        self.down_levels = nn.ModuleList()
        for i, spec in enumerate(levels):
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayerWithoutCond(spec.width, spec.d_ff, spec.self_attn.d_head, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayerWithoutCond(spec.width, spec.d_ff, spec.self_attn.d_head, spec.self_attn.kernel_size, dropout=spec.dropout)
            elif isinstance(spec.self_attn, ShiftedWindowAttentionSpec):
                layer_factory = lambda i: ShiftedWindowTransformerLayerWithoutCond(spec.width, spec.d_ff, spec.self_attn.d_head, spec.self_attn.window_size, i, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NoAttentionSpec):
                layer_factory = lambda _: NoAttentionTransformerLayerWithoutCond(spec.width, spec.d_ff, dropout=spec.dropout)
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")

            if i < len(levels) - 1:
                self.down_levels.append(Level([layer_factory(i) for i in range(spec.depth)]))
            else:
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])

        self.merges = nn.ModuleList([TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])

    def forward(self, x):
        # Patching
        x = x.movedim(-3, -1)
        x = self.patch_in(x)
        # TODO: pixel aspect ratio for nonsquare patches
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(x.shape[-3], x.shape[-2], 2)

        # Hourglass transformer
        skips = []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos)
            skips.append(x)
            x = merge(x)
            pos = downscale_pos(pos)

        x = self.mid_level(x, pos)
        skips.append(x)
        return {"control": skips}