# --------------------------------------------------------
# SparseFormer building blocks
# Copyright 2023 Ziteng Gao
# Licensed under The MIT License
# Written by Ziteng Gao (https://github.com/showlab/sparseformer)
#
# Modified: reduced to the units used by the recurrent model; the unit's
# per-iteration step is exposed so a caller can run one repeat at a time.
# --------------------------------------------------------
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

LAYER_NORM = partial(nn.LayerNorm, eps=1e-6)


def maybe_promote(x: torch.Tensor) -> torch.Tensor:
    """fp16 -> fp32 for tensors that feed the sampling / RoI arithmetic."""
    return x.float() if x.dtype == torch.float16 else x


@torch.no_grad()
def init_layer_norm_unit_norm(layer: nn.LayerNorm, gamma=1.0):
    width = layer.normalized_shape[0]
    nn.init.ones_(layer.weight)
    layer.weight.data.mul_(gamma * (width ** -0.5))


@torch.no_grad()
def init_linear_params(weight, bias):
    nn.init.trunc_normal_(weight, std=.02)
    if bias is not None:
        nn.init.constant_(bias, 0)


class RestrictGradNorm(torch.autograd.Function):
    """Identity in the forward pass; clamps the gradient of the sampling coordinates
    to a local range, since bilinear sampling only gives meaningful gradients within
    about one feature-map cell.  GRAD_SCALE follows the AMP loss scale."""
    GRAD_SCALE = 1.0

    @staticmethod
    def forward(ctx, x, norm=0.1):
        ctx.norm = norm
        return x

    @staticmethod
    def backward(ctx, grad_output):
        norm = ctx.norm * RestrictGradNorm.GRAD_SCALE
        return grad_output.clone().clamp(-norm, norm), None


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Stochastic depth.  Tokens are laid out [N, B, C], so the batch is dim 1."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (1, x.shape[1],) + (1,) * (x.ndim - 2)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


@autocast('cuda', enabled=False)
def roi_adjust(token_roi: torch.Tensor, token_adjust: torch.Tensor):
    """Shift a box by (dx, dy) in units of its own size and rescale it by exp(dw, dh)."""
    token_xy = (token_roi[..., 2:] + token_roi[..., :2]) * 0.5
    token_wh = (token_roi[..., 2:] - token_roi[..., :2]).abs()
    token_xy = token_xy + token_adjust[..., :2] * token_wh
    token_wh = token_wh * token_adjust[..., 2:].exp()
    return torch.cat([token_xy - 0.5 * token_wh, token_xy + 0.5 * token_wh], dim=-1)


@autocast('cuda', enabled=False)
def translate_to_absolute_coordinates(token_roi, relative_offset_xy, num_heads, num_points):
    """Per-token sampling offsets (standardised per token) -> absolute image coordinates."""
    batch, num_tokens, _ = relative_offset_xy.shape
    relative_offset_xy = relative_offset_xy.view(batch, num_tokens, 1, num_heads * num_points, 2)
    offset_mean = relative_offset_xy.mean(-2, keepdim=True)
    offset_std = relative_offset_xy.std(-2, keepdim=True) + 1e-7
    relative_offset_xy = (relative_offset_xy - offset_mean) / (3 * offset_std)

    token_xy = (token_roi[:, :, 2:] + token_roi[:, :, :2]) / 2.0
    roi_wh = token_roi[:, :, 2:] - token_roi[:, :, :2]
    relative_offset_xy = relative_offset_xy[..., :2] * roi_wh.view(batch, num_tokens, 1, 1, 2)
    return token_xy.view(batch, num_tokens, 1, 1, 2) + relative_offset_xy


@autocast('cuda', enabled=False)
def feat_sampling_2d(sample_points, value, n_points=1):
    """Bilinear reads of the feature map at the given points -> [B, N, 1, P, C]."""
    batch, Hq, Wq, num_points_per_head, _ = sample_points.shape
    batch, channel, height, width = value.shape
    n_heads = num_points_per_head // n_points

    sample_points = sample_points.view(batch, Hq, Wq, n_heads, n_points, 2) \
        .permute(0, 3, 1, 2, 4, 5).contiguous().flatten(0, 1)
    sample_points = RestrictGradNorm.apply(sample_points, 1.0 / height)
    sample_points = sample_points.flatten(2, 3) * 2.0 - 1.0
    value = value.view(batch * n_heads, channel // n_heads, height, width)
    out = F.grid_sample(value, sample_points, mode='bilinear', padding_mode='zeros',
                        align_corners=False)
    out = out.view(batch, n_heads, channel // n_heads, Hq, Wq, n_points)
    return out.permute(0, 3, 4, 1, 5, 2).flatten(1, 2)


@autocast('cuda', enabled=False)
def layer_norm_by_dim(x: torch.Tensor, dim=-1):
    x = x - x.mean(dim, keepdim=True)
    return x / (x.var(dim=dim, keepdim=True) + 1e-7).sqrt()


class AdaptiveMixing(nn.Module):
    """Mixes a token's sampled points with channel and point mixing weights that are
    generated from the token itself."""

    def __init__(self, in_dim, in_points, n_groups, query_dim=None, out_dim=None,
                 out_points=None, out_query_dim=None):
        super().__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        out_query_dim = out_query_dim if out_query_dim is not None else query_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_points = out_points
        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = self.eff_in_dim
        total = self.eff_in_dim * self.eff_out_dim + self.in_points * self.out_points

        self.parameter_generator = nn.Sequential(
            LAYER_NORM(self.query_dim),
            nn.Linear(self.query_dim, self.query_dim // 4),
            nn.Linear(self.query_dim // 4, total * self.n_groups),
        )
        self.m_beta = nn.Parameter(torch.zeros(self.eff_out_dim))
        self.s_beta = nn.Parameter(torch.zeros(self.out_points))
        # named `tuo_proj` (sic) upstream; kept for checkpoint compatibility
        self.tuo_proj = nn.Linear(n_groups * self.eff_out_dim * self.out_points, out_query_dim)
        self.act = nn.GELU()

    @torch.no_grad()
    def init_layer(self):
        init_layer_norm_unit_norm(self.parameter_generator[0], gamma=1.)
        nn.init.zeros_(self.parameter_generator[-1].weight)
        bias = self.parameter_generator[-1].bias
        n = self.eff_in_dim * self.eff_out_dim
        nn.init.xavier_uniform_(bias[:n].view(self.eff_in_dim, self.eff_out_dim), gain=1)
        nn.init.xavier_uniform_(bias[n:].view(self.in_points, self.out_points), gain=1)

    def forward(self, x, query):
        B, N, g, P, C = x.size()
        G = self.n_groups
        out = x.reshape(B * N * G, P, C)
        params = self.parameter_generator(query).reshape(B * N * G, -1)
        channel_mixing, spatial_mixing = params.split_with_sizes(
            [self.eff_in_dim * self.eff_out_dim, self.out_points * self.in_points], dim=-1)
        channel_mixing = channel_mixing.reshape(B * N * G, self.eff_in_dim, self.eff_out_dim)
        spatial_mixing = spatial_mixing.reshape(B * N * G, self.out_points, self.in_points)

        out = torch.baddbmm(self.m_beta.view(1, 1, self.eff_out_dim), out, channel_mixing)
        out = self.act(out)
        out = torch.baddbmm(self.s_beta.view(1, self.out_points, 1), spatial_mixing, out)
        out = self.act(out)
        return self.tuo_proj(out.reshape(B, N, -1))


class SFUnit(nn.Module):
    """One SparseFormer block over a set of latent tokens [N, B, C] with boxes [N, B, 4].

    A step is: self-attention -> (RoI adjustment) -> (sampling + adaptive mixing) -> FFN.
    `repeats` steps share the weights; each has its own drop-path rate."""

    def __init__(self, dim, conv_dim, num_sampling_points, sampling_enabled=False,
                 adjusting_enabled=False, mlp_ratio=4, repeats=1, drop_path=(0.,), head_dim=0):
        super().__init__()
        self.dim = dim
        self.num_sampling_points = num_sampling_points
        self.sampling_enabled = sampling_enabled
        self.adjusting_enabled = adjusting_enabled
        self.repeats = repeats
        self.drop_path = list(drop_path)

        self.adaptive_mixing = AdaptiveMixing(conv_dim, num_sampling_points, 1, query_dim=dim,
                                              out_query_dim=dim) if sampling_enabled else None
        self.ffn = nn.Sequential(LAYER_NORM(dim), nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))
        # head_dim=0: the upstream rule of 64 heads at every width
        num_heads = (dim // head_dim) if head_dim else (64 if dim >= 64 else dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0)
        self.ln_attn = LAYER_NORM(dim)
        if sampling_enabled:
            self.roi_offset_module = nn.Sequential(
                LAYER_NORM(dim), nn.Linear(dim, num_sampling_points * 2, bias=False))
            self.roi_offset_bias = nn.Parameter(torch.randn(num_sampling_points * 2))
        if adjusting_enabled:
            self.roi_adjust_module = nn.Sequential(LAYER_NORM(dim), nn.Linear(dim, 4, bias=False))
        self.dropout = DropPath(self.drop_path[0])

    def strip_acquisition(self):
        """Turn the unit into a plain transformer block (no RoI, no sampling)."""
        for name in ('adaptive_mixing', 'roi_offset_module', 'roi_offset_bias', 'roi_adjust_module'):
            if hasattr(self, name):
                delattr(self, name)
        self.adaptive_mixing = None
        self.sampling_enabled = self.adjusting_enabled = False

    @torch.no_grad()
    def init_layer(self):
        init_linear_params(self.attn.in_proj_weight, self.attn.in_proj_bias)
        if self.adjusting_enabled:
            init_layer_norm_unit_norm(self.roi_adjust_module[0], 1.)
            nn.init.zeros_(self.roi_adjust_module[-1].weight)
        if self.sampling_enabled:
            init_layer_norm_unit_norm(self.roi_offset_module[0], 1.)
            # initial sampling pattern: a regular grid inside the box
            root = int(self.num_sampling_points ** 0.5)
            x = torch.linspace(0.5 / root, 1 - 0.5 / root, root).view(1, -1, 1).repeat(root, 1, 1)
            y = torch.linspace(0.5 / root, 1 - 0.5 / root, root).view(-1, 1, 1).repeat(1, root, 1)
            grid = torch.cat([x, y], dim=-1).view(root ** 2, -1)
            self.roi_offset_bias.view(-1, 2).data[:root ** 2] = grid - 0.5

    def shift_token_roi(self, tokens, token_roi):
        return roi_adjust(token_roi, self.dropout(self.roi_adjust_module(tokens)))

    def sampling_mixing(self, feat, tokens, token_roi):
        roi_offset = self.roi_offset_bias.view(1, 1, -1).repeat(tokens.size(0), tokens.size(1), 1)
        roi_offset = roi_offset + self.roi_offset_module(tokens)
        points = translate_to_absolute_coordinates(
            token_roi.transpose(0, 1), roi_offset.transpose(0, 1), 1, self.num_sampling_points)
        sampled = feat_sampling_2d(points, feat, n_points=self.num_sampling_points)
        src = self.adaptive_mixing(sampled, tokens.transpose(0, 1)).transpose(0, 1)
        return tokens + self.dropout(src)

    def step(self, feat, tokens, token_roi, rep=0):
        self.dropout.drop_prob = self.drop_path[min(rep, len(self.drop_path) - 1)]
        src = self.ln_attn(tokens)
        tokens = tokens + self.dropout(self.attn(src, src, src)[0])
        if self.adjusting_enabled:
            token_roi = self.shift_token_roi(tokens, token_roi)
        if self.sampling_enabled:
            tokens = self.sampling_mixing(feat, tokens, token_roi)
        tokens = tokens + self.dropout(self.ffn(tokens))
        return tokens, token_roi


class AvgTokenHead(nn.Module):
    def __init__(self, in_dim, num_classes=1000):
        super().__init__()
        self.norm = LAYER_NORM(in_dim)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, tokens):                          # [N, B, C] -> [B, classes]
        return self.classifier(self.norm(tokens.mean(dim=0)))


class EarlyConvolution(nn.Module):
    """The conv stem: one 7x7 stride-2 conv and a max-pool, i.e. stride 4 overall."""

    def __init__(self, conv_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv_dim, kernel_size=7, stride=2, padding=3, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return layer_norm_by_dim(self.maxpool(self.relu(self.conv1(x))), 1)
