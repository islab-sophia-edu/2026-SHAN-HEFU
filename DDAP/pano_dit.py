"""Spherical DiT with a wide, shallow DDT head for PanoMAE token flow matching."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_PanoMAE_gctt import OrientedSphericalPositionalEncoding, SphericalAttentionBias


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1.0 + scale) + shift


def apply_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if gate.ndim == 2:
        gate = gate.unsqueeze(1)
    return x * gate


class ContinuousTimeEmbedder(nn.Module):
    """Gaussian Fourier embedding for continuous time t in [0,1]."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        frequency_scale: float = 16.0,
    ) -> None:
        super().__init__()
        if frequency_embedding_size % 2 != 0:
            raise ValueError("frequency_embedding_size must be even")
        half = frequency_embedding_size // 2
        frequencies = torch.randn(half) * float(frequency_scale)
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError(f"time must be [B], got {tuple(time.shape)}")
        phase = 2.0 * math.pi * time.float().unsqueeze(1) * self.frequencies.unsqueeze(0)
        embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)
        return self.mlp(embedding)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, drop: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attention = attention + attn_bias.to(dtype=attention.dtype)
        attention = self.attn_drop(attention.softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaLNZeroBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, drop: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, drop)
        self.mlp = FeedForward(hidden_size, mlp_ratio, drop)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(condition).chunk(6, dim=-1)
        x = x + apply_gate(
            self.attn(modulate(self.norm1(x), shift_a, scale_a), attn_bias),
            gate_a,
        )
        x = x + apply_gate(
            self.mlp(modulate(self.norm2(x), shift_m, scale_m)),
            gate_m,
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class PanoDiTDDT(nn.Module):
    """Velocity predictor on spherical PanoMAE latents.

    For high-dimensional representation latents, both the base width and the
    wide head must not be narrower than the encoder token dimension.  The class
    raises immediately instead of silently training an undercomplete denoiser.
    """

    def __init__(
        self,
        latent_dim: int,
        base_hidden_size: int,
        base_depth: int = 12,
        base_num_heads: int = 16,
        head_hidden_size: int = 2048,
        head_depth: int = 2,
        head_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        geometric_bias: bool = True,
        use_gctt: bool = True,
        use_angular_bias: bool = True,
        use_gauge_bias: bool = True,
        angular_bias_init_slope: float = 1.0,
        gauge_bias_init: float = 0.0,
        dropout: float = 0.0,
        head_use_geometry: bool = False,
    ) -> None:
        super().__init__()
        latent_dim = int(latent_dim)
        base_hidden_size = int(base_hidden_size)
        head_hidden_size = int(head_hidden_size)
        if base_hidden_size < latent_dim:
            raise ValueError(
                f"base_hidden_size={base_hidden_size} is narrower than latent_dim={latent_dim}. "
                "Use at least the PanoMAE token dimension."
            )
        if head_hidden_size < latent_dim:
            raise ValueError(
                f"head_hidden_size={head_hidden_size} is narrower than latent_dim={latent_dim}."
            )

        self.latent_dim = latent_dim
        self.base_hidden_size = base_hidden_size
        self.head_hidden_size = head_hidden_size
        self.use_gctt = bool(use_gctt)
        self.use_angular_bias = bool(use_angular_bias)

        self.base_input = nn.Linear(latent_dim, base_hidden_size)
        self.head_input = nn.Linear(latent_dim, head_hidden_size)
        self.base_geometry = OrientedSphericalPositionalEncoding(
            d_model=base_hidden_size,
            geometric_bias=geometric_bias,
            use_oriented_frame=use_gctt,
        )
        self.head_geometry = (
            OrientedSphericalPositionalEncoding(
                d_model=head_hidden_size,
                geometric_bias=geometric_bias,
                use_oriented_frame=use_gctt,
            )
            if head_use_geometry else None
        )
        self.time_embedder = ContinuousTimeEmbedder(base_hidden_size)

        self.base_blocks = nn.ModuleList([
            AdaLNZeroBlock(base_hidden_size, base_num_heads, mlp_ratio, dropout)
            for _ in range(base_depth)
        ])
        self.base_to_head = (
            nn.Linear(base_hidden_size, head_hidden_size)
            if base_hidden_size != head_hidden_size else nn.Identity()
        )
        self.head_blocks = nn.ModuleList([
            AdaLNZeroBlock(head_hidden_size, head_num_heads, mlp_ratio, dropout)
            for _ in range(head_depth)
        ])
        self.final_layer = FinalLayer(head_hidden_size, latent_dim)

        if self.use_angular_bias:
            self.base_attn_bias = SphericalAttentionBias(
                num_heads=base_num_heads,
                init_max_slope=angular_bias_init_slope,
                use_gauge_bias=use_gauge_bias and use_gctt,
                gauge_bias_init=gauge_bias_init,
            )
            self.head_attn_bias = SphericalAttentionBias(
                num_heads=head_num_heads,
                init_max_slope=angular_bias_init_slope,
                use_gauge_bias=use_gauge_bias and use_gctt,
                gauge_bias_init=gauge_bias_init,
            )
        else:
            self.base_attn_bias = None
            self.head_attn_bias = None

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(init)
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)
        for block in list(self.base_blocks) + list(self.head_blocks):
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.modulation[-1].weight)
        nn.init.zeros_(self.final_layer.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    @staticmethod
    def _ensure_gauge(gauge_angles: Optional[torch.Tensor], angles: torch.Tensor) -> torch.Tensor:
        if gauge_angles is None:
            return torch.zeros(
                angles.shape[0], angles.shape[1], 1,
                device=angles.device, dtype=angles.dtype,
            )
        if gauge_angles.ndim == 2:
            gauge_angles = gauge_angles.unsqueeze(-1)
        return gauge_angles.to(device=angles.device, dtype=angles.dtype)

    def forward(
        self,
        z_t: torch.Tensor,
        time: torch.Tensor,
        angles: torch.Tensor,
        gauge_angles: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z_t.ndim != 3 or z_t.shape[-1] != self.latent_dim:
            raise ValueError(f"z_t must be [B,N,{self.latent_dim}], got {tuple(z_t.shape)}")
        if angles.shape[:2] != z_t.shape[:2]:
            raise ValueError("z_t and angles must have matching [B,N]")
        if time.ndim != 1 or time.shape[0] != z_t.shape[0]:
            raise ValueError("time must be [B]")

        gauge = self._ensure_gauge(gauge_angles, angles) if self.use_gctt else None
        base_bias = head_bias = None
        if self.use_angular_bias:
            base_bias = self.base_attn_bias(
                angles,
                gauge_angles=gauge if self.use_gctt else None,
                include_cls=False,
            )
            head_bias = self.head_attn_bias(
                angles,
                gauge_angles=gauge if self.use_gctt else None,
                include_cls=False,
            )

        time_condition = self.time_embedder(time).to(dtype=z_t.dtype)
        base = self.base_input(z_t)
        base = base + self.base_geometry(
            angles,
            gauge_angles=gauge if self.use_gctt else None,
        ).to(dtype=base.dtype)
        for block in self.base_blocks:
            base = block(base, time_condition, base_bias)

        token_condition = self.base_to_head(F.silu(base + time_condition.unsqueeze(1)))
        head = self.head_input(z_t)
        if self.head_geometry is not None:
            head = head + self.head_geometry(
                angles,
                gauge_angles=gauge if self.use_gctt else None,
            ).to(dtype=head.dtype)
        for block in self.head_blocks:
            head = block(head, token_condition, head_bias)
        return self.final_layer(head, token_condition)
