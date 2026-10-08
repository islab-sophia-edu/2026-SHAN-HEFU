"""Minimal Gaussian diffusion utilities for PanoMAE token latents."""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000.0 / timesteps
    return torch.linspace(scale * 1e-4, scale * 2e-2, timesteps).clamp(max=0.999)


def extract(buffer: torch.Tensor, t: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    out = buffer.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """Fixed-variance DDPM training with deterministic/stochastic DDIM sampling."""

    def __init__(self, timesteps: int = 1000, schedule: str = "cosine") -> None:
        super().__init__()
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        self.timesteps = int(timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars)
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alpha_bars, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alpha_bars, t, x_start.shape) * noise
        )

    @staticmethod
    def noise_loss(
        predicted_noise: torch.Tensor,
        true_noise: torch.Tensor,
        token_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_token = (predicted_noise.float() - true_noise.float()).square().mean(-1)
        if token_weights is None:
            return per_token.mean()
        token_weights = token_weights.to(per_token.device, per_token.dtype)
        return (per_token * token_weights).sum() / token_weights.sum().clamp_min(1e-6)

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Sequence[int],
        angles: torch.Tensor,
        gauge_angles: Optional[torch.Tensor] = None,
        sampling_steps: int = 50,
        eta: float = 0.0,
        device: Optional[torch.device] = None,
        initial_noise: Optional[torch.Tensor] = None,
        x0_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        device = device or angles.device
        x = (
            initial_noise.to(device)
            if initial_noise is not None
            else torch.randn(*shape, device=device)
        )
        if x.shape[:2] != angles.shape[:2]:
            raise ValueError("sample shape and geometry token count are inconsistent")

        step_ids = torch.linspace(
            self.timesteps - 1,
            0,
            sampling_steps,
            device=device,
        ).long()
        step_ids = torch.unique_consecutive(step_ids)

        for index, current in enumerate(step_ids):
            t = torch.full((shape[0],), int(current.item()), device=device, dtype=torch.long)
            predicted_noise = model(x, t, angles, gauge_angles)

            alpha_bar_t = self.alpha_bars[current].to(x.dtype)
            if index + 1 < len(step_ids):
                previous = step_ids[index + 1]
                alpha_bar_prev = self.alpha_bars[previous].to(x.dtype)
            else:
                alpha_bar_prev = torch.ones((), device=device, dtype=x.dtype)

            predicted_x0 = (
                x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_bar_t)

            # The Stage-1 decoder was trained only on parameter-free
            # LayerNorm tokens (per token: mean=0, variance=1).  At large t,
            # epsilon-prediction error is amplified by 1/sqrt(alpha_bar_t);
            # without projecting x0 back to the decoder's latent manifold, DDIM
            # can diverge and the decoder receives extreme out-of-distribution
            # tokens.
            if x0_transform is not None:
                predicted_x0 = x0_transform(predicted_x0.float()).to(x.dtype)

            sigma = float(eta) * torch.sqrt(
                ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
                * (1.0 - alpha_bar_t / alpha_bar_prev)
            )
            direction = torch.sqrt(
                (1.0 - alpha_bar_prev - sigma.square()).clamp_min(0.0)
            ) * predicted_noise
            noise = torch.randn_like(x) if float(eta) > 0 and index + 1 < len(step_ids) else 0.0
            x = torch.sqrt(alpha_bar_prev) * predicted_x0 + direction + sigma * noise

        if x0_transform is not None:
            x = x0_transform(x.float()).to(x.dtype)
        return x
