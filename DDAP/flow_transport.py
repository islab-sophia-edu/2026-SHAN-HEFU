"""Stable dimension-shifted rectified-flow transport for PanoMAE latents.

Clean data is at t=0 and Gaussian noise is at t=1:
    x_t = (1 - t) * x_0 + t * epsilon
    v_t = epsilon - x_0

The default base-time distribution is logit-normal N(0, 1), matching the
released RAE Stage-2 recipe more closely than uniform time sampling.  The base
time is then shifted as a function of the total latent dimension.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FlowBatch:
    x_t: torch.Tensor
    time: torch.Tensor
    base_time: torch.Tensor
    target_velocity: torch.Tensor
    noise: torch.Tensor


class DimensionShiftedRectifiedFlow:
    def __init__(
        self,
        token_count: int,
        token_dim: int,
        base_dimension: int = 4096,
        time_eps: float = 1e-3,
        time_distribution: str = "logit_normal",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
    ) -> None:
        if token_count <= 0 or token_dim <= 0 or base_dimension <= 0:
            raise ValueError("Dimensions must be positive")
        if not 0.0 <= time_eps < 0.5:
            raise ValueError("time_eps must be in [0, 0.5)")
        if time_distribution not in {"uniform", "logit_normal"}:
            raise ValueError("time_distribution must be 'uniform' or 'logit_normal'")
        if logit_normal_std <= 0:
            raise ValueError("logit_normal_std must be positive")

        self.token_count = int(token_count)
        self.token_dim = int(token_dim)
        self.base_dimension = int(base_dimension)
        self.effective_dimension = self.token_count * self.token_dim
        self.shift_alpha = math.sqrt(self.effective_dimension / self.base_dimension)
        self.time_eps = float(time_eps)
        self.time_distribution = str(time_distribution)
        self.logit_normal_mean = float(logit_normal_mean)
        self.logit_normal_std = float(logit_normal_std)

    def shift_time(self, base_time: torch.Tensor) -> torch.Tensor:
        u = base_time.clamp(0.0, 1.0)
        alpha = torch.as_tensor(self.shift_alpha, device=u.device, dtype=u.dtype)
        return alpha * u / (1.0 + (alpha - 1.0) * u)

    def sample_base_time(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.time_distribution == "uniform":
            raw = torch.rand(
                batch_size,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        else:
            logits = torch.randn(
                batch_size,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            logits = logits * self.logit_normal_std + self.logit_normal_mean
            raw = torch.sigmoid(logits)
        return raw.clamp(self.time_eps, 1.0 - self.time_eps)

    def make_training_batch(
        self,
        x0: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        base_time: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> FlowBatch:
        if x0.ndim != 3 or x0.shape[1:] != (self.token_count, self.token_dim):
            raise ValueError(
                f"x0 must be [B,{self.token_count},{self.token_dim}], got {tuple(x0.shape)}"
            )
        batch = x0.shape[0]
        if noise is None:
            noise = torch.randn(
                x0.shape,
                device=x0.device,
                dtype=x0.dtype,
                generator=generator,
            )
        if base_time is None:
            base_time = self.sample_base_time(
                batch,
                device=x0.device,
                dtype=x0.dtype,
                generator=generator,
            )
        else:
            if base_time.ndim != 1 or base_time.shape[0] != batch:
                raise ValueError(f"base_time must be [B], got {tuple(base_time.shape)}")
            base_time = base_time.to(device=x0.device, dtype=x0.dtype)
            base_time = base_time.clamp(self.time_eps, 1.0 - self.time_eps)

        time = self.shift_time(base_time)
        t_view = time.view(batch, 1, 1)
        x_t = (1.0 - t_view) * x0 + t_view * noise
        target_velocity = noise - x0
        return FlowBatch(
            x_t=x_t,
            time=time,
            base_time=base_time,
            target_velocity=target_velocity,
            noise=noise,
        )

    @staticmethod
    def velocity_loss(
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        token_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_token = (
            predicted_velocity.float() - target_velocity.float()
        ).square().mean(dim=-1)
        if token_weights is None:
            return per_token.mean()
        weights = token_weights.to(per_token.device, per_token.dtype)
        return (per_token * weights).sum() / weights.sum().clamp_min(1e-6)

    @torch.no_grad()
    def integrate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        angles: torch.Tensor,
        gauge_angles: Optional[torch.Tensor],
        time_start: float | torch.Tensor,
        time_end: float | torch.Tensor,
        num_steps: int,
        solver: str = "euler",
    ) -> torch.Tensor:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        if x.ndim != 3 or x.shape[1:] != (self.token_count, self.token_dim):
            raise ValueError("x has incompatible latent shape")

        device, dtype = x.device, x.dtype
        start = torch.as_tensor(time_start, device=device, dtype=dtype)
        end = torch.as_tensor(time_end, device=device, dtype=dtype)
        if start.numel() != 1 or end.numel() != 1:
            raise ValueError("time_start/time_end must be scalar")
        time_grid = torch.linspace(
            float(start), float(end), num_steps + 1, device=device, dtype=dtype
        )
        batch = x.shape[0]

        for index in range(num_steps):
            t_now = time_grid[index]
            t_next = time_grid[index + 1]
            dt = t_next - t_now
            t_batch = torch.full((batch,), float(t_now), device=device, dtype=dtype)
            velocity = model(x, t_batch, angles, gauge_angles)
            if not torch.isfinite(velocity).all():
                raise FloatingPointError(
                    f"Non-finite velocity during ODE integration at step {index}"
                )

            if solver == "euler" or index == num_steps - 1:
                x = x + dt * velocity
            else:
                x_euler = x + dt * velocity
                t_next_batch = torch.full(
                    (batch,), float(t_next), device=device, dtype=dtype
                )
                velocity_next = model(x_euler, t_next_batch, angles, gauge_angles)
                if not torch.isfinite(velocity_next).all():
                    raise FloatingPointError(
                        f"Non-finite Heun velocity during ODE integration at step {index}"
                    )
                x = x + 0.5 * dt * (velocity + velocity_next)

            if not torch.isfinite(x).all():
                raise FloatingPointError(
                    f"Non-finite latent during ODE integration at step {index}"
                )
        return x

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Sequence[int],
        angles: torch.Tensor,
        gauge_angles: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        solver: str = "euler",
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(shape) != 3 or tuple(shape[1:]) != (self.token_count, self.token_dim):
            raise ValueError("shape must match [B,N,D] transport dimensions")
        device = angles.device
        x = (
            initial_noise.to(device=device)
            if initial_noise is not None
            else torch.randn(*shape, device=device, dtype=angles.dtype)
        )
        start = self.shift_time(
            torch.tensor(1.0 - self.time_eps, device=device, dtype=x.dtype)
        )
        end = self.shift_time(
            torch.tensor(self.time_eps, device=device, dtype=x.dtype)
        )
        return self.integrate(
            model,
            x,
            angles,
            gauge_angles,
            time_start=start,
            time_end=end,
            num_steps=num_steps,
            solver=solver,
        )

    @torch.no_grad()
    def denoise_from_base_time(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        angles: torch.Tensor,
        gauge_angles: Optional[torch.Tensor],
        base_time: float,
        num_steps: int = 50,
        solver: str = "euler",
    ) -> torch.Tensor:
        if not self.time_eps <= base_time <= 1.0 - self.time_eps:
            raise ValueError(
                f"base_time must lie in [{self.time_eps}, {1.0 - self.time_eps}]"
            )
        start = self.shift_time(
            torch.tensor(base_time, device=x_t.device, dtype=x_t.dtype)
        )
        end = self.shift_time(
            torch.tensor(self.time_eps, device=x_t.device, dtype=x_t.dtype)
        )
        return self.integrate(
            model,
            x_t,
            angles,
            gauge_angles,
            time_start=start,
            time_end=end,
            num_steps=num_steps,
            solver=solver,
        )
