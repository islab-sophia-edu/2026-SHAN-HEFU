"""Dataset-level latent standardization for PanoMAE representation tokens.

The Stage-1 RAE already applies parameter-free LayerNorm independently to every
PanoMAE token.  Stage 2 additionally standardizes each fixed token-position and
channel across the training dataset.  The statistics therefore have shape
[1, N, D], not one global scalar.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass(frozen=True)
class LatentStatsMetadata:
    count: int
    token_count: int
    latent_dim: int
    eps: float
    fingerprint: str


class LatentStandardizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, count: int, eps: float = 1e-6):
        mean = torch.as_tensor(mean, dtype=torch.float32, device="cpu")
        std = torch.as_tensor(std, dtype=torch.float32, device="cpu")
        if mean.ndim == 2:
            mean = mean.unsqueeze(0)
        if std.ndim == 2:
            std = std.unsqueeze(0)
        if mean.ndim != 3 or std.shape != mean.shape or mean.shape[0] != 1:
            raise ValueError(
                "Latent mean/std must both have shape [1,N,D]; "
                f"received mean={tuple(mean.shape)}, std={tuple(std.shape)}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise FloatingPointError("Latent statistics contain NaN/Inf")
        if (std <= 0).any():
            raise ValueError("All latent standard deviations must be positive")
        self.mean_cpu = mean.contiguous()
        self.std_cpu = std.clamp_min(float(eps)).contiguous()
        self.count = int(count)
        self.eps = float(eps)
        self._device_cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def token_count(self) -> int:
        return int(self.mean_cpu.shape[1])

    @property
    def latent_dim(self) -> int:
        return int(self.mean_cpu.shape[2])

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.mean_cpu.numpy().tobytes())
        digest.update(self.std_cpu.numpy().tobytes())
        digest.update(str(self.count).encode("utf-8"))
        return digest.hexdigest()[:16]

    @property
    def metadata(self) -> LatentStatsMetadata:
        return LatentStatsMetadata(
            count=self.count,
            token_count=self.token_count,
            latent_dim=self.latent_dim,
            eps=self.eps,
            fingerprint=self.fingerprint,
        )

    def validate_shape(self, token_count: int, latent_dim: int) -> None:
        expected = (int(token_count), int(latent_dim))
        actual = (self.token_count, self.latent_dim)
        if actual != expected:
            raise RuntimeError(
                f"Latent stats shape {actual} does not match model shape {expected}. "
                "Recompute stats using the exact same PanoMAE/RAE/grid configuration."
            )

    def _on(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = f"{tensor.device}:{tensor.dtype}"
        cached = self._device_cache.get(key)
        if cached is None:
            cached = (
                self.mean_cpu.to(device=tensor.device, dtype=tensor.dtype),
                self.std_cpu.to(device=tensor.device, dtype=tensor.dtype),
            )
            self._device_cache[key] = cached
        return cached

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3 or z.shape[1:] != (self.token_count, self.latent_dim):
            raise ValueError(
                f"z must be [B,{self.token_count},{self.latent_dim}], got {tuple(z.shape)}"
            )
        mean, std = self._on(z)
        return (z - mean) / std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3 or z.shape[1:] != (self.token_count, self.latent_dim):
            raise ValueError(
                f"z must be [B,{self.token_count},{self.latent_dim}], got {tuple(z.shape)}"
            )
        mean, std = self._on(z)
        return z * std + mean

    def checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "mean": self.mean_cpu,
            "std": self.std_cpu,
            "count": self.count,
            "eps": self.eps,
            "fingerprint": self.fingerprint,
            "format": "pano_latent_stats_v1",
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), path)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "LatentStandardizer":
        if not isinstance(payload, dict) or "mean" not in payload or "std" not in payload:
            raise TypeError("Invalid latent stats payload")
        obj = cls(
            mean=payload["mean"],
            std=payload["std"],
            count=int(payload.get("count", 0)),
            eps=float(payload.get("eps", 1e-6)),
        )
        saved_fingerprint = payload.get("fingerprint")
        if saved_fingerprint and saved_fingerprint != obj.fingerprint:
            raise RuntimeError("Latent stats fingerprint mismatch; file may be corrupted")
        return obj

    @classmethod
    def load(cls, path: str | Path) -> "LatentStandardizer":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_payload(payload)


class OnlinePositionChannelStats:
    """Exact dataset moments for fixed-position token latents [B,N,D]."""

    def __init__(self, token_count: int, latent_dim: int):
        self.token_count = int(token_count)
        self.latent_dim = int(latent_dim)
        self.count = 0
        self.sum = torch.zeros(self.token_count, self.latent_dim, dtype=torch.float64)
        self.sum_sq = torch.zeros_like(self.sum)

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        if z.ndim != 3 or z.shape[1:] != (self.token_count, self.latent_dim):
            raise ValueError(
                f"Expected [B,{self.token_count},{self.latent_dim}], got {tuple(z.shape)}"
            )
        z64 = z.detach().to(device="cpu", dtype=torch.float64)
        self.sum += z64.sum(dim=0)
        self.sum_sq += z64.square().sum(dim=0)
        self.count += int(z.shape[0])

    def finalize(self, eps: float = 1e-6) -> LatentStandardizer:
        if self.count < 2:
            raise RuntimeError("At least two samples are required to estimate latent variance")
        mean = self.sum / self.count
        variance = self.sum_sq / self.count - mean.square()
        variance = variance.clamp_min(float(eps) ** 2)
        return LatentStandardizer(
            mean=mean.float().unsqueeze(0),
            std=variance.sqrt().float().unsqueeze(0),
            count=self.count,
            eps=eps,
        )
