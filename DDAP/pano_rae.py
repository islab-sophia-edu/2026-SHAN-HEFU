"""PanoMAE-based representation autoencoder for latent diffusion.

This module deliberately removes the VAE.  The frozen PanoMAE encoder maps
N tangent RGB views to N high-dimensional representation tokens.  A trainable
full-token decoder maps those tokens back to tangent RGB views.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PanoRepresentationAutoencoder(nn.Module):
    """Wrap an existing ``PanoramicMAE`` as a representation autoencoder.

    Important differences from masked pre-training:
      * no random masking is used;
      * the encoder CLS token is discarded from the diffusion latent;
      * patch tokens are normalized independently across channels;
      * a learnable decoder-only CLS token is prepended before the original
        PanoMAE decoder.

    The latent passed to diffusion has shape ``[B, N, D]``.  For the paper's
    16 x 32 tangent grid, N=512.
    """

    def __init__(
        self,
        pano_mae: nn.Module,
        normalize_latents: bool = True,
    ) -> None:
        super().__init__()
        self.pano_mae = pano_mae
        self.normalize_latents = bool(normalize_latents)

        if not hasattr(pano_mae, "cls_token"):
            raise TypeError("pano_mae must expose cls_token like PanoramicMAE")
        self.latent_dim = int(pano_mae.cls_token.shape[-1])

        # The original MAE decoder expects an encoder-space CLS token.  During
        # generative diffusion we only model patch tokens, so this decoder-only
        # token is learned during Stage 1 decoder adaptation.
        self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        nn.init.normal_(self.decoder_cls_token, std=0.02)

        out_features = int(pano_mae.decoder_pred.out_features)
        in_chans = getattr(pano_mae.view_embed.proj, "in_chans", 3)
        in_chans = int(in_chans)
        patch_area = out_features // in_chans
        patch_size = int(round(math.sqrt(patch_area)))
        if patch_size * patch_size * in_chans != out_features:
            raise ValueError(
                "Cannot infer square tangent patch size from decoder_pred: "
                f"out_features={out_features}, in_chans={in_chans}."
            )
        self.in_chans = in_chans
        self.patch_size = patch_size

    @staticmethod
    def _ensure_gauge(
        gauge_angles: Optional[torch.Tensor],
        angles: torch.Tensor,
    ) -> torch.Tensor:
        if gauge_angles is None:
            return torch.zeros(
                angles.shape[0],
                angles.shape[1],
                1,
                device=angles.device,
                dtype=angles.dtype,
            )
        if gauge_angles.ndim == 2:
            gauge_angles = gauge_angles.unsqueeze(-1)
        return gauge_angles.to(device=angles.device, dtype=angles.dtype)

    @staticmethod
    def token_layer_norm(z: torch.Tensor) -> torch.Tensor:
        """Parameter-free LayerNorm for each token across channels."""
        return F.layer_norm(z, (z.shape[-1],), weight=None, bias=None, eps=1e-6)

    def encode(
        self,
        views: torch.Tensor,
        angles: torch.Tensor,
        gauge_angles: Optional[torch.Tensor] = None,
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        """Encode every tangent view without masking.

        Args:
            views: ``[B, N, 3, S, S]`` ImageNet-normalized tangent views.
            angles: ``[B, N, 2]`` in degrees, ordered ``[longitude, latitude]``.
            gauge_angles: optional ``[B, N, 1]`` in degrees.
            normalize: override ``self.normalize_latents``.

        Returns:
            Patch representation tokens ``[B, N, D]``.  CLS is excluded.
        """
        if views.ndim != 5:
            raise ValueError(f"views must be [B,N,C,H,W], got {tuple(views.shape)}")
        if angles.ndim != 3 or angles.shape[-1] != 2:
            raise ValueError(f"angles must be [B,N,2], got {tuple(angles.shape)}")

        b, n, c, h, w = views.shape
        if n != angles.shape[1]:
            raise ValueError("views and angles have different token counts")

        use_gctt = bool(getattr(self.pano_mae, "use_gctt", False))
        gauge = self._ensure_gauge(gauge_angles, angles) if use_gctt else None

        x = self.pano_mae.view_embed(views.reshape(b * n, c, h, w)).reshape(b, n, -1)
        x = x + self.pano_mae.angle_pos_embed(
            angles,
            gauge_angles=gauge if use_gctt else None,
        )

        cls = self.pano_mae.cls_token.expand(b, -1, -1)
        latent_with_cls = self.pano_mae.forward_encoder(
            torch.cat([cls, x], dim=1),
            visible_angles=angles,
            visible_gauge_angles=gauge,
        )
        z = latent_with_cls[:, 1:, :]

        do_norm = self.normalize_latents if normalize is None else bool(normalize)
        if do_norm:
            z = self.token_layer_norm(z)
        return z

    def decode(
        self,
        z: torch.Tensor,
        angles: torch.Tensor,
        gauge_angles: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode full patch-token latents to tangent RGB views.

        Args:
            z: ``[B, N, D]`` normalized PanoMAE patch tokens.
            angles: ``[B, N, 2]``.
            gauge_angles: optional ``[B, N, 1]``.

        Returns:
            Predicted normalized tangent views ``[B, N, 3, S, S]``.
        """
        if z.ndim != 3 or z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z must be [B,N,{self.latent_dim}], got {tuple(z.shape)}"
            )
        b, n, _ = z.shape
        if angles.shape[:2] != (b, n):
            raise ValueError("z and angles have incompatible shapes")

        use_gctt = bool(getattr(self.pano_mae, "use_gctt", False))
        gauge = self._ensure_gauge(gauge_angles, angles) if use_gctt else None

        decoder_cls = self.decoder_cls_token.expand(b, -1, -1)
        latent_with_cls = torch.cat([decoder_cls, z], dim=1)
        ids_restore = torch.arange(n, device=z.device).unsqueeze(0).expand(b, -1)

        pred_flat = self.pano_mae.forward_decoder(
            latent_with_cls,
            ids_restore,
            angles,
            all_gauge_angles=gauge,
        )
        return pred_flat.reshape(
            b, n, self.in_chans, self.patch_size, self.patch_size
        )

    def forward(
        self,
        views: torch.Tensor,
        angles: torch.Tensor,
        gauge_angles: Optional[torch.Tensor] = None,
        latent_noise_std: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(views, angles, gauge_angles)
        if latent_noise_std > 0:
            z_for_decode = z + float(latent_noise_std) * torch.randn_like(z)
        else:
            z_for_decode = z
        pred = self.decode(z_for_decode, angles, gauge_angles)
        return pred, z

    def freeze_encoder(self) -> None:
        """Freeze every encoder-side parameter and leave decoder trainable."""
        encoder_prefixes = (
            "cls_token",
            "view_embed",
            "angle_pos_embed",
            "enc_ang_bias",
            "blocks",
            "norm",
        )
        for name, parameter in self.pano_mae.named_parameters():
            parameter.requires_grad = not name.startswith(encoder_prefixes)
        # Full-token decoding never inserts MAE mask tokens.
        if hasattr(self.pano_mae, "mask_token"):
            self.pano_mae.mask_token.requires_grad_(False)
        self.decoder_cls_token.requires_grad_(True)

    def freeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def train(self, mode: bool = True):
        """Keep the frozen encoder in eval mode while training the decoder."""
        super().train(mode)
        if mode:
            self.pano_mae.view_embed.eval()
            self.pano_mae.angle_pos_embed.eval()
            self.pano_mae.blocks.eval()
            self.pano_mae.norm.eval()
            if getattr(self.pano_mae, "enc_ang_bias", None) is not None:
                self.pano_mae.enc_ang_bias.eval()
        return self


def latitude_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> torch.Tensor:
    """One reconstruction objective: latitude-area-weighted MSE.

    ``weights`` should have shape ``[B,N]`` and mean approximately one.  Passing
    ``None`` gives ordinary MSE.
    """
    per_token = (pred.float() - target.float()).square().mean(dim=(2, 3, 4))
    if weights is None:
        return per_token.mean()
    weights = weights.to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * weights).sum() / weights.sum().clamp_min(1e-6)


def load_checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("rae", "model", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state = checkpoint[key]
            if isinstance(state, dict):
                return state
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    return checkpoint


def load_pretrained_pano_mae(
    pano_mae: nn.Module,
    checkpoint_path: str,
    strict: bool = True,
) -> Tuple[list, list]:
    """Load a PanoMAE pre-training checkpoint with shape validation."""
    state = load_checkpoint_state(checkpoint_path)

    # A Stage-1 RAE checkpoint prefixes the original model with ``pano_mae.``.
    if any(key.startswith("pano_mae.") for key in state):
        state = {
            key[len("pano_mae."):]: value
            for key, value in state.items()
            if key.startswith("pano_mae.")
        }

    model_state = pano_mae.state_dict()
    mismatched = [
        (key, tuple(value.shape), tuple(model_state[key].shape))
        for key, value in state.items()
        if key in model_state
        and hasattr(value, "shape")
        and tuple(value.shape) != tuple(model_state[key].shape)
    ]
    if mismatched:
        preview = "\n".join(
            f"  {key}: checkpoint {src} != model {dst}"
            for key, src, dst in mismatched[:20]
        )
        raise RuntimeError(
            "PanoMAE checkpoint is incompatible with the constructed model. "
            "The most common cause is a different grid_height/patch_size or "
            f"model size.\n{preview}"
        )

    incompatible = pano_mae.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    critical_missing = [
        key for key in missing
        if key.startswith((
            "view_embed", "angle_pos_embed", "blocks", "norm",
            "decoder_embed", "decoder_blocks", "decoder_norm", "decoder_pred",
        ))
    ]
    if strict and (critical_missing or unexpected):
        raise RuntimeError(
            "Checkpoint loading was not exact.\n"
            f"Critical missing keys: {critical_missing[:30]}\n"
            f"Unexpected keys: {unexpected[:30]}"
        )
    return missing, unexpected
