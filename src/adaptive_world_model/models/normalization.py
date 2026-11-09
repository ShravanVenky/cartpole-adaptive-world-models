"""Nominal/offline normalization shared by every CartPole model family."""

from __future__ import annotations

from pathlib import Path
import math
from numbers import Real

import torch
from torch import nn


class FeatureNormalizer(nn.Module):
    """Fixed six-observable/four-delta statistics, never refit at adaptation time.

    Fit on the nominal training split only. Angle targets are the difference of
    the simulator's unwrapped angles; neither targets nor reconstruction wrap.
    """

    input_dim = 6
    output_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(self.input_dim))
        self.register_buffer("x_scale", torch.ones(self.input_dim))
        self.register_buffer("y_mean", torch.zeros(self.output_dim))
        self.register_buffer("y_scale", torch.ones(self.output_dim))
        self.register_buffer("fitted", torch.tensor(False))

    @classmethod
    def fit(cls, features: torch.Tensor, deltas: torch.Tensor,
            minimum_scale: float = 1e-4) -> "FeatureNormalizer":
        if features.ndim != 2 or features.shape[1] != cls.input_dim:
            raise ValueError("Normalization features must have shape [N, 6].")
        if deltas.shape != (len(features), cls.output_dim) or len(features) < 2:
            raise ValueError("Normalization deltas must have shape [N, 4], N >= 2.")
        if isinstance(minimum_scale, bool) or not isinstance(minimum_scale, Real) or not math.isfinite(minimum_scale) or minimum_scale <= 0:
            raise ValueError("minimum_scale must be a finite positive number.")
        if not torch.isfinite(features).all() or not torch.isfinite(deltas).all():
            raise ValueError("Cannot fit normalization on NaN/Inf data.")
        # Statistics are buffers, detached even when the caller supplied tensors
        # requiring gradients. They remain fixed through support adaptation.
        result = cls().to(device=features.device, dtype=features.dtype)
        with torch.no_grad():
            result.x_mean.copy_(features.mean(0))
            result.x_scale.copy_(features.std(0, unbiased=False).clamp_min(minimum_scale))
            result.y_mean.copy_(deltas.mean(0))
            result.y_scale.copy_(deltas.std(0, unbiased=False).clamp_min(minimum_scale))
            result.fitted.fill_(True)
        return result

    def encode_x(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError("Expected six common observable features.")
        return (features - self.x_mean) / self.x_scale

    def encode_y(self, deltas: torch.Tensor) -> torch.Tensor:
        if deltas.shape[-1] != self.output_dim:
            raise ValueError("Expected four unwrapped state deltas.")
        return (deltas - self.y_mean) / self.y_scale

    def decode_y(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.output_dim:
            raise ValueError("Expected four normalized state deltas.")
        return values * self.y_scale + self.y_mean

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": "cartpole_normalization_v1", "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "FeatureNormalizer":
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if checkpoint.get("schema") != "cartpole_normalization_v1":
            raise ValueError("Unsupported normalization checkpoint schema.")
        result = cls().to(device=map_location, dtype=checkpoint["state"]["x_mean"].dtype)
        result.load_state_dict(checkpoint["state"], strict=True)
        return result
