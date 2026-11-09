"""Small generic observable-feature residual, with ordinary/meta initialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .dynamics import seeded_mlp


class ResidualAdapter(nn.Module):
    """6 -> 16 -> 16 -> 4 tanh adapter, 452 parameters at default widths.

    Output is a correction in the base model's fixed normalized-delta units.
    A zero final layer makes ordinary initialization exactly neutral.
    Meta-training later learns an initialization of this same architecture.
    """

    checkpoint_schema = "adaptive_world_model.cartpole.adapter.v1"

    def __init__(self, seed: int = 11, hidden_sizes: tuple[int, ...] = (16, 16)) -> None:
        super().__init__()
        self.seed = int(seed)
        self.hidden_sizes = tuple(int(width) for width in hidden_sizes)
        if not self.hidden_sizes or any(width <= 0 for width in self.hidden_sizes):
            raise ValueError("Adapter hidden sizes must be positive.")
        self.network = seeded_mlp(self.seed, (6, *self.hidden_sizes, 4), nn.Tanh, zero_output=True)
        self.meta_trained = False
        self.checkpoint_metadata: dict[str, Any] = {}

    def forward(self, normalized_features: torch.Tensor) -> torch.Tensor:
        if normalized_features.shape[-1] != 6:
            raise ValueError("Adapter receives only the six common observable features.")
        return self.network(normalized_features)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        details = dict(self.checkpoint_metadata)
        details.update(metadata or {})
        torch.save({"schema": self.checkpoint_schema, "seed": self.seed,
                    "hidden_sizes": list(self.hidden_sizes), "meta_trained": bool(self.meta_trained),
                    "state": self.state_dict(), "metadata": details}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path,
                        map_location: str | torch.device = "cpu") -> "ResidualAdapter":
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if checkpoint.get("schema") != cls.checkpoint_schema:
            raise ValueError("Unsupported residual adapter checkpoint schema.")
        dtype = next(value.dtype for value in checkpoint["state"].values() if value.is_floating_point())
        result = cls(seed=checkpoint["seed"], hidden_sizes=tuple(checkpoint["hidden_sizes"])).to(
            device=map_location, dtype=dtype)
        result.load_state_dict(checkpoint["state"], strict=True)
        result.meta_trained = bool(checkpoint.get("meta_trained", False))
        result.checkpoint_metadata = checkpoint.get("metadata", {})
        return result
