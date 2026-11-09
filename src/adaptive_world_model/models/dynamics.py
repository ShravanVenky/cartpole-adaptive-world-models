"""Compact nominal CartPole dynamics model and common predictive interface."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from adaptive_world_model.environments.cartpole import observable_features
from adaptive_world_model.utils.freezing import assert_frozen as verify_frozen_weights, freeze_snapshot
from .normalization import FeatureNormalizer


class PredictionSafetyError(RuntimeError):
    """Explicit failure for a nonfinite or implausibly large model prediction."""


class PredictiveModel(Protocol):
    def predict_delta(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...
    def predict_next(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...
    def rollout(self, initial_state: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor: ...
    def reset_adaptation(self) -> None: ...
    def adapt(self, support_data: Any) -> dict[str, Any]: ...


class RolloutMixin:
    """Open-loop rollout, with no true-state feedback after the initial state."""

    def predict_next(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return state + self.predict_delta(state, action)

    def rollout(self, initial_state: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        if initial_state.shape[-1] != 4 or action_sequence.ndim < 2 or action_sequence.shape[-1] != 1:
            raise ValueError("Rollout expects initial_state[...,4] and actions[...,T,1].")
        batch_shape = torch.broadcast_shapes(initial_state.shape[:-1], action_sequence.shape[:-2])
        current = initial_state.expand(*batch_shape, 4)
        actions = action_sequence.expand(*batch_shape, action_sequence.shape[-2], 1)
        states = [current]
        for timestep in range(actions.shape[-2]):
            current = self.predict_next(current, actions[..., timestep, :])
            states.append(current)
        return torch.stack(states, dim=-2)


def seeded_mlp(seed: int, widths: tuple[int, ...], activation: type[nn.Module],
               zero_output: bool = False) -> nn.Sequential:
    """Seeded initialization without perturbing the caller's global RNG stream."""
    with torch.random.fork_rng(devices=[]):
        # Touch only the CPU generator: torch.manual_seed also resets CUDA RNGs
        # even when creating a CPU module. CUDA streams must stay independent.
        torch.random.default_generator.manual_seed(int(seed))
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            layer = nn.Linear(left, right, device="cpu")
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if index < len(widths) - 2:
                layers.append(activation())
        if zero_output:
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)


class FrozenDynamicsModel(nn.Module, RolloutMixin):
    """6 -> 256 -> 256 -> 4 SiLU nominal normalized-delta network.

    The name describes deployment. Parameters are trainable until ``freeze`` is
    called so the future nominal trainer can optimize the same implementation.
    """

    checkpoint_schema = "adaptive_world_model.cartpole.frozen.v1"

    def __init__(self, normalizer: FeatureNormalizer | None = None, seed: int = 11,
                 hidden_sizes: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.seed = int(seed)
        self.hidden_sizes = tuple(int(width) for width in hidden_sizes)
        if not self.hidden_sizes or any(width <= 0 for width in self.hidden_sizes):
            raise ValueError("Hidden sizes must be positive.")
        self.network = seeded_mlp(self.seed, (6, *self.hidden_sizes, 4), nn.SiLU)
        self.normalizer = normalizer if normalizer is not None else FeatureNormalizer()
        self.trained = False
        self.checkpoint_metadata: dict[str, Any] = {}
        self._frozen_snapshot: OrderedDict[str, torch.Tensor] | None = None

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.network.parameters())

    def normalized_delta(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(self.normalizer.encode_x(features))

    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None,
                normalize_output: bool = False) -> torch.Tensor:
        features = state if action is None else observable_features(state, action)
        values = self.normalized_delta(features)
        return values if normalize_output else self.normalizer.decode_y(values)

    def predict_delta(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        values = self.forward(state, action)
        if not torch.isfinite(values).all():
            raise PredictionSafetyError("Frozen dynamics produced a NaN/Inf delta.")
        return values

    def reset_adaptation(self) -> None:
        return None

    def adapt(self, support_data: Any) -> dict[str, Any]:
        self.assert_frozen()
        return {"method": "frozen", "accepted": True, "support_count": len(support_data.state),
                "gradient_steps": 0, "adaptation_seconds": 0.0, "parameter_update_norm": 0.0}

    def freeze(self) -> OrderedDict[str, torch.Tensor]:
        if self._frozen_snapshot is not None:
            self.assert_frozen()
            return self._frozen_snapshot
        self.eval()
        self._frozen_snapshot = OrderedDict(freeze_snapshot(self))
        return self._frozen_snapshot

    def assert_frozen(self, snapshot: dict[str, torch.Tensor] | None = None) -> None:
        expected = snapshot if snapshot is not None else self._frozen_snapshot
        if expected is None:
            raise AssertionError("Dynamics model has not been frozen.")
        verify_frozen_weights(self, expected)

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        details = dict(self.checkpoint_metadata)
        details.update(metadata or {})
        torch.save({"schema": self.checkpoint_schema, "seed": self.seed,
                    "hidden_sizes": list(self.hidden_sizes), "trained": bool(self.trained),
                    "state": self.state_dict(), "metadata": details}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu",
                        freeze: bool = True) -> "FrozenDynamicsModel":
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if checkpoint.get("schema") != cls.checkpoint_schema:
            raise ValueError("Unsupported frozen dynamics checkpoint schema.")
        model = cls(seed=checkpoint["seed"], hidden_sizes=tuple(checkpoint["hidden_sizes"]))
        dtype = next(value.dtype for value in checkpoint["state"].values() if value.is_floating_point())
        model.to(device=map_location, dtype=dtype)
        model.load_state_dict(checkpoint["state"], strict=True)
        model.trained = bool(checkpoint.get("trained", False))
        model.checkpoint_metadata = checkpoint.get("metadata", {})
        if freeze:
            model.freeze()
        return model
