"""Minimal internal low-rank plasticity for the immutable V1 dynamics MLP.

This module does not patch, merge, or copy a new trainable base network. A
LoRAAdapter owns only A/B factors; its forward explicitly receives the original
frozen model. LoRALinear is a standalone generic mathematical layer for tests
and reuse, with frozen weight/bias buffers and the same factor initialization.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .dynamics import FrozenDynamicsModel


def _validate_rank_alpha(rank: int, alpha: float | None) -> tuple[int, float]:
    if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 1:
        raise ValueError("LoRA rank must be a positive integer.")
    value = float(rank) if alpha is None else alpha
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("LoRA alpha must be a finite positive number.")
    return int(rank), float(value)


class LoRAFactors(nn.Module):
    """Only A[r,in] and B[out,r] are trainable; there is no adaptive bias."""

    def __init__(self, in_features: int, out_features: int, rank: int = 1,
                 alpha: float | None = None, seed: int = 11,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.rank, self.alpha = _validate_rank_alpha(rank, alpha)
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
               for value in (in_features, out_features)):
            raise ValueError("LoRA input/output dimensions must be positive integers.")
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.scale = self.alpha / self.rank
        self.seed = int(seed)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        left = torch.empty(self.rank, self.in_features, device="cpu", dtype=dtype)
        # Standard fan-in initialization for A, exactly zero B. Explicit local
        # RNG means architecture construction does not alter experiment streams.
        nn.init.kaiming_uniform_(left, a=math.sqrt(5), generator=generator)
        self.A = nn.Parameter(left)
        self.B = nn.Parameter(torch.zeros(self.out_features, self.rank, dtype=dtype, device="cpu"))

    @property
    def parameter_count(self) -> int:
        return self.rank * (self.in_features + self.out_features)

    def forward(self, values: torch.Tensor, A: torch.Tensor | None = None,
                B: torch.Tensor | None = None) -> torch.Tensor:
        return self.scale * F.linear(F.linear(values, self.A if A is None else A),
                                     self.B if B is None else B)

    def weight_update(self) -> torch.Tensor:
        return self.scale * self.B @ self.A


class LoRALinear(nn.Module):
    """Frozen W x + b plus scale B A x; no merging of the two weight sets."""

    def __init__(self, frozen_layer: nn.Linear, rank: int = 1, alpha: float | None = None,
                 seed: int = 11) -> None:
        super().__init__()
        if not isinstance(frozen_layer, nn.Linear):
            raise TypeError("LoRALinear requires a torch.nn.Linear reference layer.")
        self.register_buffer("weight", frozen_layer.weight.detach().clone())
        self.register_buffer("bias", None if frozen_layer.bias is None else frozen_layer.bias.detach().clone())
        self.factors = LoRAFactors(frozen_layer.in_features, frozen_layer.out_features,
                                  rank, alpha, seed, frozen_layer.weight.dtype).to(frozen_layer.weight)

    @property
    def A(self) -> nn.Parameter:
        return self.factors.A

    @property
    def B(self) -> nn.Parameter:
        return self.factors.B

    @property
    def rank(self) -> int:
        return self.factors.rank

    @property
    def scale(self) -> float:
        return self.factors.scale

    @property
    def parameter_count(self) -> int:
        return self.factors.parameter_count

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, self.weight, self.bias) + self.factors(values)


class LoRAAdapter(nn.Module):
    """Internal factors attached by layer index, leaving the V1 model untouched.

    The default 6->256 and 256->256 hidden layers have 774*r parameters:
    rank 1 = 774; rank 2 = 1548. Final layer index 4 is an explicit optional
    placement ablation. No physical parameters or adaptive biases are used.
    """

    checkpoint_schema = "adaptive_world_model.cartpole.lora_adapter.v1"

    def __init__(self, base: FrozenDynamicsModel, rank: int = 1,
                 layer_indices: tuple[int, ...] = (0, 2), alpha: float | None = None,
                 seed: int = 11) -> None:
        super().__init__()
        base.freeze()
        self.rank, self.alpha = _validate_rank_alpha(rank, alpha)
        self.seed = int(seed)
        self.layer_indices = tuple(layer_indices)
        if (not self.layer_indices or len(set(self.layer_indices)) != len(self.layer_indices)
                or any(isinstance(index, bool) or not isinstance(index, Integral)
                       or index < 0 or index >= len(base.network) for index in self.layer_indices)):
            raise ValueError("LoRA placements must be distinct valid integer layer indices.")
        if any(not isinstance(base.network[index], nn.Linear) for index in self.layer_indices):
            raise ValueError("LoRA may be placed only on Linear layers.")
        self.layer_indices = tuple(int(index) for index in self.layer_indices)
        self.base_hidden_sizes = tuple(base.hidden_sizes)
        self.base_linear_shapes = tuple((index, layer.in_features, layer.out_features)
                                        for index, layer in enumerate(base.network) if isinstance(layer, nn.Linear))
        self.factors = nn.ModuleDict()
        for index in self.layer_indices:
            layer = base.network[index]
            self.factors[str(index)] = LoRAFactors(layer.in_features, layer.out_features,
                                                   self.rank, self.alpha,
                                                   self.seed + 104729 * index,
                                                   layer.weight.dtype).to(layer.weight)
        self.meta_trained = False
        self.checkpoint_metadata: dict[str, Any] = {}
        base.assert_frozen()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_base(self, base: FrozenDynamicsModel) -> None:
        if any(parameter.requires_grad for parameter in base.parameters()):
            raise AssertionError("LoRA requires frozen base parameters; only A/B may receive gradients.")
        shapes = tuple((index, layer.in_features, layer.out_features)
                       for index, layer in enumerate(base.network) if isinstance(layer, nn.Linear))
        if shapes != self.base_linear_shapes or tuple(base.hidden_sizes) != self.base_hidden_sizes:
            raise ValueError("LoRA architecture does not match the supplied frozen base.")

    def normalized_delta(self, base: FrozenDynamicsModel, normalized_features: torch.Tensor,
                         params: Mapping[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Raw combined normalized delta with full input/LoRA gradient flow.

        The caller applies the shared safety bound to the net correction relative
        to ``base.network(xn)``. Base layers are frozen but their operations stay
        in the graph: downstream fixed W must transmit gradients to upstream A/B.
        """
        self._validate_base(base)
        if normalized_features.shape[-1] != 6:
            raise ValueError("LoRA receives the same six normalized observable features as V1.")
        if params is not None and set(params) != set(dict(self.named_parameters())):
            raise ValueError("Functional LoRA parameter names must match the A/B factor set exactly.")
        current = normalized_features
        for index, layer in enumerate(base.network):
            if str(index) in self.factors:
                factors = self.factors[str(index)]
                A = None if params is None else params[f"factors.{index}.A"]
                B = None if params is None else params[f"factors.{index}.B"]
                current = F.linear(current, layer.weight, layer.bias) + factors(current, A, B)
            else:
                current = layer(current)
        return current

    def forward(self, base: FrozenDynamicsModel, normalized_features: torch.Tensor) -> torch.Tensor:
        return self.normalized_delta(base, normalized_features)

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        details = dict(self.checkpoint_metadata)
        details.update(metadata or {})
        torch.save({"schema": self.checkpoint_schema, "rank": self.rank, "alpha": self.alpha,
                    "layer_indices": list(self.layer_indices), "seed": self.seed,
                    "base_hidden_sizes": list(self.base_hidden_sizes),
                    "base_linear_shapes": [list(shape) for shape in self.base_linear_shapes],
                    "meta_trained": bool(self.meta_trained), "state": self.state_dict(),
                    "metadata": details}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, base: FrozenDynamicsModel,
                        map_location: str | torch.device = "cpu") -> "LoRAAdapter":
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if checkpoint.get("schema") != cls.checkpoint_schema:
            raise ValueError("Unsupported LoRA adapter checkpoint schema.")
        if (tuple(checkpoint["base_hidden_sizes"]) != tuple(base.hidden_sizes)
                or tuple(tuple(shape) for shape in checkpoint["base_linear_shapes"]) != tuple(
                    (index, layer.in_features, layer.out_features)
                    for index, layer in enumerate(base.network) if isinstance(layer, nn.Linear))):
            raise ValueError("LoRA checkpoint architecture is incompatible with the frozen base.")
        result = cls(base, rank=checkpoint["rank"], layer_indices=tuple(checkpoint["layer_indices"]),
                     alpha=checkpoint["alpha"], seed=checkpoint["seed"])
        result.load_state_dict(checkpoint["state"], strict=True)
        result.meta_trained = bool(checkpoint.get("meta_trained", False))
        result.checkpoint_metadata = checkpoint.get("metadata", {})
        return result
