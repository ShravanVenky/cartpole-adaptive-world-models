"""Optional shared model-only CEM controller; no diagnostic actions or physics."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

import torch
from torch import Tensor

from ..environments import wrap_angle


class Controller(Protocol):
    def action(self, model: Any, state: Tensor, seed: int = 0) -> Tensor: ...


@dataclass
class CEMController:
    horizon: int = 20
    candidates: int = 128
    elites: int = 16
    iterations: int = 3
    force_limit: float = 10.0

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.horizon, self.candidates, self.elites, self.iterations)):
            raise ValueError("CEM dimensions must be integers")
        if min(self.horizon, self.candidates, self.elites, self.iterations) < 1 or self.elites > self.candidates:
            raise ValueError("CEM dimensions must be positive with elites <= candidates")
        if not math.isfinite(self.force_limit) or self.force_limit <= 0:
            raise ValueError("force_limit must be positive")

    @staticmethod
    def cost(states: Tensor, actions: Tensor) -> Tensor:
        # Generic upright balancing task; no parameters of the environment.
        moving = states[..., 1:, :]
        stage = (moving[..., 0].square() + 0.05 * moving[..., 1].square()
                 + 5.0 * wrap_angle(moving[..., 2]).square()
                 + 0.1 * moving[..., 3].square() + 0.001 * actions.squeeze(-1).square())
        return stage.sum(-1)

    @torch.no_grad()
    def action(self, model: Any, state: Tensor, seed: int = 0) -> Tensor:
        if state.ndim not in (1, 2) or state.shape[-1] != 4 or not bool(torch.isfinite(state).all()):
            raise ValueError("controller state must be finite with shape (4,) or (B,4)")
        squeeze = state.ndim == 1
        current = state.unsqueeze(0) if squeeze else state
        batch = len(current)
        generator = torch.Generator(device=state.device).manual_seed(int(seed))
        mean = torch.zeros((batch, self.horizon, 1), device=state.device, dtype=state.dtype)
        scale = torch.full_like(mean, self.force_limit * 0.5)
        initial = current[:, None, :].expand(batch, self.candidates, 4)
        for _ in range(self.iterations):
            noise = torch.randn((batch, self.candidates, self.horizon, 1), generator=generator,
                                device=state.device, dtype=state.dtype)
            sequences = (mean[:, None] + scale[:, None] * noise).clamp(-self.force_limit, self.force_limit)
            sequences[:, 0] = mean
            states = model.rollout(initial, sequences)
            costs = self.cost(states, sequences)
            if not bool(torch.isfinite(costs).all()):
                raise FloatingPointError("controller model rollout produced nonfinite costs")
            best = costs.topk(self.elites, dim=1, largest=False).indices
            elite = sequences.gather(1, best[:, :, None, None].expand(batch, self.elites, self.horizon, 1))
            mean = elite.mean(1)
            scale = elite.std(1, unbiased=False).clamp_min(0.01 * self.force_limit)
        result = mean[:, 0]
        return result.squeeze(0) if squeeze else result
