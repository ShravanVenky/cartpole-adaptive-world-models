"""Transition containers with explicit held-out trajectory provenance."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor

from ..environments import observable_features


@dataclass
class TransitionBatch:
    state: Tensor
    action: Tensor
    next_state: Tensor
    trajectory_id: Tensor
    transition_id: Tensor
    provenance: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state.ndim != 2 or self.state.shape[-1] != 4:
            raise ValueError("transition states must have flat shape (N, 4)")
        if self.next_state.shape != self.state.shape:
            raise ValueError("state and next_state shapes differ")
        count = len(self.state)
        if self.action.shape == (count,):
            self.action = self.action.unsqueeze(-1)
        if self.action.shape != (count, 1):
            raise ValueError("actions must have shape (N, 1)")
        tensors = (self.state, self.action, self.next_state)
        if not all(t.is_floating_point() for t in tensors):
            raise TypeError("state, action, next_state must be floating point tensors")
        if len({(t.device, t.dtype) for t in tensors}) != 1:
            raise ValueError("transition tensors must share device and dtype")
        if not all(bool(torch.isfinite(t).all()) for t in tensors):
            raise ValueError("transition contains NaN or Inf")
        for name in ("trajectory_id", "transition_id"):
            ids = getattr(self, name)
            if ids.shape != (count,) or ids.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must have integer shape (N,)")
        if not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError("nonempty provenance is required")
        identities = list(zip(self.trajectory_id.cpu().tolist(), self.transition_id.cpu().tolist()))
        if len(set(identities)) != count:
            raise ValueError("duplicate transition identities within batch")

    @property
    def states(self) -> Tensor:
        return self.state

    @property
    def actions(self) -> Tensor:
        return self.action

    @property
    def next_states(self) -> Tensor:
        return self.next_state

    def __len__(self) -> int:
        return len(self.state)

    def features(self) -> Tensor:
        return observable_features(self.state, self.action)

    def delta(self) -> Tensor:
        # Physical theta is unwrapped; do not train on a wrapped discontinuity.
        return self.next_state - self.state

    def select(self, indices: Tensor | slice | list[int]) -> "TransitionBatch":
        return TransitionBatch(
            self.state[indices], self.action[indices], self.next_state[indices],
            self.trajectory_id[indices], self.transition_id[indices],
            self.provenance, dict(self.metadata),
        )

    def prefix(self, count: int) -> "TransitionBatch":
        """Take count total transitions; a batch_size=1 collector gives K-shot support."""
        if not isinstance(count, int) or count < 1 or count > len(self):
            raise ValueError("prefix count must be an integer in [1, len(batch)]")
        return self.select(slice(0, count))

    def to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> "TransitionBatch":
        target_dtype = self.state.dtype if dtype is None else dtype
        return TransitionBatch(
            self.state.to(device=device, dtype=target_dtype),
            self.action.to(device=device, dtype=target_dtype),
            self.next_state.to(device=device, dtype=target_dtype),
            self.trajectory_id.to(device), self.transition_id.to(device),
            self.provenance, dict(self.metadata),
        )

    def support_hash(self) -> str:
        """Hash exact observations and provenance to verify matched method support."""
        digest = hashlib.sha256(self.provenance.encode("utf-8"))
        for tensor in (self.state, self.action, self.next_state, self.trajectory_id, self.transition_id):
            values = tensor.detach().cpu().contiguous()
            digest.update(str((tuple(values.shape), str(values.dtype))).encode("ascii"))
            digest.update(values.numpy().tobytes())
        return digest.hexdigest()

    def assert_disjoint(self, other: "TransitionBatch") -> None:
        """Require different held-out trajectories, not merely distinct time indices."""
        if self.provenance != other.provenance:
            return
        overlap = set(self.trajectory_id.cpu().tolist()) & set(other.trajectory_id.cpu().tolist())
        if overlap:
            raise ValueError(f"support/query share trajectory identities in {self.provenance!r}: {sorted(overlap)[:5]}")


def assert_disjoint(support: TransitionBatch, query: TransitionBatch) -> None:
    support.assert_disjoint(query)
