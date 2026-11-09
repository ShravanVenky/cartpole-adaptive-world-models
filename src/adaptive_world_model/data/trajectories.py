"""Validated held-out trajectory windows for recursive query objectives.

This module adds no new sampler: it reconstructs the original V1 time-major
transition records and preserves their recorded actions and trajectory identity.
"""
from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor

from .transitions import TransitionBatch


def trajectory_windows(batch: TransitionBatch, horizon: int, stride: int | None = None
                       ) -> tuple[Tensor, Tensor, Tensor]:
    """Return initial[W,4], actions[W,H,1], truth[W,H+1,4].

    Default stride H makes nonoverlapping full windows. Incomplete suffixes
    are excluded. A gap, duplicate ID or broken state chain is an error, never
    silently bridged with another trajectory. IID data is not a trajectory.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    stride = horizon if stride is None else stride
    if isinstance(stride, bool) or not isinstance(stride, Integral) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if len(batch) == 0:
        raise ValueError("trajectory batch must be nonempty")
    collector = batch.metadata.get("collector")
    if collector is not None and collector != "sequential":
        raise ValueError("recursive windows require sequential trajectories")
    starts, actions, truths = [], [], []
    for trajectory in torch.unique(batch.trajectory_id, sorted=True).cpu().tolist():
        selected = torch.where(batch.trajectory_id == trajectory)[0]
        times = batch.transition_id[selected]
        order = torch.argsort(times)
        selected, times = selected[order], times[order]
        if bool((times < 0).any()) or (len(times) > 1 and not bool((times[1:] - times[:-1] == 1).all())):
            raise ValueError(f"trajectory {trajectory} has duplicate or discontinuous transition IDs")
        states, following = batch.state[selected], batch.next_state[selected]
        if len(selected) > 1 and not torch.allclose(following[:-1], states[1:], rtol=1e-6, atol=1e-7):
            raise ValueError(f"trajectory {trajectory} has a broken state chain")
        for start in range(0, len(selected) - int(horizon) + 1, int(stride)):
            window = selected[start:start + int(horizon)]
            initial = batch.state[window[0]]
            starts.append(initial)
            actions.append(batch.action[window])
            truths.append(torch.cat((initial.unsqueeze(0), batch.next_state[window]), dim=0))
    if not starts:
        raise ValueError(f"no complete trajectory window reaches horizon {horizon}")
    return torch.stack(starts), torch.stack(actions), torch.stack(truths)
