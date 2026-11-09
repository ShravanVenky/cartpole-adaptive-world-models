"""Deterministic passive sequential data and an explicit scattered reference."""
from __future__ import annotations

import hashlib
from dataclasses import asdict

import torch

from ..environments import CartPolePhysics, sample_initial_state, step
from .transitions import TransitionBatch


def _stream_seed(seed: int, *parts: str) -> int:
    """Stable stream namespaces, independent of Python's randomized hash."""
    payload = "|".join((str(int(seed)), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _validate_request(length: int, batch_size: int, stream: str) -> None:
    if not isinstance(length, int) or length < 1:
        raise ValueError("length must be a positive integer")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a nonempty provenance namespace")


def collect_sequential(physics: CartPolePhysics, length: int, seed: int,
                       regime: str = "perturbed_balance", device: str | torch.device = "cpu",
                       batch_size: int = 1, stream: str = "support") -> TransitionBatch:
    """Collect uninterrupted trajectories using bounded generic exploration.

    Outputs are time-major flat (length*batch_size, ...). No mismatch detector,
    parameter-aware controller, active diagnosis, state wrapping or resets are
    used. A slow colored-force component continually excites the dynamics.
    """
    _validate_request(length, batch_size, stream)
    target_device = torch.device(device)
    generator = torch.Generator(device=target_device).manual_seed(_stream_seed(seed, "sequential", stream, regime))
    state = sample_initial_state(batch_size, generator, regime, target_device)
    colored = torch.zeros(batch_size, device=target_device)
    states, actions, next_states = [], [], []
    for _ in range(length):
        noise = torch.rand(batch_size, generator=generator, device=target_device) * 2.0 - 1.0
        colored = 0.8 * colored + 0.2 * noise
        # A deliberately ordinary, imperfect balancing feedback policy; the
        # same observable-only gains are used in every physical environment.
        if regime == "perturbed_balance":
            feedback = 9.0 * state[:, 2].sin() + 1.5 * state[:, 3] - 0.5 * state[:, 0] - 0.4 * state[:, 1]
            command = feedback + 0.55 * physics.force_limit * colored + 0.15 * physics.force_limit * noise
        else:
            command = physics.force_limit * (0.85 * colored + 0.15 * noise)
        command = command.clamp(-physics.force_limit, physics.force_limit)
        following = step(state, command, physics)
        states.append(state)
        actions.append(command.unsqueeze(-1))
        next_states.append(following)
        state = following
    ids = torch.arange(batch_size, device=target_device).repeat(length)
    time = torch.arange(length, device=target_device).repeat_interleave(batch_size)
    return TransitionBatch(
        torch.cat(states), torch.cat(actions), torch.cat(next_states), ids, time,
        provenance=f"sequential/seed={int(seed)}/stream={stream}/regime={regime}",
        metadata={"collector": "sequential", "regime": regime, "stream": stream,
                  "seed": int(seed), "batch_size": batch_size, "length": length,
                  "physics": asdict(physics), "policy": "bounded_feedback_colored_exploration" if regime == "perturbed_balance" else "bounded_colored_exploration"},
    )


def collect_iid(physics: CartPolePhysics, length: int, seed: int,
                regime: str = "perturbed_balance", device: str | torch.device = "cpu",
                batch_size: int = 1, stream: str = "iid_support") -> TransitionBatch:
    """Scattered one-step transitions: an explicit reference, not sequential support."""
    _validate_request(length, batch_size, stream)
    target_device = torch.device(device)
    count = length * batch_size
    generator = torch.Generator(device=target_device).manual_seed(_stream_seed(seed, "iid", stream, regime))
    state = sample_initial_state(count, generator, regime, target_device)
    action = (torch.rand((count, 1), generator=generator, device=target_device) * 2.0 - 1.0) * physics.force_limit
    following = step(state, action, physics)
    return TransitionBatch(
        state, action, following, torch.arange(count, device=target_device),
        torch.zeros(count, dtype=torch.int64, device=target_device),
        provenance=f"iid/seed={int(seed)}/stream={stream}/regime={regime}",
        metadata={"collector": "iid", "regime": regime, "stream": stream, "seed": int(seed),
                  "batch_size": batch_size, "length": length, "physics": asdict(physics)},
    )
