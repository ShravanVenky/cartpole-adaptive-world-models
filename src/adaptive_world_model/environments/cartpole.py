"""Continuous CartPole with a uniform rod and unwrapped physical angles.

The simulator is ground truth, never a component of a learned predictor. The
two coupled manipulator equations are solved directly and integrated with RK4.
Theta zero is upright; ``pole_half_length`` is half the full rod length.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass(frozen=True)
class CartPolePhysics:
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_half_length: float = 0.5
    motor_gain: float = 1.0
    cart_friction: float = 0.0
    pole_friction: float = 0.0
    g: float = 9.81
    dt: float = 0.02
    force_limit: float = 10.0
    rk4_substeps: int = 2

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
        for name in ("cart_mass", "pole_mass", "pole_half_length", "motor_gain", "g", "dt", "force_limit"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.cart_friction < 0 or self.pole_friction < 0:
            raise ValueError("friction coefficients must be nonnegative")
        if isinstance(self.rk4_substeps, bool) or int(self.rk4_substeps) != self.rk4_substeps or self.rk4_substeps < 1:
            raise ValueError("rk4_substeps must be a positive integer")


NOMINAL_PHYSICS = CartPolePhysics()


def _validate_state(state: Tensor) -> None:
    if not isinstance(state, Tensor) or not state.is_floating_point():
        raise TypeError("state must be a floating point torch.Tensor")
    if state.ndim < 1 or state.shape[-1] != 4:
        raise ValueError("state must have shape (..., 4)")
    if not bool(torch.isfinite(state).all()):
        raise ValueError("state contains NaN or Inf")


def action_tensor(state: Tensor, action: Tensor | float) -> Tensor:
    """Return force shaped like state[..., 0], accepting scalar or trailing 1."""
    force = torch.as_tensor(action, dtype=state.dtype, device=state.device)
    shape = state.shape[:-1]
    if force.shape == shape + (1,):
        force = force.squeeze(-1)
    elif force.shape == (1,) and not shape:
        force = force.squeeze(0)
    try:
        force = torch.broadcast_to(force, shape)
    except RuntimeError as exc:
        raise ValueError(f"action shape {tuple(force.shape)} cannot match state shape {tuple(state.shape)}") from exc
    if not bool(torch.isfinite(force).all()):
        raise ValueError("action contains NaN or Inf")
    return force


def wrap_angle(angle: Tensor) -> Tensor:
    """Wrap an angle for errors/costs only; physical states stay unwrapped."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def angle_difference(predicted: Tensor, target: Tensor) -> Tensor:
    return wrap_angle(predicted - target)


def observable_features(state: Tensor, action: Tensor | float) -> Tensor:
    """Generic observable inputs, with no masses, lengths, gain or friction."""
    _validate_state(state)
    force = action_tensor(state, action)
    x, velocity, theta, omega = state.unbind(-1)
    return torch.stack((x, velocity, theta.sin(), theta.cos(), omega, force), dim=-1)


def _derivative(state: Tensor, applied_force: Tensor, physics: CartPolePhysics) -> Tensor:
    _, velocity, theta, omega = state.unbind(-1)
    sine, cosine = theta.sin(), theta.cos()
    total_mass = physics.cart_mass + physics.pole_mass
    pole_mass_length = physics.pole_mass * physics.pole_half_length
    # Hinge friction is a viscous torque, in N m per rad/s. Cart friction is
    # a viscous horizontal force, in N per m/s. No hidden state clamping.
    horizontal = (applied_force - physics.cart_friction * velocity
                  + pole_mass_length * omega.square() * sine) / total_mass
    angular_acceleration = (
        physics.g * sine - cosine * horizontal
        - physics.pole_friction * omega / pole_mass_length
    ) / (physics.pole_half_length * (4.0 / 3.0 - physics.pole_mass * cosine.square() / total_mass))
    acceleration = horizontal - pole_mass_length * angular_acceleration * cosine / total_mass
    return torch.stack((velocity, acceleration, omega, angular_acceleration), dim=-1)


def step(state: Tensor, action: Tensor | float, physics: CartPolePhysics = NOMINAL_PHYSICS) -> Tensor:
    """Advance one dt under constant commanded force, on CPU or CUDA.

    The command is clipped before motor gain. RK4 does not wrap theta or clip
    velocities. Nonfinite input/output is an explicit error, not a fallback.
    """
    _validate_state(state)
    if not isinstance(physics, CartPolePhysics):
        raise TypeError("physics must be CartPolePhysics")
    force = action_tensor(state, action).clamp(-physics.force_limit, physics.force_limit) * physics.motor_gain
    duration = physics.dt / physics.rk4_substeps
    result = state
    for _ in range(int(physics.rk4_substeps)):
        k1 = _derivative(result, force, physics)
        k2 = _derivative(result + 0.5 * duration * k1, force, physics)
        k3 = _derivative(result + 0.5 * duration * k2, force, physics)
        k4 = _derivative(result + duration * k3, force, physics)
        result = result + (duration / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("CartPole integration produced NaN or Inf")
    return result


def sample_initial_state(batch_size: int, generator: torch.Generator, regime: str,
                         device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> Tensor:
    """Sample visibly moving starts; this is not equilibrium-only evaluation."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    draw = torch.rand((batch_size, 4), generator=generator, device=device, dtype=dtype)
    if regime == "perturbed_balance":
        state = (draw * 2.0 - 1.0) * torch.tensor((0.35, 0.7, 0.2, 0.8), device=device, dtype=dtype)
        # Every trajectory starts noticeably tilted: |theta| in [0.15, 0.35].
        sign = torch.where(draw[:, 2] < 0.5, -torch.ones_like(draw[:, 2]), torch.ones_like(draw[:, 2]))
        state[:, 2] = sign * (0.15 + 0.2 * ((draw[:, 2] * 2.0 - 1.0).abs()))
        return state
    if regime in ("broad", "swing_up"):
        return (draw * 2.0 - 1.0) * torch.tensor((1.0, 2.0, math.pi, 3.0), device=device, dtype=dtype)
    raise ValueError(f"unknown data regime: {regime!r}")


class CartPoleEnv:
    """Small batched simulator, without Gym termination or auto-reset behavior."""

    def __init__(self, physics: CartPolePhysics = NOMINAL_PHYSICS, batch_size: int = 1,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.physics, self.batch_size = physics, batch_size
        self.device, self.dtype = torch.device(device), dtype
        self.state: Tensor | None = None

    def reset(self, seed: int = 0, state: Tensor | None = None,
              regime: str = "perturbed_balance") -> Tensor:
        if state is None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            initial = sample_initial_state(self.batch_size, generator, regime, self.device, self.dtype)
        else:
            initial = torch.as_tensor(state, device=self.device, dtype=self.dtype)
            if initial.shape == (4,) and self.batch_size == 1:
                initial = initial.unsqueeze(0)
            if initial.shape != (self.batch_size, 4):
                raise ValueError(f"reset state must have shape ({self.batch_size}, 4)")
            _validate_state(initial)
        self.state = initial.clone()
        return self.state.clone()

    def step(self, action: Tensor | float) -> Tensor:
        if self.state is None:
            raise RuntimeError("call reset before step")
        self.state = step(self.state, action, self.physics)
        return self.state.clone()
