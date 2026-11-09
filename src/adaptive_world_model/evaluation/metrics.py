"""Held-out prediction metrics, using nominal scales and wrapped angle errors."""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor

from ..data import TransitionBatch
from ..environments import angle_difference
from ..models.dynamics import PredictionSafetyError


STATE_SCALE = (2.4, 3.0, math.pi, 4.0)
STATE_COMPONENTS = ("x", "xdot", "theta", "thetadot")


def _error(prediction: Tensor, truth: Tensor) -> Tensor:
    error = prediction - truth
    error = error.clone()
    error[..., 2] = angle_difference(prediction[..., 2], truth[..., 2])
    return error


def _failure(reason: str) -> dict[str, Any]:
    return {"nmse": None, "component_nmse": [None] * 4,
            "component_rmse": [None] * 4, "finite": False, "failure": reason}


def _scaled_metrics(error: Tensor, scale: Tensor) -> dict[str, Any]:
    if not bool(torch.isfinite(error).all()):
        return _failure("prediction contains NaN or Inf")
    squared = error.square().reshape(-1, 4).mean(0)
    if not bool(torch.isfinite(squared).all()):
        return _failure("squared prediction error overflowed")
    normalized = squared / scale.square()
    if not bool(torch.isfinite(normalized).all()):
        return _failure("normalized prediction error overflowed")
    return {"nmse": float(normalized.mean()), "component_nmse": normalized.cpu().tolist(),
            "component_rmse": squared.sqrt().cpu().tolist(), "finite": True, "failure": None}


@torch.no_grad()
def one_step_metrics(model: Any, data: TransitionBatch) -> dict[str, Any]:
    """NMSE divides physical errors by frozen nominal delta standard deviations.

    The nominal mean cancels when comparing prediction and truth; normalizers
    are never fitted on support or query data. Component RMSE is physical units.
    """
    if not len(data):
        raise ValueError("evaluation data must be nonempty")
    normalizer = getattr(model, "normalizer", None)
    if normalizer is None:
        normalizer = getattr(getattr(model, "base", None), "normalizer", None)
    if normalizer is None:
        raise ValueError("model must expose nominal normalization via normalizer or base.normalizer")
    scale = normalizer.y_scale.to(device=data.state.device, dtype=data.state.dtype)
    if scale.shape != (4,) or not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("nominal delta scales must be four finite positive values")
    try:
        prediction = model.predict_next(data.state, data.action)
    except (PredictionSafetyError, FloatingPointError) as exc:
        return _failure(str(exc))
    if prediction.shape != data.next_state.shape:
        raise ValueError("predict_next returned the wrong state shape")
    return _scaled_metrics(_error(prediction, data.next_state), scale)


@torch.no_grad()
def rollout_metrics(model: Any, initial_states: Tensor, actions: Tensor, truth: Tensor,
                    horizons: Sequence[int] = (5, 20, 50, 100),
                    state_scale: Sequence[float] = STATE_SCALE) -> dict[int, dict[str, Any]]:
    """Open-loop errors over steps 1..H, plus endpoint NMSE at H.

    State scales are predeclared generic scales, not hidden physics or query
    statistics. A late numerical failure preserves earlier valid horizons and
    marks every horizon beyond it failed. No true-state feedback is used.
    """
    if initial_states.ndim != 2 or initial_states.shape[-1] != 4 or len(initial_states) < 1:
        raise ValueError("initial_states must have shape (B, 4)")
    if actions.ndim != 3 or actions.shape[0] != len(initial_states) or actions.shape[-1] != 1:
        raise ValueError("actions must have shape (B, T, 1)")
    if truth.shape != (len(initial_states), actions.shape[1] + 1, 4):
        raise ValueError("truth must have shape (B, T+1, 4), including initial states")
    if not bool(torch.isfinite(initial_states).all()) or not bool(torch.isfinite(actions).all()) or not bool(torch.isfinite(truth).all()):
        raise ValueError("rollout inputs contain NaN or Inf")
    if not torch.allclose(truth[:, 0], initial_states):
        raise ValueError("truth[:, 0] must equal the rollout initial states")
    if any(isinstance(horizon, bool) or not isinstance(horizon, int) for horizon in horizons):
        raise ValueError("horizons must be integers")
    requested = tuple(horizons)
    if not requested or any(h < 1 or h > actions.shape[1] for h in requested) or len(set(requested)) != len(requested):
        raise ValueError("horizons must be distinct positive integers within action sequence")
    scale = torch.as_tensor(state_scale, device=initial_states.device, dtype=initial_states.dtype)
    if scale.shape != (4,) or not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("state_scale must have four finite positive values")
    current = initial_states
    predictions: list[Tensor] = []
    failure = None
    for index in range(max(requested)):
        try:
            current = model.predict_next(current, actions[:, index])
        except (PredictionSafetyError, FloatingPointError) as exc:
            failure = str(exc)
            break
        if current.shape != initial_states.shape:
            raise ValueError("predict_next returned the wrong state shape")
        if not bool(torch.isfinite(current).all()):
            failure = f"nonfinite prediction at rollout step {index + 1}"
            break
        predictions.append(current)
    results: dict[int, dict[str, Any]] = {}
    for horizon in requested:
        if horizon > len(predictions):
            metrics = _failure(failure or "rollout did not reach horizon")
            metrics["endpoint_nmse"] = None
        else:
            prediction = torch.stack(predictions[:horizon], dim=1)
            error = _error(prediction, truth[:, 1:horizon + 1])
            metrics = _scaled_metrics(error, scale)
            metrics["endpoint_nmse"] = _scaled_metrics(error[:, -1], scale)["nmse"]
        metrics.update({"horizon": horizon, "valid_steps": len(predictions),
                        "scale": scale.cpu().tolist(), "aggregation": "mean over steps 1..H and trajectories"})
        results[horizon] = metrics
    return results


def fraction_error_removed(after: float | None, before: float | None,
                           epsilon: float = 1e-12) -> float | None:
    if after is None or before is None or not math.isfinite(after) or not math.isfinite(before) or abs(before) <= epsilon:
        return None
    return 1.0 - after / before


def recovery_ratio(after: float | None, before: float | None, id_error: float | None,
                   epsilon: float = 1e-12) -> float | None:
    """Unclamped recovery relative to ID; undefined denominator returns None."""
    if any(value is None or not math.isfinite(value) for value in (after, before, id_error)):
        return None
    denominator = before - id_error
    return None if abs(denominator) <= epsilon else 1.0 - (after - id_error) / denominator
