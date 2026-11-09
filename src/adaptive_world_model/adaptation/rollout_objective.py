"""One-step and truly recursive query objectives for the V2 factorial study.

Inner deployment updates remain one-step in every condition. Only the outer
held-out objective changes. This file never performs an optimizer update.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from numbers import Integral, Real
from typing import Any

import torch
from torch import Tensor
from torch.func import functional_call

from ..data.trajectories import trajectory_windows
from ..environments import angle_difference, observable_features
from ..models.dynamics import FrozenDynamicsModel, PredictionSafetyError
from .meta import AdaptationSettings, inner_adapt


def _settings(settings: AdaptationSettings | dict[str, Any] | None) -> AdaptationSettings:
    return AdaptationSettings() if settings is None else AdaptationSettings(**settings) if isinstance(settings, dict) else settings


def _positive_weight(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def neural_meta_objective(
    base: FrozenDynamicsModel, adapter: Any, support: Any, query: Any,
    settings: AdaptationSettings | dict[str, Any] | None = None,
    objective: str = "one_step", horizon: int = 5, rollout_weight: float = 0.1,
    first_order: bool = False, kind: str = "residual",
    nominal_data: Any | None = None, nominal_weight: float = 0.0,
) -> tuple[Tensor, dict[str, Any]]:
    """Return an outer loss with the inner/outer graph preserved.

    Residual one_step uses the exact V1 residual-target loss arithmetic.
    Rollout adds lambda times mean((wrapped physical error / nominal delta
    scale)^2), uniformly over windows, h=1..H and four state components.
    Optional nominal preservation penalizes the *unadapted initialization's*
    deviation from frozen prediction; its default weight is zero.
    """
    if objective not in ("one_step", "rollout") or kind not in ("residual", "lora"):
        raise ValueError("objective must be one_step/rollout and kind residual/lora")
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    _positive_weight(rollout_weight, "rollout_weight")
    _positive_weight(nominal_weight, "nominal_weight")
    if nominal_weight and nominal_data is None:
        raise ValueError("nominal preservation requires separate nominal_data")
    support.assert_disjoint(query)
    support_physics, query_physics = support.metadata.get("physics"), query.metadata.get("physics")
    if support_physics is not None and query_physics is not None and support_physics != query_physics:
        raise ValueError("Meta support/query must belong to the same physical environment")
    base.assert_frozen()
    reference = next(adapter.parameters())
    support = support.to(reference.device, dtype=reference.dtype)
    query = query.to(reference.device, dtype=reference.dtype)
    actual = _settings(settings)
    with torch.no_grad():
        sx = base.normalizer.encode_x(observable_features(support.state, support.action))
        sy = base.normalizer.encode_y(support.delta()) - base.network(sx)
        qx = base.normalizer.encode_x(observable_features(query.state, query.action))
        q_base = base.network(qx)
        q_target = base.normalizer.encode_y(query.delta())
        q_residual_target = q_target - q_base
    if kind == "residual":
        params, log = inner_adapt(adapter, sx, sy, actual, first_order=first_order)
        correction = functional_call(adapter, params, (qx,)).clamp(-actual.residual_output_clip, actual.residual_output_clip)
        one_step = (correction - q_residual_target).square().mean()
        query_prediction = q_base + correction
    else:
        from .lora import lora_inner_adapt
        params, log = lora_inner_adapt(base, adapter, support, actual, first_order=first_order)
        raw = adapter.normalized_delta(base, qx, params)
        query_prediction = q_base + (raw - q_base).clamp(-actual.residual_output_clip, actual.residual_output_clip)
        one_step = (query_prediction - q_target).square().mean()

    def check_delta(normalized_delta: Tensor) -> Tensor:
        physical = base.normalizer.decode_y(normalized_delta)
        if not bool(torch.isfinite(physical.detach()).all()) or physical.detach().abs().max() > actual.max_prediction_magnitude:
            raise PredictionSafetyError("Query objective has nonfinite or excessive physical deltas")
        return physical

    def normalized_forward(state: Tensor, action: Tensor, plastic: OrderedDict | None) -> Tensor:
        features = base.normalizer.encode_x(observable_features(state, action))
        # Frozen weights still propagate derivatives through predicted inputs.
        # A no_grad block here would break genuine recursive outer gradients.
        frozen = base.network(features)
        if kind == "residual":
            correction = functional_call(adapter, plastic, (features,)) if plastic is not None else adapter(features)
        else:
            correction = adapter.normalized_delta(base, features, plastic) - frozen
        return frozen + correction.clamp(-actual.residual_output_clip, actual.residual_output_clip)

    check_delta(query_prediction)
    rollout = one_step.new_zeros(())
    nominal = one_step.new_zeros(())
    windows = 0
    if objective == "rollout":
        initial, actions, truth = trajectory_windows(query, int(horizon))
        scale = base.normalizer.y_scale
        if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
            raise ValueError("rollout normalization needs fixed finite positive nominal delta scales")
        predicted = initial
        errors = []
        for timestep in range(int(horizon)):
            predicted = predicted + check_delta(normalized_forward(predicted, actions[:, timestep], params))
            if not bool(torch.isfinite(predicted.detach()).all()):
                raise PredictionSafetyError("Recursive query state became nonfinite")
            error = predicted - truth[:, timestep + 1]
            error = error.clone()
            error[:, 2] = angle_difference(predicted[:, 2], truth[:, timestep + 1, 2])
            errors.append(error / scale)
        rollout = torch.stack(errors, dim=1).square().mean()
        windows = len(initial)
    if nominal_weight:
        support.assert_disjoint(nominal_data)
        query.assert_disjoint(nominal_data)
        nominal_data = nominal_data.to(reference.device, dtype=reference.dtype)
        nominal_x = base.normalizer.encode_x(observable_features(nominal_data.state, nominal_data.action))
        baseline = base.network(nominal_x)
        nominal_prediction = normalized_forward(nominal_data.state, nominal_data.action, None)
        check_delta(nominal_prediction)
        nominal = (nominal_prediction - baseline).square().mean()
    total = one_step + (rollout_weight * rollout if objective == "rollout" else 0) + nominal_weight * nominal
    if not bool(torch.isfinite(total.detach())):
        raise PredictionSafetyError("Nonfinite outer objective; optimizer update must be rejected")
    log.update({"objective": objective, "kind": kind, "inner_objective": "one_step",
                "one_step_loss": float(one_step.detach()), "rollout_loss": float(rollout.detach()),
                "nominal_preservation_loss": float(nominal.detach()), "total_loss": float(total.detach()),
                "query_loss": float(total.detach()), "rollout_horizon": int(horizon) if objective == "rollout" else 0,
                "rollout_weight": float(rollout_weight) if objective == "rollout" else 0.0,
                "rollout_windows": windows, "rollout_timestep_weights": "uniform",
                "rollout_normalization": "fixed_nominal_delta_scales",
                "nominal_weight": float(nominal_weight), "support_count": len(support), "query_count": len(query),
                "support_query_disjoint": True, "same_environment_verified": support_physics is not None and query_physics is not None})
    base.assert_frozen()
    return total, log
