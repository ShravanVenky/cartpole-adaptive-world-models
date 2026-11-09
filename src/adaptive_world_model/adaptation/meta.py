"""Guarded exact/first-order MAML tensor flow; no training starts on import."""

from __future__ import annotations

import time
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

from adaptive_world_model.environments.cartpole import observable_features
from adaptive_world_model.models.dynamics import FrozenDynamicsModel, PredictionSafetyError, RolloutMixin
from adaptive_world_model.models.residual import ResidualAdapter


@dataclass(frozen=True)
class AdaptationSettings:
    learning_rate: float = 0.01
    steps: int = 10
    gradient_clip: float | None = 1.0
    step_norm_bound: float = 0.1
    max_parameter_delta: float = 2.0
    residual_output_clip: float = 8.0
    max_prediction_magnitude: float = 50.0
    max_support_loss: float = 100.0
    loss_growth_factor: float = 4.0

    def __post_init__(self) -> None:
        positive = (self.learning_rate, self.step_norm_bound, self.max_parameter_delta,
                    self.residual_output_clip, self.max_prediction_magnitude,
                    self.max_support_loss, self.loss_growth_factor)
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Adaptation learning rate and numerical bounds must be finite positive numbers.")
        if isinstance(self.steps, bool) or not isinstance(self.steps, Integral) or self.steps < 1:
            raise ValueError("Adaptation steps must be a positive integer; booleans/floats are invalid.")
        if self.gradient_clip is not None and (
            isinstance(self.gradient_clip, bool) or not isinstance(self.gradient_clip, Real)
            or not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0
        ):
            raise ValueError("gradient_clip must be None or a finite positive number.")


def _norm(values: Any) -> torch.Tensor:
    return torch.linalg.vector_norm(torch.cat([value.reshape(-1) for value in values]))


def _finite(values: Any) -> bool:
    return all(bool(torch.isfinite(value.detach()).all()) for value in values)


def inner_adapt(adapter: ResidualAdapter, xn: torch.Tensor, residual_target: torch.Tensor,
                settings: AdaptationSettings | dict[str, Any] | None = None,
                first_order: bool = False,
                initial_params: OrderedDict[str, torch.Tensor] | None = None
                ) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """Differentiate through support updates without overwriting initialization.

    Exact MAML is the default: ``create_graph=True`` keeps support Hessian terms.
    First order detaches support gradients while preserving each updated
    parameter's identity connection to the initialization. Rejections explicitly
    return the pre-adaptation parameters; no corrupted candidate is installed.
    """
    if settings is None:
        settings = AdaptationSettings()
    elif isinstance(settings, dict):
        settings = AdaptationSettings(**settings)
    if xn.ndim != 2 or xn.shape[1] != 6 or residual_target.shape != (len(xn), 4) or len(xn) == 0:
        raise ValueError("Inner adaptation expects nonempty xn[N,6], residual_target[N,4].")
    params = OrderedDict(adapter.named_parameters()) if initial_params is None else OrderedDict(initial_params)
    if set(params) != set(dict(adapter.named_parameters())) or not all(p.requires_grad for p in params.values()):
        raise ValueError("Inner parameters must match adapter and require gradients.")
    anchor = OrderedDict(params)
    log: dict[str, Any] = {"accepted": False, "reason": None, "gradient_steps": 0,
                          "steps_attempted": 0,
                          "support_losses": [], "gradient_norms": [], "step_norms": [],
                          "parameter_update_norm": 0.0, "max_gradient_norm": 0.0,
                          "max_step_norm": 0.0, "first_order": bool(first_order)}
    if not _finite([xn, residual_target, *params.values()]):
        log.update(reason="nonfinite_support_or_initialization", rolled_back=True)
        return anchor, log

    def objective(candidate: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        residual = functional_call(adapter, candidate, (xn,)).clamp(
            -settings.residual_output_clip, settings.residual_output_clip)
        return (residual - residual_target).square().mean()

    initial_loss = objective(params)
    initial_value = float(initial_loss.detach())
    log["support_losses"].append(initial_value)
    if not torch.isfinite(initial_loss.detach()) or initial_value > settings.max_support_loss:
        log.update(reason="initial_support_loss_bound", rolled_back=True)
        return anchor, log
    allowed_loss = min(settings.max_support_loss, settings.loss_growth_factor * initial_value + 1e-6)
    for _ in range(settings.steps):
        loss = objective(params)
        gradients = torch.autograd.grad(loss, tuple(params.values()), create_graph=not first_order)
        log["steps_attempted"] += 1
        if not _finite(gradients):
            log["reason"] = "nonfinite_gradient"
            break
        if first_order:
            gradients = tuple(gradient.detach() for gradient in gradients)
        gradient_norm = _norm(gradients)
        log["gradient_norms"].append(float(gradient_norm.detach()))
        log["max_gradient_norm"] = max(log["max_gradient_norm"], log["gradient_norms"][-1])
        factor = gradient_norm.new_tensor(1.0)
        if settings.gradient_clip is not None:
            factor = (settings.gradient_clip / gradient_norm.clamp_min(1e-12)).clamp_max(1.0)
        step_values = [settings.learning_rate * factor * gradient for gradient in gradients]
        raw_step_norm = _norm(step_values)
        step_factor = (settings.step_norm_bound / raw_step_norm.clamp_min(1e-12)).clamp_max(1.0)
        steps = [step_factor * value for value in step_values]
        log["step_norms"].append(float(_norm(steps).detach()))
        log["max_step_norm"] = max(log["max_step_norm"], log["step_norms"][-1])
        candidate = OrderedDict((name, parameter - step)
                                for (name, parameter), step in zip(params.items(), steps))
        update_norm = _norm([candidate[name] - anchor[name] for name in candidate])
        log["candidate_update_norm"] = float(update_norm.detach())
        candidate_loss = objective(candidate)
        if not _finite([*candidate.values(), candidate_loss]):
            log["reason"] = "nonfinite_candidate"
            break
        if update_norm.detach() > settings.max_parameter_delta:
            log["reason"] = "parameter_update_norm_bound"
            break
        if float(candidate_loss.detach()) > allowed_loss:
            log["reason"] = "support_loss_growth_bound"
            break
        params = candidate
        log["support_losses"].append(float(candidate_loss.detach()))
        log["gradient_steps"] += 1
        log["parameter_update_norm"] = float(update_norm.detach())
    if log["reason"] is not None:
        log["rolled_back"] = True
        log["rejected_update_norm"] = log.get("candidate_update_norm", log["parameter_update_norm"])
        log["parameter_update_norm"] = 0.0
        return anchor, log
    log["accepted"] = True
    log["rolled_back"] = False
    return params, log


def meta_objective(base: FrozenDynamicsModel, adapter: ResidualAdapter,
                   support: Any, query: Any,
                   settings: AdaptationSettings | dict[str, Any] | None = None,
                   first_order: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
    """One support/query task; held-out trajectories may not overlap support.

    The caller later averages task losses and performs the outer Adam update.
    This function never updates a module, optimizer, or base normalization.
    """
    support.assert_disjoint(query)
    # Evaluator-only metadata is used to validate task identity, never as a
    # feature or predictor input. Held-out query must come from the same task.
    support_physics = support.metadata.get("physics")
    query_physics = query.metadata.get("physics")
    if support_physics is not None and query_physics is not None and support_physics != query_physics:
        raise ValueError("Meta support/query must belong to the same physical environment.")
    base.assert_frozen()
    reference = next(adapter.parameters())
    support = support.to(reference.device, dtype=reference.dtype)
    query = query.to(reference.device, dtype=reference.dtype)
    with torch.no_grad():
        sx = base.normalizer.encode_x(observable_features(support.state, support.action))
        sy = base.normalizer.encode_y(support.next_state - support.state) - base.network(sx)
        qx = base.normalizer.encode_x(observable_features(query.state, query.action))
        qy = base.normalizer.encode_y(query.next_state - query.state) - base.network(qx)
    actual_settings = (AdaptationSettings() if settings is None else
                       AdaptationSettings(**settings) if isinstance(settings, dict) else settings)
    params, log = inner_adapt(adapter, sx, sy, actual_settings, first_order=first_order)
    prediction = functional_call(adapter, params, (qx,)).clamp(
        -actual_settings.residual_output_clip, actual_settings.residual_output_clip)
    loss = (prediction - qy).square().mean()
    if not torch.isfinite(loss.detach()):
        raise PredictionSafetyError("Nonfinite held-out meta objective; outer update must be rejected.")
    with torch.no_grad():
        physical_delta = base.normalizer.decode_y(base.network(qx) + prediction)
        if not torch.isfinite(physical_delta).all() or physical_delta.abs().max() > actual_settings.max_prediction_magnitude:
            raise PredictionSafetyError("Meta objective has excessive physical deltas; reject outer update.")
    log["query_loss"] = float(loss.detach())
    log["support_count"] = len(support.state)
    log["query_count"] = len(query.state)
    log["support_query_disjoint"] = True
    log["same_environment_verified"] = support_physics is not None and query_physics is not None
    base.assert_frozen()
    return loss, log


class MetaResidualModel(nn.Module, RolloutMixin):
    """Deployment wrapper: learned initialization + independent task updates."""

    checkpoint_schema = "adaptive_world_model.cartpole.meta_residual.v1"

    def __init__(self, base: FrozenDynamicsModel, adapter: ResidualAdapter,
                 settings: AdaptationSettings | dict[str, Any] | None = None) -> None:
        super().__init__()
        self.base = base
        self.base.freeze()
        reference = next(base.parameters())
        self.adapter = adapter.to(reference)
        self.settings = (AdaptationSettings() if settings is None else
                         AdaptationSettings(**settings) if isinstance(settings, dict) else settings)
        self._adapted_params: OrderedDict[str, torch.Tensor] | None = None
        self.last_adaptation: dict[str, Any] = {}

    @property
    def parameter_count(self) -> int:
        return self.adapter.parameter_count

    def _params(self) -> OrderedDict[str, torch.Tensor]:
        return (OrderedDict(self.adapter.named_parameters()) if self._adapted_params is None
                else self._adapted_params)

    def _apply(self, fn, recurse: bool = True):
        # Adapted parameters are task-local tensors, not learned initialization
        # Parameters; keep them aligned when a caller moves the wrapper/device.
        super()._apply(fn, recurse=recurse)
        if self._adapted_params is not None:
            self._adapted_params = OrderedDict((name, fn(value))
                                               for name, value in self._adapted_params.items())
        return self

    def predict_delta(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        xn = self.base.normalizer.encode_x(observable_features(state, action))
        residual = functional_call(self.adapter, self._params(), (xn,)).clamp(
            -self.settings.residual_output_clip, self.settings.residual_output_clip)
        values = self.base.normalizer.decode_y(self.base.network(xn) + residual)
        if not torch.isfinite(values).all() or values.detach().abs().max() > self.settings.max_prediction_magnitude:
            raise PredictionSafetyError("Meta adapter produced a nonfinite or excessive physical delta.")
        return values

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predict_delta(state, action)

    def reset_adaptation(self) -> None:
        self._adapted_params = None
        self.last_adaptation = {}

    def adapt(self, support_data: Any) -> dict[str, Any]:
        """Update a detached task copy, preserving the reusable phi_0 exactly."""
        self.base.assert_frozen()
        reference = next(self.adapter.parameters())
        if reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        started = time.perf_counter()
        support = support_data.to(reference.device, dtype=reference.dtype)
        before = (None if self._adapted_params is None else
                  OrderedDict((name, value.detach().clone()) for name, value in self._adapted_params.items()))
        initial = OrderedDict((name, value.detach().clone().requires_grad_(True))
                              for name, value in self._params().items())
        with torch.no_grad():
            xn = self.base.normalizer.encode_x(observable_features(support.state, support.action))
            target = self.base.normalizer.encode_y(support.next_state - support.state) - self.base.network(xn)
        # Deployment does not need an outer graph; first-order gradients avoid
        # retaining Hessian graphs in evaluation, but use the identical updates.
        with torch.enable_grad():
            params, log = inner_adapt(self.adapter, xn, target, self.settings,
                                      first_order=True, initial_params=initial)
        if log["accepted"]:
            with torch.no_grad():
                values = self.base.normalizer.decode_y(
                    self.base.network(xn) + functional_call(self.adapter, params, (xn,)).clamp(
                        -self.settings.residual_output_clip, self.settings.residual_output_clip))
            if not torch.isfinite(values).all() or values.abs().max() > self.settings.max_prediction_magnitude:
                log.update(accepted=False, reason="prediction_magnitude_bound", rolled_back=True,
                           rejected_update_norm=log["parameter_update_norm"], parameter_update_norm=0.0)
        if log["accepted"]:
            self._adapted_params = OrderedDict((name, value.detach()) for name, value in params.items())
        else:
            self._adapted_params = before
        if reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        log.update(method="meta", support_count=len(support.state),
                   adaptation_seconds=time.perf_counter() - started,
                   adapter_parameter_count=self.adapter.parameter_count)
        self.last_adaptation = log
        self.base.assert_frozen()
        return log

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": self.checkpoint_schema, "adapter_seed": self.adapter.seed,
                    "adapter_hidden_sizes": list(self.adapter.hidden_sizes),
                    "meta_trained": bool(self.adapter.meta_trained),
                    "adapter_state": self.adapter.state_dict(), "settings": asdict(self.settings),
                    "adapted_params": self._adapted_params, "metadata": metadata or {}}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, base: FrozenDynamicsModel,
                        map_location: str | torch.device = "cpu") -> "MetaResidualModel":
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if checkpoint.get("schema") != cls.checkpoint_schema:
            raise ValueError("Unsupported meta-residual checkpoint schema.")
        adapter = ResidualAdapter(seed=checkpoint["adapter_seed"],
                                  hidden_sizes=tuple(checkpoint["adapter_hidden_sizes"])).to(map_location)
        adapter.load_state_dict(checkpoint["adapter_state"], strict=True)
        adapter.meta_trained = bool(checkpoint["meta_trained"])
        result = cls(base, adapter, AdaptationSettings(**checkpoint["settings"]))
        if checkpoint["adapted_params"] is not None:
            reference = next(base.parameters())
            result._adapted_params = OrderedDict((name, value.to(reference))
                                                for name, value in checkpoint["adapted_params"].items())
        return result
