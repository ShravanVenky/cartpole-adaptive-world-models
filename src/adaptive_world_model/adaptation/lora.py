"""Guarded support updates of internal A/B factors; immutable V1 code is reused."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
import time
from typing import Any, Mapping

import torch
from torch import nn

from adaptive_world_model.environments import observable_features
from adaptive_world_model.models.dynamics import FrozenDynamicsModel, PredictionSafetyError, RolloutMixin
from adaptive_world_model.models.lora import LoRAAdapter
from .meta import AdaptationSettings, _finite, _norm


def lora_normalized_delta(base: FrozenDynamicsModel, adapter: LoRAAdapter, xn: torch.Tensor,
                          params: Mapping[str, torch.Tensor] | None = None,
                          residual_output_clip: float | None = None) -> torch.Tensor:
    """Functional combined prediction; optional identical net-correction bound."""
    prediction = adapter.normalized_delta(base, xn, params)
    if residual_output_clip is None:
        return prediction
    frozen = base.network(xn)
    return frozen + (prediction - frozen).clamp(-residual_output_clip, residual_output_clip)


def lora_inner_adapt(base: FrozenDynamicsModel, adapter: LoRAAdapter, support: Any,
                     settings: AdaptationSettings | dict[str, Any] | None = None,
                     first_order: bool = False,
                     initial_params: OrderedDict[str, torch.Tensor] | None = None
                     ) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """One-step support loss, with exact MAML graph by default, updating A/B only.

    Separate held-out query/rollout objectives are supplied by the experiment.
    Nothing here selects support actions or reads hidden physical parameters.
    """
    actual = (AdaptationSettings() if settings is None else
              AdaptationSettings(**settings) if isinstance(settings, dict) else settings)
    base.assert_frozen()
    reference = next(adapter.parameters())
    support = support.to(reference.device, dtype=reference.dtype)
    if len(support) == 0:
        raise ValueError("LoRA adaptation requires nonempty support data.")
    params = OrderedDict(adapter.named_parameters()) if initial_params is None else OrderedDict(initial_params)
    if set(params) != set(dict(adapter.named_parameters())) or not all(value.requires_grad for value in params.values()):
        raise ValueError("Inner LoRA parameters must be the exact A/B set and require gradients.")
    anchor = OrderedDict(params)
    log: dict[str, Any] = {"accepted": False, "reason": None, "gradient_steps": 0,
                          "steps_attempted": 0, "support_losses": [], "gradient_norms": [],
                          "step_norms": [], "parameter_update_norm": 0.0,
                          "max_gradient_norm": 0.0, "max_step_norm": 0.0,
                          "first_order": bool(first_order), "support_count": len(support)}
    with torch.no_grad():
        xn = base.normalizer.encode_x(observable_features(support.state, support.action))
        target = base.normalizer.encode_y(support.next_state - support.state)
        frozen = base.network(xn)
    if not _finite([xn, target, frozen, *params.values()]):
        log.update(reason="nonfinite_support_or_initialization", rolled_back=True)
        return anchor, log

    def prediction(candidate: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raw = adapter.normalized_delta(base, xn, candidate)
        return frozen + (raw - frozen).clamp(-actual.residual_output_clip, actual.residual_output_clip)

    initial_prediction_normalized = prediction(params)
    initial_loss = (initial_prediction_normalized - target).square().mean()
    initial_value = float(initial_loss.detach())
    log["support_losses"].append(initial_value)
    initial_prediction = base.normalizer.decode_y(initial_prediction_normalized.detach())
    if not _finite([initial_loss, initial_prediction]) or initial_value > actual.max_support_loss:
        log.update(reason="initial_support_loss_bound", rolled_back=True)
        return anchor, log
    if initial_prediction.detach().abs().max() > actual.max_prediction_magnitude:
        log.update(reason="initial_prediction_magnitude_bound", rolled_back=True)
        return anchor, log
    allowed_loss = min(actual.max_support_loss, actual.loss_growth_factor * initial_value + 1e-6)
    current_loss = initial_loss
    for _ in range(actual.steps):
        gradients = torch.autograd.grad(current_loss, tuple(params.values()), create_graph=not first_order)
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
        if actual.gradient_clip is not None:
            factor = (actual.gradient_clip / gradient_norm.clamp_min(1e-12)).clamp_max(1.0)
        raw_steps = [actual.learning_rate * factor * gradient for gradient in gradients]
        step_factor = (actual.step_norm_bound / _norm(raw_steps).clamp_min(1e-12)).clamp_max(1.0)
        updates = [step_factor * value for value in raw_steps]
        log["step_norms"].append(float(_norm(updates).detach()))
        log["max_step_norm"] = max(log["max_step_norm"], log["step_norms"][-1])
        candidate = OrderedDict((name, parameter - update)
                                for (name, parameter), update in zip(params.items(), updates))
        update_norm = _norm([candidate[name] - anchor[name] for name in candidate])
        log["candidate_update_norm"] = float(update_norm.detach())
        candidate_prediction = prediction(candidate)
        candidate_loss = (candidate_prediction - target).square().mean()
        physical_delta = base.normalizer.decode_y(candidate_prediction.detach())
        if not _finite([*candidate.values(), candidate_loss, physical_delta]):
            log["reason"] = "nonfinite_candidate"
            break
        if update_norm.detach() > actual.max_parameter_delta:
            log["reason"] = "parameter_update_norm_bound"
            break
        if float(candidate_loss.detach()) > allowed_loss:
            log["reason"] = "support_loss_growth_bound"
            break
        if physical_delta.detach().abs().max() > actual.max_prediction_magnitude:
            log["reason"] = "prediction_magnitude_bound"
            break
        params = candidate
        current_loss = candidate_loss
        log["support_losses"].append(float(candidate_loss.detach()))
        log["gradient_steps"] += 1
        log["parameter_update_norm"] = float(update_norm.detach())
    if log["reason"] is not None:
        log.update(rolled_back=True, rejected_update_norm=log.get("candidate_update_norm", 0.0),
                   parameter_update_norm=0.0)
        base.assert_frozen()
        return anchor, log
    log.update(accepted=True, rolled_back=False)
    base.assert_frozen()
    return params, log


class MetaLoRAModel(nn.Module, RolloutMixin):
    """Deployment interface identical to output residual, with only A/B plastic."""

    checkpoint_schema = "adaptive_world_model.cartpole.meta_lora.v1"

    def __init__(self, base: FrozenDynamicsModel, adapter: LoRAAdapter,
                 settings: AdaptationSettings | dict[str, Any] | None = None) -> None:
        super().__init__()
        self.base = base
        self.base.freeze()
        self.adapter = adapter.to(next(base.parameters()))
        self.adapter._validate_base(base)
        self.settings = (AdaptationSettings() if settings is None else
                         AdaptationSettings(**settings) if isinstance(settings, dict) else settings)
        self._adapted_params: OrderedDict[str, torch.Tensor] | None = None
        self.last_adaptation: dict[str, Any] = {}

    @property
    def parameter_count(self) -> int:
        return self.adapter.parameter_count

    def _params(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(self.adapter.named_parameters()) if self._adapted_params is None else self._adapted_params

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        if self._adapted_params is not None:
            self._adapted_params = OrderedDict((name, fn(value)) for name, value in self._adapted_params.items())
        return self

    def predict_delta(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        xn = self.base.normalizer.encode_x(observable_features(state, action))
        normalized = lora_normalized_delta(self.base, self.adapter, xn, self._params(),
                                           self.settings.residual_output_clip)
        values = self.base.normalizer.decode_y(normalized)
        if not torch.isfinite(values).all() or values.detach().abs().max() > self.settings.max_prediction_magnitude:
            raise PredictionSafetyError("LoRA produced a nonfinite or excessive physical delta.")
        return values

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predict_delta(state, action)

    def reset_adaptation(self) -> None:
        self._adapted_params = None
        self.last_adaptation = {}

    def adapt(self, support_data: Any) -> dict[str, Any]:
        self.base.assert_frozen()
        reference = next(self.adapter.parameters())
        if reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        started = time.perf_counter()
        before = (None if self._adapted_params is None else
                  OrderedDict((name, value.detach().clone()) for name, value in self._adapted_params.items()))
        initial = OrderedDict((name, value.detach().clone().requires_grad_(True)) for name, value in self._params().items())
        with torch.enable_grad():
            params, log = lora_inner_adapt(self.base, self.adapter, support_data, self.settings,
                                          first_order=True, initial_params=initial)
        self._adapted_params = (OrderedDict((name, value.detach()) for name, value in params.items())
                                if log["accepted"] else before)
        if reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        log.update(method="meta_lora", support_count=len(support_data),
                   adaptation_seconds=time.perf_counter() - started,
                   adapter_parameter_count=self.adapter.parameter_count)
        self.last_adaptation = log
        self.base.assert_frozen()
        return log

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": self.checkpoint_schema, "rank": self.adapter.rank,
                    "alpha": self.adapter.alpha, "layer_indices": list(self.adapter.layer_indices),
                    "adapter_seed": self.adapter.seed, "meta_trained": bool(self.adapter.meta_trained),
                    "adapter_state": self.adapter.state_dict(),
                    "adapter_metadata": self.adapter.checkpoint_metadata,
                    "settings": asdict(self.settings), "adapted_params": self._adapted_params,
                    "metadata": metadata or {}}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, base: FrozenDynamicsModel,
                        map_location: str | torch.device = "cpu") -> "MetaLoRAModel":
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if checkpoint.get("schema") != cls.checkpoint_schema:
            raise ValueError("Unsupported meta LoRA wrapper checkpoint schema.")
        adapter = LoRAAdapter(base, checkpoint["rank"], tuple(checkpoint["layer_indices"]),
                              checkpoint["alpha"], checkpoint["adapter_seed"])
        adapter.load_state_dict(checkpoint["adapter_state"], strict=True)
        adapter.meta_trained = bool(checkpoint["meta_trained"])
        adapter.checkpoint_metadata = checkpoint["adapter_metadata"]
        result = cls(base, adapter, checkpoint["settings"])
        if checkpoint["adapted_params"] is not None:
            result._adapted_params = OrderedDict((name, value.to(next(base.parameters())))
                                                 for name, value in checkpoint["adapted_params"].items())
        return result
