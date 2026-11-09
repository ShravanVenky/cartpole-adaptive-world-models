"""Fair linear/generic polynomial Ridge on the models' common observables."""

from __future__ import annotations

import time
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from torch import nn

from adaptive_world_model.environments.cartpole import observable_features
from adaptive_world_model.models.dynamics import FrozenDynamicsModel, PredictionSafetyError, RolloutMixin


class RidgeResidualModel(nn.Module, RolloutMixin):
    """Fit a residual using ONLY normalized [x,v,sin(theta),cos(theta),w,force].

    Degree 1 adds an intercept: 7 features x 4 targets = 28 coefficients.
    Optional degree 2 applies *every* unordered input pair mechanically:
    6 linear + 21 products + 1 intercept = 28 features / 112 coefficients.
    Neither variant has access to physical parameters or analytical dynamics.
    """

    checkpoint_schema = "adaptive_world_model.cartpole.ridge.v1"

    def __init__(self, base: FrozenDynamicsModel, regularization: float = 0.1, degree: int = 1,
                 max_coefficient_norm: float = 20.0, residual_output_clip: float = 8.0,
                 max_prediction_magnitude: float = 50.0) -> None:
        super().__init__()
        if isinstance(degree, bool) or not isinstance(degree, Integral) or degree not in (1, 2):
            raise ValueError("Ridge degree must be the integer 1 or 2.")
        bounds = (regularization, max_coefficient_norm, residual_output_clip, max_prediction_magnitude)
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) or value <= 0 for value in bounds):
            raise ValueError("Ridge regularization and safety bounds must be finite positive numbers.")
        self.base = base
        self.base.freeze()
        self.regularization = float(regularization)
        self.degree = int(degree)
        self.max_coefficient_norm = float(max_coefficient_norm)
        self.residual_output_clip = float(residual_output_clip)
        self.max_prediction_magnitude = float(max_prediction_magnitude)
        self.feature_count = 7 if degree == 1 else 28
        reference = next(base.network.parameters())
        self.register_buffer("coefficients", reference.new_zeros(self.feature_count, 4))
        self.last_adaptation: dict[str, Any] = {}

    @property
    def coefficient_count(self) -> int:
        return self.coefficients.numel()

    @property
    def method_name(self) -> str:
        return "ridge" if self.degree == 1 else "generic_polynomial_ridge"

    def ridge_features(self, normalized_features: torch.Tensor) -> torch.Tensor:
        if normalized_features.shape[-1] != 6:
            raise ValueError("Ridge accepts exactly the six common observable features.")
        columns = [normalized_features]
        if self.degree == 2:
            columns.append(torch.stack([
                normalized_features[..., left] * normalized_features[..., right]
                for left in range(6) for right in range(left, 6)], dim=-1))
        columns.append(torch.ones_like(normalized_features[..., :1]))
        return torch.cat(columns, dim=-1)

    def predict_delta(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = observable_features(state, action)
        xn = self.base.normalizer.encode_x(features)
        residual = (self.ridge_features(xn) @ self.coefficients).clamp(
            -self.residual_output_clip, self.residual_output_clip)
        values = self.base.normalizer.decode_y(self.base.network(xn) + residual)
        if not torch.isfinite(values).all() or values.detach().abs().max() > self.max_prediction_magnitude:
            raise PredictionSafetyError("Ridge produced a nonfinite or excessive physical delta.")
        return values

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predict_delta(state, action)

    def reset_adaptation(self) -> None:
        self.coefficients.zero_()
        self.last_adaptation = {}

    def adapt(self, support_data: Any) -> dict[str, Any]:
        self.base.assert_frozen()
        reference = self.coefficients
        if reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        started = time.perf_counter()
        state = support_data.state.to(reference)
        action = support_data.action.to(reference)
        next_state = support_data.next_state.to(reference)
        if len(state) == 0:
            raise ValueError("Cannot adapt Ridge on an empty support set.")
        before = self.coefficients.detach().clone()
        log: dict[str, Any] = {"method": self.method_name, "accepted": False,
                              "support_count": len(state), "gradient_steps": 0,
                              "coefficient_count": self.coefficient_count,
                              "parameter_update_norm": 0.0, "reason": None}
        with torch.no_grad():
            features = observable_features(state, action)
            xn = self.base.normalizer.encode_x(features)
            target = self.base.normalizer.encode_y(next_state - state) - self.base.network(xn)
            design = self.ridge_features(xn)
            if not torch.isfinite(design).all() or not torch.isfinite(target).all():
                log["reason"] = "nonfinite_support"
            else:
                try:
                    # Solve in float64; small K and duplicate/constant features
                    # are regularized without numerical dependence on inverses.
                    matrix = design.double()
                    rhs = target.double()
                    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
                    candidate = torch.linalg.solve(
                        matrix.T @ matrix + self.regularization * identity, matrix.T @ rhs).to(reference)
                    magnitude = torch.linalg.vector_norm(candidate)
                    update = torch.linalg.vector_norm(candidate - before)
                    prediction = self.base.normalizer.decode_y(
                        self.base.network(xn) + (design @ candidate).clamp(
                            -self.residual_output_clip, self.residual_output_clip))
                    if not torch.isfinite(candidate).all() or not torch.isfinite(prediction).all():
                        log["reason"] = "nonfinite_solution"
                    elif magnitude > self.max_coefficient_norm or update > self.max_coefficient_norm:
                        log["reason"] = "coefficient_or_update_norm_bound"
                    elif prediction.abs().max() > self.max_prediction_magnitude:
                        log["reason"] = "prediction_magnitude_bound"
                    else:
                        self.coefficients.copy_(candidate)
                        log.update(accepted=True, coefficient_norm=float(magnitude),
                                   parameter_update_norm=float(update),
                                   support_loss_before=float((target - design @ before).square().mean()),
                                   support_loss_after=float((target - design @ candidate).square().mean()))
                except torch.linalg.LinAlgError as exc:
                    log["reason"] = f"linear_solve_failed: {exc}"
        if not log["accepted"]:
            self.coefficients.copy_(before)
            log["rolled_back"] = True
        if reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        log["adaptation_seconds"] = time.perf_counter() - started
        self.last_adaptation = log
        self.base.assert_frozen()
        return log

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": self.checkpoint_schema, "degree": self.degree,
                    "regularization": self.regularization,
                    "max_coefficient_norm": self.max_coefficient_norm,
                    "residual_output_clip": self.residual_output_clip,
                    "max_prediction_magnitude": self.max_prediction_magnitude,
                    "coefficients": self.coefficients, "metadata": metadata or {}}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, base: FrozenDynamicsModel,
                        map_location: str | torch.device = "cpu") -> "RidgeResidualModel":
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if checkpoint.get("schema") != cls.checkpoint_schema:
            raise ValueError("Unsupported Ridge checkpoint schema.")
        result = cls(base, regularization=checkpoint["regularization"], degree=checkpoint["degree"],
                     max_coefficient_norm=checkpoint["max_coefficient_norm"],
                     residual_output_clip=checkpoint["residual_output_clip"],
                     max_prediction_magnitude=checkpoint["max_prediction_magnitude"])
        result.coefficients.copy_(checkpoint["coefficients"].to(result.coefficients))
        return result
