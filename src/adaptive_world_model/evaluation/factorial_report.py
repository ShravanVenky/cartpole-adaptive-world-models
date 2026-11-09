"""Saved-metric reporting for the controlled CartPole V2 factorial experiment.

This module never loads models, collects observations, adapts, or trains. Its
verdict is conditional on a complete declared registry and V1 reproduction.
"""
from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IDENTIFIERS = ["seed", "trial", "environment", "category", "regime", "support_mode", "K"]
METHODS = ("frozen", "ridge", "meta_residual_1step", "meta_residual_rollout",
           "meta_lora_1step", "meta_lora_rollout")
NEURAL = METHODS[2:]
CORE_HORIZONS = (1, 5, 20, 50, 100)
LABELS = {"frozen": "Frozen", "ridge": "Ridge",
          "meta_residual_1step": "MetaResidual-1Step", "meta_residual_rollout": "MetaResidual-Rollout",
          "meta_lora_1step": "MetaLoRA-1Step", "meta_lora_rollout": "MetaLoRA-Rollout"}
COLORS = dict(zip(METHODS, ("#6b7280", "#13795b", "#3358a4", "#6899d3", "#a63ca6", "#e19b25")))
BASELINE_METHODS = ("frozen", "ridge", "meta_residual_1step")
EFFECTS = {
    "objective_residual": ("meta_residual_rollout", "meta_residual_1step"),
    "objective_lora": ("meta_lora_rollout", "meta_lora_1step"),
    "location_one_step": ("meta_lora_1step", "meta_residual_1step"),
    "location_rollout": ("meta_lora_rollout", "meta_residual_rollout"),
}


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _flags(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No completed rows."
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.5g}" if isinstance(value, float) and math.isfinite(value)
                                          else "undefined" if pd.isna(value) else str(value).replace("|", "/")
                                          for value in row) + " |")
    return "\n".join(lines)


def _criteria(config: dict[str, Any]) -> dict[str, Any]:
    supplied = config.get("factorial_report", config.get("report", {}))
    return {"material_benefit": float(supplied.get("material_benefit", supplied.get("material_improvement", 0.10))),
            "positive_trial_fraction": float(supplied.get("positive_trial_fraction", supplied.get("required_win_fraction", 0.60))),
            "ridge_joint_win_fraction": float(supplied.get("ridge_joint_win_fraction", supplied.get("required_win_fraction", 0.60))),
            "one_step_preservation_tolerance": float(supplied.get("one_step_preservation_tolerance", supplied.get("preserve_one_step_tolerance", 0.05))),
            "seed_wins": int(supplied.get("seed_wins", supplied.get("required_seed_wins", 4))),
            "required_seeds": max(5, int(supplied.get("required_seeds", 5))),
            "reproduction_tolerance": float(supplied.get("reproduction_tolerance", 1e-6)),
            "invalid_and_rejected_pairs_count_as_losses": True,
            "main_comparison": "sequential support, non-ID environments",
            "one_step_scale": "frozen nominal delta scale; separate held-out query transitions",
            "rollout_scale": "fixed state scale; mean errors at steps 1..H",
            "no_numerical_failures_required_for_material_claim": True}


def _main(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sequential = frame.support_mode.astype(str).str.lower().isin(("sequential", "seq"))
    category = frame.category.astype(str).str.lower().str.replace(" ", "_")
    return frame[sequential & ~category.isin(("id", "id_interpolation", "interpolation", "nominal"))].copy()


def _prepare(frame: pd.DataFrame, *, rollout: bool = False) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    required = set(IDENTIFIERS + ["method", "nmse", "stable", "update_accepted", "support_hash", "query_hash"])
    if rollout:
        required.update(("horizon", "endpoint_nmse"))
    absent = required - set(frame.columns)
    if absent:
        raise ValueError(f"{'rollout' if rollout else 'trial'} metrics missing {sorted(absent)}")
    keys = IDENTIFIERS + ["method"] + (["horizon"] if rollout else [])
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate matched method/condition metrics would make pairing ambiguous")
    frame = frame.copy()
    for metric in ("nmse", "endpoint_nmse"):
        if metric in frame:
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame["accepted"] = frame.nmse.notna() & frame.nmse.ge(0) & _flags(frame.stable) & _flags(frame.update_accepted)
    if rollout:
        frame["accepted"] &= frame.endpoint_nmse.notna() & frame.endpoint_nmse.ge(0)
    # All six methods must see identical observations, even if frozen ignores support.
    for field in ("support_hash", "query_hash"):
        groups = frame.groupby(IDENTIFIERS, dropna=False)[field]
        if bool((groups.nunique(dropna=False) > 1).any()):
            raise AssertionError(f"Matched methods received different {field} observations")
        if frame[field].isna().any() or frame[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {field} prevents matched-data verification")
    return frame


def _pairs(frame: pd.DataFrame, metric: str = "nmse") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=IDENTIFIERS + list(METHODS))
    return frame.pivot(index=IDENTIFIERS, columns="method", values=metric).reindex(columns=METHODS).reset_index()


def _comparison(frame: pd.DataFrame, candidate: str, comparator: str,
                criteria: dict[str, Any], metric: str = "nmse") -> dict[str, Any]:
    pairs = _pairs(_main(frame), metric)
    flags = _pairs(_main(frame), "accepted")
    denominator = pairs[comparator].where(pairs[comparator] > 1e-12)
    valid = (flags[candidate].fillna(False).astype(bool) & flags[comparator].fillna(False).astype(bool)
             & denominator.notna() & pairs[candidate].notna())
    benefits = ((pairs[comparator] - pairs[candidate]) / denominator).where(valid)
    seeds = benefits.groupby(pairs.seed).median()
    median = float(benefits.median()) if benefits.notna().any() else None
    positive = benefits.gt(0)
    material = benefits.ge(criteria["material_benefit"])
    seed_wins = int(seeds.gt(0).sum())
    return {"candidate": candidate, "comparator": comparator, "metric": metric,
            "matched_conditions": len(pairs), "accepted_finite_pairs": int(valid.sum()),
            "invalid_or_rejected_pairs": int((~valid).sum()), "median_paired_benefit": median,
            "positive_trial_fraction": float(positive.mean()) if len(pairs) else None,
            "material_trial_fraction": float(material.mean()) if len(pairs) else None,
            "seed_median_benefits": seeds.to_dict(), "positive_seed_count": seed_wins,
            "material": bool(len(pairs) and valid.all() and median is not None
                             and median >= criteria["material_benefit"]
                             and positive.mean() >= criteria["positive_trial_fraction"]
                             and len(seeds) >= criteria["required_seeds"] and seed_wins >= criteria["seed_wins"])}


def _preserved(one_step: pd.DataFrame, method: str, criteria: dict[str, Any]) -> dict[str, Any]:
    result = _comparison(one_step, method, "meta_residual_1step", criteria)
    benefit = result["median_paired_benefit"]
    result["preserved"] = bool(result["matched_conditions"] and result["invalid_or_rejected_pairs"] == 0
                               and benefit is not None and benefit >= -criteria["one_step_preservation_tolerance"]
                               and len(result["seed_median_benefits"]) >= criteria["required_seeds"])
    return result


def _joint_ridge(one_step: pd.DataFrame, hundred: pd.DataFrame, method: str,
                 criteria: dict[str, Any]) -> dict[str, Any]:
    tables = []
    for name, rows in (("one_step", one_step), ("rollout_100", hundred)):
        pairs = _pairs(_main(rows))
        accepted = _pairs(_main(rows), "accepted")
        good = (accepted[method].fillna(False).astype(bool) & accepted.ridge.fillna(False).astype(bool)
                & pairs.ridge.gt(1e-12) & pairs[method].notna())
        pairs[name] = ((pairs.ridge - pairs[method]) / pairs.ridge.where(pairs.ridge > 1e-12)).where(good)
        tables.append(pairs[IDENTIFIERS + [name]])
    joined = tables[0].merge(tables[1], on=IDENTIFIERS, how="outer", validate="one_to_one")
    valid = joined[["one_step", "rollout_100"]].notna().all(axis=1)
    wins = joined.one_step.ge(criteria["material_benefit"]) & joined.rollout_100.ge(criteria["material_benefit"])
    seeds = joined.groupby("seed")[["one_step", "rollout_100"]].median()
    seed_wins = int(seeds.ge(criteria["material_benefit"]).all(axis=1).sum())
    return {"candidate": method, "joint_trial_win_fraction": float(wins.mean()) if len(joined) else None,
            "matched_conditions": len(joined), "invalid_or_rejected_pairs": int((~valid).sum()),
            "joint_seed_wins": seed_wins, "seed_median_benefits": seeds.to_dict(orient="index"),
            "passes": bool(len(joined) and valid.all() and wins.mean() >= criteria["ridge_joint_win_fraction"]
                           and len(seeds) >= criteria["required_seeds"] and seed_wins >= criteria["seed_wins"])}


def _interaction(frame: pd.DataFrame) -> dict[str, Any]:
    pairs = _pairs(_main(frame))
    accepted = _pairs(_main(frame), "accepted")
    valid = accepted[list(NEURAL)].fillna(False).astype(bool).all(axis=1) & pairs[list(NEURAL)].gt(1e-12).all(axis=1)
    # Positive: the rollout objective reduces error proportionately more inside LoRA.
    effect = (np.log(pairs.meta_lora_1step / pairs.meta_lora_rollout)
              - np.log(pairs.meta_residual_1step / pairs.meta_residual_rollout)).where(valid)
    seeds = effect.groupby(pairs.seed).median()
    return {"definition": "log(B1/B2) - log(A1/A2) at the same evaluation horizon",
            "positive_interpretation": "larger proportional rollout-objective effect for internal LoRA",
            "matched_conditions": len(pairs), "invalid_or_rejected_pairs": int((~valid).sum()),
            "median_log_interaction": float(effect.median()) if effect.notna().any() else None,
            "positive_trial_fraction": float(effect.gt(0).mean()) if len(effect) else None,
            "seed_median_log_interactions": seeds.to_dict(),
            "interpretation": "paired descriptive interaction; no significance or causal mechanism is claimed"}


def _registry(config: dict[str, Any], rows: pd.DataFrame, rollouts: pd.DataFrame,
              status: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    missing: list[str] = []
    detail: dict[str, Any] = {"declared_registry_checked": False}
    evaluation = config.get("evaluation", {})
    required = {"regimes", "trials_per_seed", "support_modes", "K", "rollout_horizons"}
    if not {"seeds", "test_environments"}.issubset(config) or not required.issubset(evaluation):
        return ["Complete saved seeds/test-environment/evaluation registry is missing"], detail
    horizons = tuple(evaluation["rollout_horizons"])
    if not set(CORE_HORIZONS).issubset(horizons):
        missing.append("Declared rollout horizons do not contain 1, 5, 20, 50, and 100")
    declared = pd.DataFrame([dict(zip(IDENTIFIERS + ["method"],
        (seed, trial, env["name"], env["category"], regime, support, k, method)))
        for seed, env, regime, trial, support, k, method in product(config["seeds"], config["test_environments"],
            evaluation["regimes"], range(evaluation["trials_per_seed"]), evaluation["support_modes"], evaluation["K"], METHODS)])
    detail.update(declared_registry_checked=True, expected_trial_rows=len(declared),
                  expected_rollout_rows=len(declared) * len(horizons))
    keys = IDENTIFIERS + ["method"]
    expected_index = pd.MultiIndex.from_frame(declared[keys])
    for label, frame in (("trial", rows), ("rollout", rollouts)):
        got = frame[keys] if set(keys).issubset(frame.columns) else pd.DataFrame(columns=keys)
        got_index = pd.MultiIndex.from_frame(got)
        if label == "trial":
            absent = len(expected_index.difference(got_index))
            unexpected = len(got_index.difference(expected_index))
        else:
            # Check each horizon using compact identity indexes. Constructing an
            # expected 230k-row object-string Cartesian DataFrame is unnecessary.
            absent = 0
            for horizon in horizons:
                at_horizon = frame.loc[frame.horizon.eq(horizon), keys] if "horizon" in frame else pd.DataFrame(columns=keys)
                absent += len(expected_index.difference(pd.MultiIndex.from_frame(at_horizon)))
            unexpected = int((~got_index.isin(expected_index)).sum())
            if "horizon" in frame:
                unexpected += int((got_index.isin(expected_index) & ~frame.horizon.isin(horizons)).sum())
        detail[f"missing_{label}_identities"] = absent
        detail[f"unexpected_{label}_identities"] = unexpected
        if absent or unexpected:
            missing.append(f"Declared {label} registry: {absent} missing and {unexpected} unexpected identities")
    declared_seeds = set(config["seeds"])
    if len(declared_seeds) != len(config["seeds"]) or len(declared_seeds) < 5:
        missing.append("Five distinct declared training seeds are required")
    stage = status.get("stages", {}).get("evaluate", {})
    if "complete" in stage and not stage["complete"]:
        missing.append("Evaluation stage is incomplete even though top-level status may be stale")
    if config.get("baseline_manifest") and not stage.get("immutable_v1_verified", False):
        missing.append("Runner has not verified the immutable V1 artifact manifest after final evaluation")
    for label, frame in (("trial", rows), ("rollout", rollouts)):
        expected = stage.get(f"expected_{label}_rows")
        if expected is not None and expected != len(frame):
            missing.append(f"Runner expected {expected} {label} rows; CSV contains {len(frame)}")
    return missing, detail


def _reproduction(directory: Path, config: dict[str, Any], criteria: dict[str, Any]) -> dict[str, Any]:
    checks = _read(directory / "baseline_reproduction.csv")
    required = {"seed", "method", "check", "passed"}
    result: dict[str, Any] = {"passed": False, "rows": len(checks), "failures": []}
    if checks.empty or not required.issubset(checks.columns):
        result["failures"].append("V1 baseline reproduction checks are missing")
        return result
    passed = _flags(checks.passed)
    if {"absolute_difference", "tolerance"}.issubset(checks):
        delta = pd.to_numeric(checks.absolute_difference, errors="coerce")
        tolerance = pd.to_numeric(checks.tolerance, errors="coerce")
        quantitative = checks["check"].astype(str).str.contains("metric|prediction|nmse|rmse|rollout", case=False, regex=True)
        valid = np.isfinite(delta) & np.isfinite(tolerance) & tolerance.ge(0) & tolerance.le(criteria["reproduction_tolerance"])
        passed &= ~quantitative | (valid & delta.le(tolerance))
    elif checks["check"].astype(str).str.contains("metric|prediction|nmse|rmse|rollout", case=False, regex=True).any():
        passed[:] = False
        result["failures"].append("Numeric reproduction checks lack absolute differences and tolerances")
    result["failed_rows"] = int((~passed).sum())
    expected = set(product(config.get("seeds", []), BASELINE_METHODS))
    present = set(zip(checks.seed, checks.method))
    absent = expected - present
    if absent:
        result["failures"].append(f"Missing seed/baseline reproduction groups: {sorted(absent)}")
    # Sampling hashes and old metric checks must cover every baseline condition.
    missing_kinds = []
    for seed, method in expected:
        names = checks.loc[checks.seed.eq(seed) & checks.method.eq(method), "check"].astype(str).str.lower()
        if (not names.str.contains("support|trajectory|sampling").any()
                or not names.str.contains("query|trajectory|sampling").any()
                or not names.str.contains("metric|prediction|nmse|rmse").any()):
            missing_kinds.append((seed, method))
    if missing_kinds:
        result["failures"].append(f"Reproduction requires observation and metric checks for {sorted(missing_kinds)}")
    if not set(IDENTIFIERS).issubset(checks.columns):
        result["failures"].append("Full baseline reproduction condition identifiers are missing")
    elif {"seeds", "test_environments", "evaluation"}.issubset(config):
        evaluation = config["evaluation"]
        required_config = {"regimes", "trials_per_seed", "support_modes", "K"}
        if required_config.issubset(evaluation):
            expected_units = pd.DataFrame([dict(zip(IDENTIFIERS + ["method"],
                (seed, trial, env["name"], env["category"], regime, support, k, method)))
                for seed, env, regime, trial, support, k, method in product(config["seeds"], config["test_environments"],
                    evaluation["regimes"], range(evaluation["trials_per_seed"]), evaluation["support_modes"], evaluation["K"], BASELINE_METHODS)])
            expected_index = pd.MultiIndex.from_frame(expected_units[IDENTIFIERS + ["method"]])
            check_names = checks["check"].astype(str).str.lower()
            patterns = {"support observations": "support|trajectory|sampling", "query observations": "query|trajectory|sampling",
                        "one-step metric": "one[_-]?step|(?<!rollout_)prediction|(?<!rollout_)metric"}
            patterns.update({f"rollout {h}": rf"rollout.*(?:h|_|-){h}(?:$|[^0-9])" for h in (5, 20, 50, 100)})
            missing_total = 0
            for label, pattern in patterns.items():
                matched = check_names.str.contains(pattern, regex=True)
                if label.startswith("rollout") and "horizon" in checks:
                    matched |= check_names.str.contains("rollout") & pd.to_numeric(checks.horizon, errors="coerce").eq(int(label.split()[-1]))
                found = pd.MultiIndex.from_frame(checks.loc[matched, IDENTIFIERS + ["method"]])
                absent_count = len(expected_index.difference(found))
                if absent_count:
                    result["failures"].append(f"Missing {label} reproduction checks for {absent_count} baseline conditions")
                    missing_total += absent_count
            result["expected_baseline_conditions"] = len(expected_units)
            result["missing_condition_check_groups"] = missing_total
    result["passed"] = bool(len(checks) and passed.all() and expected and not result["failures"])
    return result


def _growth_diagnostic(rollouts: pd.DataFrame, horizon: int = 100) -> dict[str, Any]:
    """Compare late error growth after matching initial-error strata, descriptively."""
    output: dict[str, Any] = {"interpretation": "observational; not causal proof",
                             "metric": f"endpoint state-scale NMSE at H={horizon} divided by endpoint NMSE at H=1",
                             "comparisons": {}}
    initial = _main(rollouts[rollouts.horizon.eq(1)]) if not rollouts.empty else pd.DataFrame()
    hundred = _main(rollouts[rollouts.horizon.eq(horizon)]) if not rollouts.empty else pd.DataFrame()
    if initial.empty or hundred.empty:
        return output
    merged = initial[IDENTIFIERS + ["method", "endpoint_nmse", "accepted"]].merge(
        hundred[IDENTIFIERS + ["method", "endpoint_nmse", "accepted"]], on=IDENTIFIERS + ["method"],
        suffixes=("_1", "_100"), validate="one_to_one")
    merged["growth"] = (merged.endpoint_nmse_100 / merged.endpoint_nmse_1.where(merged.endpoint_nmse_1 > 1e-12)).where(
        merged.accepted_1 & merged.accepted_100)
    merged["accepted"] = merged.accepted_1 & merged.accepted_100
    raw_fields = [column for column in ("rmse_x", "rmse_xdot", "rmse_theta", "rmse_thetadot")
                  if column in initial and column in hundred]
    if raw_fields:
        raw = initial[IDENTIFIERS + ["method", "accepted"] + raw_fields].merge(
            hundred[IDENTIFIERS + ["method", "accepted"] + raw_fields], on=IDENTIFIERS + ["method"],
            suffixes=("_1", "_100"), validate="one_to_one")
        for field in raw_fields:
            early = pd.to_numeric(raw[f"{field}_1"], errors="coerce")
            late = pd.to_numeric(raw[f"{field}_100"], errors="coerce")
            raw[field] = (late / early.where(early > 1e-12)).where(raw.accepted_1 & raw.accepted_100)
        output["physical_component_rmse_growth"] = {
            "definition": f"physical-unit RMSE averaged through H={horizon} divided by first-rollout-step RMSE",
            "endpoint_vs_cumulative_distinction": "This component diagnostic uses cumulative RMSE; endpoint NMSE above uses the endpoint only.",
            "method_median_ratios": raw.groupby("method")[raw_fields].median().to_dict(orient="index")}
    for name, (candidate, comparator) in EFFECTS.items():
        pair = merged.pivot(index=IDENTIFIERS, columns="method", values=["endpoint_nmse_1", "endpoint_nmse_100", "growth"])
        if candidate not in pair["growth"] or comparator not in pair["growth"]:
            continue
        candidate_initial, comparator_initial = pair["endpoint_nmse_1"][candidate], pair["endpoint_nmse_1"][comparator]
        ratio = candidate_initial / comparator_initial.where(comparator_initial > 1e-12)
        initial_matched = ratio.between(0.8, 1.25)
        good = pair["growth"][candidate].notna() & pair["growth"][comparator].notna()
        growth_benefit = 1 - pair["growth"][candidate] / pair["growth"][comparator].where(pair["growth"][comparator] > 1e-12)
        strata = pd.cut(ratio, [-np.inf, 0.8, 1.25, np.inf], labels=["smaller_initial_error", "comparable_initial_error", "larger_initial_error"])
        per_stratum = {}
        for label in strata.cat.categories:
            values = growth_benefit[good & strata.eq(label)]
            per_stratum[str(label)] = {"count": len(values), "median_growth_benefit": float(values.median()) if len(values) else None}
        values = growth_benefit[good & initial_matched]
        output["comparisons"][name] = {"candidate": candidate, "comparator": comparator,
            "paired_conditions": len(pair), "comparable_initial_error_pairs": int((good & initial_matched).sum()),
            "median_growth_benefit_comparable_initial_error": float(values.median()) if len(values) else None,
            "strata": per_stratum, "ratio_band": [0.8, 1.25]}
    return output


def _plot_lines(rows: pd.DataFrame, path: Path, metric: str, xlabel: str, ylabel: str,
                x: str, regimes: list[str], factor: float = 1) -> None:
    if rows.empty:
        return
    figure, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 4.2), squeeze=False)
    for axis, regime in zip(axes[0], regimes):
        subset = rows[rows.regime.eq(regime)]
        for method in METHODS:
            data = subset[subset.method.eq(method)]
            valid = data.loc[data.accepted, [x, metric]].dropna()
            summary = valid.groupby(x)[metric].median()
            if len(summary):
                axis.plot(summary.index, summary.to_numpy() * factor, label=LABELS[method], color=COLORS[method])
        axis.set_xlabel(xlabel); axis.set_ylabel(ylabel); axis.set_title(str(regime))
        axis.grid(alpha=0.2)
        if metric in ("nmse", "endpoint_nmse") and len(subset) and subset.loc[subset.accepted, metric].dropna().gt(0).all():
            axis.set_yscale("log")
        if axis.lines:
            axis.legend(fontsize=7)
        failures = int((~subset.accepted).sum())
        if failures:
            axis.text(0.02, 0.02, f"{failures} failed/rejected rows; counted as decision losses",
                      transform=axis.transAxes, fontsize=7, color="#aa2222")
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def _figures(one_step: pd.DataFrame, rollouts: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(exist_ok=True)
    main_rollout = _main(rollouts)
    main_one = _main(one_step)
    if not main_rollout.empty:
        regimes = list(main_rollout.regime.drop_duplicates())
        for k in sorted(main_rollout.K.unique()):
            subset = main_rollout[main_rollout.K.eq(k)]
            _plot_lines(subset, directory / f"rollout_error_vs_horizon_K{k}.png", "nmse", "Rollout horizon H",
                        "State-scale NMSE averaged over steps 1..H", "horizon", regimes)
            _plot_lines(subset, directory / f"endpoint_error_vs_horizon_K{k}.png", "endpoint_nmse", "Rollout horizon H",
                        "State-scale NMSE at endpoint H", "horizon", regimes)
        for environment in main_rollout.environment.unique():
            safe = "".join(char if char.isalnum() else "_" for char in str(environment))
            _plot_lines(main_rollout[main_rollout.environment.eq(environment)], directory / f"rollout_shift_{safe}.png",
                        "nmse", "Rollout horizon H", "Median state-scale NMSE; pooled K", "horizon", regimes)
    for label, data in (("one_step", main_one), ("rollout_20", main_rollout[main_rollout.horizon.eq(20)] if not main_rollout.empty else main_rollout),
                        ("rollout_100", main_rollout[main_rollout.horizon.eq(100)] if not main_rollout.empty else main_rollout)):
        if data.empty:
            continue
        keys = IDENTIFIERS + (["horizon"] if "horizon" in data else [])
        frozen = data[data.method.eq("frozen")][keys + ["nmse", "accepted"]].rename(columns={"nmse": "frozen_error", "accepted": "frozen_accepted"})
        joined = data.merge(frozen, on=keys, how="left", validate="many_to_one")
        joined["accepted"] &= joined.frozen_accepted
        joined["error_removed"] = (1 - joined.nmse / joined.frozen_error.where(joined.frozen_error > 1e-12)).where(joined.accepted)
        _plot_lines(joined, directory / f"error_removed_vs_K_{label}.png", "error_removed", "Support transitions K",
                    f"Frozen error removed (%): {label}", "K", list(data.regime.unique()), 100)


def generate_factorial_report(run_dir: str | Path, status: dict[str, Any]) -> dict[str, Any]:
    """Report all six methods; missing work blocks every final scientific verdict."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config_locked.json"
    if not config_path.exists():
        config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    criteria = _criteria(config)
    one_step = _prepare(_read(directory / "per_trial_metrics.csv"))
    rollouts = _prepare(_read(directory / "rollout_metrics.csv"), rollout=True)
    missing, registry = _registry(config, one_step, rollouts, status)
    reproduction = _reproduction(directory, config, criteria)
    if not reproduction["passed"]:
        missing.append("V1 reproduction gate has not passed")
    if not status.get("complete", False):
        missing.append("Runner marked required work incomplete")
    if one_step.empty:
        missing.append("No held-out one-step metrics")
    if rollouts.empty:
        missing.append("No held-out rollout metrics")
    summary: dict[str, Any] = {"complete": not missing, "verdict": "INCOMPLETE", "missing": missing,
        "status": status, "criteria": criteria, "completion_registry": registry, "baseline_reproduction": reproduction,
        "trial_rows": len(one_step), "rollout_rows": len(rollouts), "effects": {}, "ridge_joint_comparisons": {},
        "one_step_preservation": {}, "outcomes": {}, "questions": {}, "control": "Not evaluated",
        "configuration_source": config_path.name}
    if not one_step.empty:
        main_one = _main(one_step)
        main_long = _main(rollouts[rollouts.horizon.eq(100)]) if not rollouts.empty else pd.DataFrame()
        summary["numerical_stability"] = {"nonfinite_trial_rows": int(one_step.nmse.isna().sum()),
            "rejected_trial_updates": int((~_flags(one_step.update_accepted) & one_step.method.ne("frozen")).sum()),
            "unstable_trial_rows": int((~_flags(one_step.stable)).sum()),
            "invalid_rollout_rows": int((~rollouts.accepted).sum()) if not rollouts.empty else 0,
            "failures_per_method": one_step.groupby("method").accepted.apply(lambda x: int((~x).sum())).to_dict()}
        fields = [name for name in ("nmse", "recovery_ratio", "error_reduction", "adaptation_seconds", "steps",
            "adaptive_parameters", "base_parameters", "adaptive_percentage", "rmse_x", "rmse_xdot", "rmse_theta", "rmse_thetadot") if name in one_step]
        # Descriptive aggregates retain failed-row counts alongside finite medians.
        for filename, grouping in (("per_seed_metrics.csv", ["seed", "environment", "category", "regime", "support_mode", "K", "method"]),
                                    ("per_shift_metrics.csv", ["environment", "category", "regime", "support_mode", "K", "method"])):
            grouped = one_step.groupby(grouping, dropna=False)
            table = grouped[fields].median().reset_index()
            table["row_count"] = grouped.size().to_numpy()
            table["invalid_or_rejected_rows"] = grouped.accepted.apply(lambda x: int((~x).sum())).to_numpy()
            table.to_csv(directory / filename, index=False)
        if not rollouts.empty:
            rollout_fields = [name for name in ("nmse", "endpoint_nmse", "rmse_x", "rmse_xdot", "rmse_theta", "rmse_thetadot") if name in rollouts]
            for filename, grouping in (("per_seed_rollout_metrics.csv", ["seed", "environment", "category", "regime", "support_mode", "K", "method", "horizon"]),
                                        ("per_shift_rollout_metrics.csv", ["environment", "category", "regime", "support_mode", "K", "method", "horizon"])):
                grouped = rollouts.groupby(grouping, dropna=False)
                table = grouped[rollout_fields].median().reset_index()
                table["row_count"] = grouped.size().to_numpy()
                table["invalid_or_rejected_rows"] = grouped.accepted.apply(lambda x: int((~x).sum())).to_numpy()
                table.to_csv(directory / filename, index=False)
        summary["one_step_method_medians"] = main_one.groupby("method")[fields].median().reset_index().to_dict(orient="records")
        pairs = _pairs(main_one); pairs.to_csv(directory / "paired_trial_metrics.csv", index=False)
        if not main_long.empty:
            for name, (candidate, comparator) in EFFECTS.items():
                summary["effects"][name] = {"one_step": _comparison(one_step, candidate, comparator, criteria),
                    "rollout_100": _comparison(main_long, candidate, comparator, criteria)}
            for method in NEURAL:
                summary["one_step_preservation"][method] = _preserved(one_step, method, criteria)
                summary["ridge_joint_comparisons"][method] = _joint_ridge(one_step, main_long, method, criteria)
            summary["factorial_interaction"] = {"one_step": _interaction(one_step), "rollout_100": _interaction(main_long)}
            summary["combination_comparisons"] = {method: {"one_step": _comparison(one_step, "meta_lora_rollout", method, criteria),
                "rollout_100": _comparison(main_long, "meta_lora_rollout", method, criteria)} for method in ("ridge", "meta_residual_1step", "meta_residual_rollout", "meta_lora_1step")}
            summary["dynamics_growth_diagnostic"] = _growth_diagnostic(rollouts)
            summary["dynamics_growth_diagnostic_20"] = _growth_diagnostic(rollouts, 20)
            efficiency = main_one.groupby(["K", "method"])[fields].median().reset_index()
            for horizon in (20, 100):
                subset = _main(rollouts[rollouts.horizon.eq(horizon)])
                metric = _pairs(subset)
                recovery = metric[IDENTIFIERS].copy()
                flags = _pairs(subset, "accepted")
                for method in METHODS:
                    valid = flags[method].fillna(False).astype(bool) & flags.frozen.fillna(False).astype(bool)
                    recovery[method] = (1 - metric[method] / metric.frozen.where(metric.frozen > 1e-12)).where(valid)
                melted = recovery.melt(id_vars=IDENTIFIERS, value_vars=METHODS, var_name="method", value_name=f"error_removed_{horizon}")
                efficiency = efficiency.merge(melted.groupby(["K", "method"])[f"error_removed_{horizon}"].median().reset_index(), on=["K", "method"], how="left")
            efficiency.to_csv(directory / "efficiency.csv", index=False)
            summary["efficiency"] = efficiency.to_dict(orient="records")
            paired_rows = []
            for label, item in summary["effects"].items():
                for metric, result in item.items():
                    paired_rows.append({"effect": label, "evaluation": metric, **{key: value for key, value in result.items() if key != "seed_median_benefits"}})
            pd.DataFrame(paired_rows).to_csv(directory / "paired_comparisons.csv", index=False)
            summary["shift_effects"] = {}
            for environment in main_long.environment.unique():
                subset = main_long[main_long.environment.eq(environment)]
                summary["shift_effects"][str(environment)] = {name: _comparison(subset, *methods, criteria) for name, methods in EFFECTS.items()}
            effect = summary["effects"]
            residual_good = effect["objective_residual"]["rollout_100"]["material"] and summary["one_step_preservation"]["meta_residual_rollout"]["preserved"]
            location_good = any(effect[name]["rollout_100"]["material"] for name in ("location_one_step", "location_rollout"))
            combination_good = (effect["objective_lora"]["rollout_100"]["material"]
                                and effect["location_rollout"]["rollout_100"]["material"]
                                and summary["one_step_preservation"]["meta_lora_rollout"]["preserved"])
            ridge_winners = [method for method, comparison in summary["ridge_joint_comparisons"].items() if comparison["passes"]]
            ridge_long = {method: _comparison(main_long, "ridge", method, criteria) for method in NEURAL}
            summary["ridge_long_comparisons"] = ridge_long
            none_improved = not any(item["rollout_100"]["material"] for item in effect.values())
            summary["outcomes"] = {
                "A": {"supported": residual_good, "interpretation": "Rollout-aware residual improves long horizons and preserves one-step adaptation"},
                "B": {"supported": location_good, "interpretation": "Internal LoRA improves long horizons relative to output residual at a matched objective"},
                "C": {"supported": combination_good, "interpretation": "Both rollout objective and internal adaptation contribute"},
                "D": {"supported": all(item["material"] for item in ridge_long.values()), "interpretation": "Ridge consistently beats all neural methods at 100 steps"},
                "E": {"supported": False, "interpretation": "Shift-specific effects require per-shift evidence, not an aggregate universal winner"},
                "F": {"supported": none_improved, "interpretation": "No material improvement of the tested factorial effects"}}
            shift_effects = {label: [values[label] for values in summary["shift_effects"].values()] for label in EFFECTS}
            summary["outcomes"]["E"]["supported"] = any(
                any(value["material"] for value in values)
                and any(value["median_paired_benefit"] is not None and value["median_paired_benefit"] < 0 for value in values)
                for values in shift_effects.values())
            if summary["complete"]:
                supported = [letter for letter, item in summary["outcomes"].items() if item["supported"]]
                summary["verdict"] = "Supported outcomes: " + ", ".join(supported) if supported else "Mixed: no predeclared consistent effect"
                summary["joint_ridge_winners"] = ridge_winners
            else:
                for item in summary["outcomes"].values():
                    item["provisional_descriptive_support"] = item["supported"]
                    item["supported"] = False
        _figures(one_step, rollouts, directory / "figures")
    # Direct answers stay conditional when the registry or baseline gate is incomplete.
    effects = summary["effects"]
    if summary["complete"] and effects:
        show = lambda item: f"median paired benefit {item['median_paired_benefit']:.1%}; positive in {item['positive_trial_fraction']:.1%} of trials; {item['positive_seed_count']}/5 positive seed medians" if item["median_paired_benefit"] is not None else "no valid paired estimate"
        summary["questions"] = {
            "Q1": "Rollout objective: residual " + show(effects["objective_residual"]["rollout_100"]) + "; LoRA " + show(effects["objective_lora"]["rollout_100"]),
            "Q2": "Internal location: one-step objective " + show(effects["location_one_step"]["rollout_100"]) + "; rollout objective " + show(effects["location_rollout"]["rollout_100"]),
            "Q3": "LoRA one-step adaptation: " + show(effects["location_one_step"]["one_step"]),
            "Q4": f"One-step preserved within the predeclared {criteria['one_step_preservation_tolerance']:.0%} median tolerance: " + ", ".join(f"{LABELS[m]}={item['preserved']}" for m, item in summary["one_step_preservation"].items()),
            "Q5": "Both-factor contribution plus one-step preservation criterion: " + str(summary["outcomes"]["C"]["supported"]) + ". Pairwise comparisons with all other adaptive methods are in combination_comparisons; no universal per-case winner is assumed.",
            "Q6": "Consistent joint one-step/100-step Ridge winners: " + (", ".join(summary.get("joint_ridge_winners", [])) or "none"),
            "Q7": "See per_shift_metrics.csv and shift_effects in summary.json; signs vary by physical shift if outcome E is supported.",
            "Q8": "Compare one-step preservation and the matched 100-step effects; these have distinct scales and are not directly subtracted.",
            "Q9": "Endpoint error-growth ratios and comparable initial-error strata at 20 and 100 steps are saved in dynamics_growth_diagnostic_20 and dynamics_growth_diagnostic. A lower late-error/first-error ratio within the comparable initial-error band supports improved recursive behavior beyond a smaller initial error; it remains observational evidence, not proof of a causal dynamics mechanism.",
            "Q10": "Each effect includes all five seed medians; material claims require at least four positive seed medians and no invalid/rejected paired outcomes."}
    else:
        summary["questions"] = {f"Q{number}": "Unanswered: final evaluation/reproduction is incomplete; provisional saved numbers are not a scientific conclusion." for number in range(1, 11)}
    if not summary["complete"]:
        summary["recommended_adapter"] = "Undetermined: incomplete predictive validation"
        summary["recommendation"] = "Complete baseline reproduction and the locked five-seed predictive experiment before selecting an adapter or starting active probing."
    elif summary.get("joint_ridge_winners"):
        winners = summary["joint_ridge_winners"]
        summary["recommended_adapter"] = winners[0] if len(winners) == 1 else "No unique winner: " + ", ".join(winners)
        summary["recommendation"] = "A neural method passes the joint predictive target. Retain the paired shift/seed evidence and test the common controller before claiming task utility."
    elif summary["outcomes"].get("E", {}).get("supported"):
        summary["recommended_adapter"] = "No universal neural replacement; retain Ridge and compare shift-specific strengths"
        summary["recommendation"] = "Investigate the observed shift-specific division of strengths before choosing a hybrid; no hybrid has been validated here."
    elif summary["outcomes"].get("D", {}).get("supported") or summary["outcomes"].get("F", {}).get("supported"):
        summary["recommended_adapter"] = "Retain the V1 references; a new neural replacement is not justified"
        summary["recommendation"] = "Reconsider neural adaptation at CartPole complexity before adding more neural machinery or moving to Walker."
    else:
        summary["recommended_adapter"] = "Undetermined: no consistent predictive winner"
        summary["recommendation"] = "Predictive evidence is mixed; continue a focused adapter investigation before active diagnosis."
    cleaned = _clean(summary)
    (directory / "summary.json").write_text(json.dumps(cleaned, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# CartPole V2: adaptation location and training objective", "", f"**{summary['verdict']}**", "",
        "This report reads saved held-out metrics only. All six methods share the same V1 frozen checkpoints, features, test distributions, support observations, and untouched query trajectories. Training seeds are independent fitted initializations; trials are independent support/query draws.", "",
        "Frozen CartPole V1 remains immutable. Its sequential OOD headline is 36% versus 25% one-step frozen error removed (Meta versus Ridge), and 15% versus 28% at 100 steps. New paired baseline reproduction is required before interpreting changes.", "",
        "One-step NMSE is measured on independent query transitions using nominal delta scales. Rollout NMSE uses fixed state scales and averages errors over steps 1..H. The H=1 point on a rollout curve is not the separate one-step query metric. Angular errors wrap; recursive predictions use predicted states and the recorded action sequence.", "",
        f"Saved trial rows: {len(one_step)}. Saved rollout rows: {len(rollouts)}. V1 reproduction passed: {reproduction['passed']}.", ""]
    if missing:
        lines += ["Incomplete components:", "", *[f"- {item}" for item in missing], ""]
    lines += ["New methods: output residual × one-step/rollout objective, and internal LoRA × one-step/rollout objective. Frozen and generic Ridge are unchanged references. Inner updates use one-step support loss in every neural condition.", "",
              "Locked development configuration:", "", "```json", json.dumps(_clean({key: config.get(key) for key in ("rollout_loss", "lora", "adaptation", "development_selection")}), indent=2), "```", "",
              "Parameter efficiency and recovery (main sequential OOD medians):", "", _table(pd.DataFrame(summary.get("efficiency", []))), ""]
    if not one_step.empty:
        lines += ["One-step results, using the nominal delta scale:", "", _table(pd.DataFrame(summary.get("one_step_method_medians", []))), ""]
    if not rollouts.empty:
        for horizon in CORE_HORIZONS:
            subset = _main(rollouts[rollouts.horizon.eq(horizon)])
            fields = [name for name in ("nmse", "endpoint_nmse", "rmse_x", "rmse_xdot", "rmse_theta", "rmse_thetadot") if name in subset]
            table = subset.groupby("method")[fields].median().reset_index() if not subset.empty else pd.DataFrame()
            lines += [f"{horizon}-step rollout results (state scale; mean through H and endpoint):", "", _table(table), ""]
    lines += ["K-shot curves and rollout-error growth plots are in figures/. per_shift_metrics.csv and per_shift_rollout_metrics.csv preserve every physical shift and regime; per_seed_metrics.csv and per_seed_rollout_metrics.csv preserve independent seeds. Numerical failures and rejected updates remain in paired win denominators, rather than disappearing from comparisons.", "",
              "Numerical stability:", "", "```json", json.dumps(_clean(summary.get("numerical_stability", {})), indent=2), "```", "",
              f"Predeclared targets: material 100-step benefit requires a median paired NMSE reduction of at least {criteria['material_benefit']:.0%}, positive benefit in at least {criteria['positive_trial_fraction']:.0%} of matched trials, at least {criteria['seed_wins']} of {criteria['required_seeds']} positive seed medians, and no failed/rejected pairs. One-step preservation allows at most {criteria['one_step_preservation_tolerance']:.0%} median degradation versus the unchanged MetaResidual-1Step. A joint Ridge win requires at least {criteria['material_benefit']:.0%} benefit at both one-step and 100-step in at least {criteria['ridge_joint_win_fraction']:.0%} of the same paired trials, with {criteria['seed_wins']} of {criteria['required_seeds']} seed medians meeting both margins.", "",
              "Outcomes A–F (multiple findings can coexist):", "", _table(pd.DataFrame([{"outcome": key, **item} for key, item in summary["outcomes"].items()])), "",
              "Answers to the primary questions:", ""]
    lines += [f"- **{question}**: {answer}" for question, answer in summary["questions"].items()]
    lines += ["", "Control remains untested. Prediction improvements do not establish lower task cost or successful active diagnosis.", "",
              "Adapter for the later active-diagnosis architecture: " + summary["recommended_adapter"], "",
              "Recommendation: " + summary["recommendation"], ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return cleaned
