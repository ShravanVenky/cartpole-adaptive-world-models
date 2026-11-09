"""Reports and figures from saved metrics only; never trains or adapts models."""
from __future__ import annotations

import json
from itertools import product
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IDENTIFIERS = ["seed", "trial", "environment", "category", "regime", "support_mode", "K"]
METHODS = ("frozen", "ridge", "meta")
COLORS = {"frozen": "#6b7280", "ridge": "#13795b", "meta": "#3358a4"}
BENEFIT_THRESHOLD = 0.10
TRIAL_WIN_THRESHOLD = 0.75
MINIMUM_SEED_WINS = 4
MINIMUM_SEEDS = 5


def _criteria(directory: Path) -> dict[str, Any]:
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    settings = config.get("report", {})
    return {"paired_nmse_benefit": float(settings.get("meta_over_ridge_fractional_margin", BENEFIT_THRESHOLD)),
            "trial_win_fraction": float(settings.get("required_paired_win_fraction", TRIAL_WIN_THRESHOLD)),
            "minimum_seed_wins": int(settings.get("required_seed_wins", MINIMUM_SEED_WINS)),
            "minimum_independent_seeds": max(MINIMUM_SEEDS, int(settings.get("required_seeds", MINIMUM_SEEDS))),
            "meaningful_ood_error_ratio": float(settings.get("meaningful_ood_error_ratio", 2.0)),
            "minimum_meaningful_mismatch_fraction": float(settings.get(
                "minimum_meaningful_mismatch_fraction", settings.get("required_paired_win_fraction", TRIAL_WIN_THRESHOLD))),
            "requires_no_100_step_degradation": True, "main_support": "sequential",
            "main_environments": "non-ID categories", "rejected_nonfinite_pairs_count_as_losses": True}


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _paired(frame: pd.DataFrame, metric: str = "nmse", extra: list[str] | None = None) -> pd.DataFrame:
    keys = IDENTIFIERS + (extra or [])
    if frame.duplicated(keys + ["method"]).any():
        raise ValueError("duplicate method/trial metrics: pairing would be ambiguous")
    return frame.pivot(index=keys, columns="method", values=metric).reset_index()


def _declared_registry(directory: Path) -> tuple[pd.DataFrame, tuple[int, ...]] | None:
    """Expand saved protocol identities without inventing missing fixture config."""
    path = directory / "config.json"
    if not path.exists():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    if not {"seeds", "test_environments", "evaluation"}.issubset(config):
        return None
    evaluation = config["evaluation"]
    required = {"regimes", "trials_per_seed", "support_modes", "K", "rollout_horizons"}
    if not required.issubset(evaluation):
        return None
    rows = []
    for seed, environment, regime, trial, mode, k, method in product(
            config["seeds"], config["test_environments"], evaluation["regimes"],
            range(evaluation["trials_per_seed"]), evaluation["support_modes"], evaluation["K"], METHODS):
        rows.append({"seed": seed, "trial": trial, "environment": environment["name"],
                     "category": environment["category"], "regime": regime,
                     "support_mode": mode, "K": k, "method": method})
    return pd.DataFrame(rows, columns=IDENTIFIERS + ["method"]), tuple(evaluation["rollout_horizons"])


def _completion_registry(directory: Path, status: dict[str, Any], frame: pd.DataFrame,
                         rollouts: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    missing: list[str] = []
    details: dict[str, Any] = {}
    evaluation = status.get("stages", {}).get("evaluate", {})
    for label, data in (("trial", frame), ("rollout", rollouts)):
        field = f"expected_{label}_rows"
        if field in evaluation:
            expected = evaluation[field]
            details[f"runner_{field}"] = expected
            if len(data) != expected:
                missing.append(f"Runner expected {expected} {label} rows; CSV contains {len(data)}")
    registry = _declared_registry(directory)
    if registry is None:
        details["full_saved_config_registry_checked"] = False
        return missing, details
    expected_trials, horizons = registry
    expected_rollouts = expected_trials.merge(pd.DataFrame({"horizon": horizons}), how="cross")
    details.update(full_saved_config_registry_checked=True,
                   expected_trial_rows=len(expected_trials), expected_rollout_rows=len(expected_rollouts))
    for label, data, expected, keys in (
            ("trial", frame, expected_trials, IDENTIFIERS + ["method"]),
            ("rollout", rollouts, expected_rollouts, IDENTIFIERS + ["method", "horizon"])):
        if not set(keys).issubset(data.columns):
            missing.append(f"Declared {label} registry cannot be matched to CSV identifiers")
            continue
        correspondence = expected.merge(data[keys].drop_duplicates(), on=keys, how="outer", indicator=True)
        absent = int((correspondence._merge == "left_only").sum())
        unexpected = int((correspondence._merge == "right_only").sum())
        details[f"missing_declared_{label}_identities"] = absent
        details[f"unexpected_{label}_identities"] = unexpected
        if absent or unexpected:
            missing.append(f"Declared {label} registry: {absent} missing and {unexpected} unexpected identities")
    return missing, details


def _main(frame: pd.DataFrame) -> pd.DataFrame:
    sequential = frame["support_mode"].astype(str).str.lower().isin(("sequential", "seq"))
    category = frame["category"].astype(str).str.lower().str.replace(" ", "_")
    interpolation = category.isin(("id", "id_interpolation", "interpolation", "nominal"))
    return frame[sequential & ~interpolation].copy()


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No completed rows."
    columns = list(frame.columns)
    result = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{item:.5g}" if isinstance(item, float) and math.isfinite(item) else str(item) for item in row]
        result.append("| " + " | ".join(values) + " |")
    return "\n".join(result)


def _line_plot(frame: pd.DataFrame, metric: str, path: Path, ylabel: str,
               x: str = "K", factor: float = 1.0, logarithmic: bool = False) -> None:
    if frame.empty:
        return
    regimes = list(frame["regime"].drop_duplicates())
    figure, axes = plt.subplots(1, len(regimes), figsize=(5.0 * len(regimes), 4.0), squeeze=False)
    for axis, regime in zip(axes[0], regimes):
        subset = frame[frame["regime"] == regime]
        for method in METHODS:
            data = subset[subset["method"] == method]
            if data.empty:
                continue
            summary = data.groupby(x)[metric].agg(["median", lambda column: column.quantile(0.1), lambda column: column.quantile(0.9)])
            summary.columns = ["median", "p10", "p90"]
            axis.plot(summary.index, summary["median"] * factor, marker="o", label=method, color=COLORS[method])
            axis.fill_between(summary.index.to_numpy(dtype=float), summary["p10"].to_numpy(dtype=float) * factor,
                              summary["p90"].to_numpy(dtype=float) * factor, color=COLORS[method], alpha=0.12)
        if logarithmic and bool((subset[metric].dropna() > 0).all()) and not subset[metric].dropna().empty:
            axis.set_yscale("log")
        axis.set_xlabel("Support transitions K" if x == "K" else "Rollout horizon")
        axis.set_ylabel(ylabel)
        axis.set_title(str(regime))
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _comparison_figures(frame: pd.DataFrame, rollouts: pd.DataFrame, figures: Path) -> None:
    for mode in frame["support_mode"].drop_duplicates():
        subset = frame[frame["support_mode"] == mode]
        safe_mode = "".join(char if char.isalnum() else "_" for char in str(mode))
        _line_plot(subset, "nmse", figures / f"prediction_vs_K_{safe_mode}.png", "One-step nominal-scale NMSE", logarithmic=True)
        if "error_reduction" in subset:
            _line_plot(subset, "error_reduction", figures / f"error_reduction_{safe_mode}.png", "Frozen error removed (%)", factor=100.0)
    if not rollouts.empty:
        for support in sorted(rollouts["K"].drop_duplicates()):
            subset = rollouts[(rollouts["K"] == support) & (rollouts["support_mode"] == "sequential")]
            _line_plot(subset, "nmse", figures / f"rollout_K{support}.png", "Open-loop state-scale NMSE", x="horizon", logarithmic=True)
    modes = list(frame["support_mode"].drop_duplicates())
    if len(modes) > 1:
        figure, axis = plt.subplots(figsize=(7, 4))
        for method in METHODS:
            for mode in modes:
                data = frame[(frame["method"] == method) & (frame["support_mode"] == mode)].groupby("K")["nmse"].median()
                axis.plot(data.index, data.values, marker="o", linestyle="-" if mode == "sequential" else "--",
                          color=COLORS[method], label=f"{method}: {mode}")
        axis.set_xlabel("Support transitions K")
        axis.set_ylabel("Median one-step NMSE")
        if bool((frame.nmse.dropna() > 0).all()) and not frame.nmse.dropna().empty:
            axis.set_yscale("log")
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(figures / "sequential_vs_iid.png", dpi=160)
        plt.close(figure)


def _direction(pairs: pd.DataFrame, rollouts: pd.DataFrame, candidate: str,
               comparator: str, criteria: dict[str, Any]) -> dict[str, Any]:
    denominator = pairs[comparator].where(pairs[comparator] > 1e-12)
    benefit = (pairs[comparator] - pairs[candidate]) / denominator
    valid = pairs[candidate].notna() & pairs[comparator].notna()
    for method in (candidate, comparator):
        flag = f"accepted_{method}"
        if flag in pairs:
            valid &= pairs[flag].fillna(False).astype(bool)
    benefit = benefit.where(valid)
    # Missing/nonfinite outcomes count against the win fraction, rather than
    # disappearing from the denominator and rewarding unstable methods.
    wins = benefit >= criteria["paired_nmse_benefit"]
    by_seed = benefit.groupby(pairs["seed"]).median()
    seed_wins = int((by_seed >= criteria["paired_nmse_benefit"]).sum())
    long = rollouts[rollouts["horizon"] == 100] if not rollouts.empty else pd.DataFrame()
    no_long_degradation = False
    long_ratio = None
    if not long.empty:
        paired_long = _paired(_main(long), extra=["horizon"])
        if candidate in paired_long and comparator in paired_long and len(paired_long):
            ratio = paired_long[candidate] / paired_long[comparator].where(paired_long[comparator] > 1e-12)
            if "stable" in long:
                accepted_long = _main(long).copy()
                accepted_long["accepted"] = accepted_long.stable.astype(str).str.lower().isin(("true", "1"))
                flags = _paired(accepted_long, "accepted", extra=["horizon"])
                ratio = ratio.where(flags[candidate].fillna(False).astype(bool)
                                    & flags[comparator].fillna(False).astype(bool))
            long_ratio = float(ratio.median())
            # Require all per-seed medians to avoid hiding one failed training seed.
            seed_ratio = ratio.groupby(paired_long["seed"]).median()
            no_long_degradation = bool(len(seed_ratio) >= criteria["minimum_independent_seeds"] and seed_ratio.notna().all()
                                       and (seed_ratio <= 1.0).all() and ratio.notna().all())
    repair = _repair_evidence(pairs, candidate, criteria)["demonstrated"]
    return {"candidate": candidate, "comparator": comparator,
            "median_paired_nmse_benefit": float(benefit.median()) if benefit.notna().any() else None,
            "trial_win_fraction": float(wins.mean()) if len(wins) else None,
            "rejected_or_nonfinite_comparison_count": int((~valid).sum()),
            "seed_wins": seed_wins, "seed_median_benefits": by_seed.to_dict(),
            "rollout_100_median_error_ratio": long_ratio, "no_100_step_degradation": no_long_degradation,
            "improves_frozen": repair,
            "passes": bool(len(pairs) and wins.mean() >= criteria["trial_win_fraction"] and seed_wins >= criteria["minimum_seed_wins"]
                           and no_long_degradation and repair)}


def _repair_evidence(pairs: pd.DataFrame, method: str, criteria: dict[str, Any]) -> dict[str, Any]:
    """A retained fallback prediction is not evidence that an update repaired it."""
    accepted = pairs[method].notna() & pairs["frozen"].notna()
    flag = f"accepted_{method}"
    if flag in pairs:
        accepted &= pairs[flag].fillna(False).astype(bool)
    reduction = ((pairs["frozen"] - pairs[method])
                 / pairs["frozen"].where(pairs["frozen"] > 1e-12)).where(accepted)
    seed_reductions = reduction.groupby(pairs["seed"]).median()
    accepted_fraction = float(accepted.mean()) if len(accepted) else 0.0
    demonstrated = bool(len(pairs) and accepted_fraction >= criteria["trial_win_fraction"]
                        and reduction.notna().any() and reduction.median() > 0
                        and (seed_reductions > 0).sum() >= criteria["minimum_seed_wins"])
    return {"demonstrated": demonstrated, "accepted_finite_fraction": accepted_fraction,
            "median_frozen_error_removed": float(reduction.median()) if reduction.notna().any() else None,
            "seed_median_frozen_error_removed": seed_reductions.to_dict()}


def _mismatch_evidence(main_rows: pd.DataFrame, criteria: dict[str, Any]) -> dict[str, Any]:
    """Use one frozen row per matched condition; never select favorable methods."""
    frozen = main_rows[main_rows.method == "frozen"]
    if not {"E_before", "E_ID"}.issubset(frozen.columns) or frozen.empty:
        return {"sufficient": False, "reason": "nominal counterfactual error floor missing",
                "meaningful_fraction": None, "seed_meaningful_fractions": {}}
    before = pd.to_numeric(frozen.E_before, errors="coerce")
    floor = pd.to_numeric(frozen.E_ID, errors="coerce")
    gap = before - floor
    valid = np.isfinite(before) & np.isfinite(floor) & (floor >= 0) & (before > 1e-12)
    meaningful = valid & (gap > 1e-12) & (before >= criteria["meaningful_ood_error_ratio"] * floor)
    fractions = meaningful.groupby(frozen.seed).mean()
    minimum = criteria["minimum_meaningful_mismatch_fraction"]
    sufficient = bool(meaningful.mean() >= minimum
                      and (fractions >= minimum).sum() >= criteria["minimum_seed_wins"])
    return {"sufficient": sufficient, "reason": None if sufficient else "frozen error is not consistently above the nominal counterfactual floor",
            "meaningful_fraction": float(meaningful.mean()),
            "positive_gap_fraction": float((valid & (gap > 1e-12)).mean()),
            "seed_meaningful_fractions": fractions.to_dict()}


def generate_report(run_dir: str | Path, status: dict[str, Any]) -> dict[str, Any]:
    """Create a report from CSVs, retaining incomplete status and failed trials."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    frame = _read(directory / "per_trial_metrics.csv")
    rollouts = _read(directory / "rollout_metrics.csv")
    criteria = _criteria(directory)
    summary: dict[str, Any] = {
        "status": status, "complete": False, "verdict": "INCOMPLETE",
        "criteria": criteria,
        "trial_rows": len(frame), "rollout_rows": len(rollouts), "missing": [],
    }
    registry_missing, registry_details = _completion_registry(directory, status, frame, rollouts)
    summary["missing"].extend(registry_missing)
    summary["completion_registry"] = registry_details
    if frame.empty:
        summary["missing"].append("No completed held-out trial metrics")
        pd.DataFrame(columns=IDENTIFIERS + ["method", "nmse"]).to_csv(directory / "per_seed_metrics.csv", index=False)
    else:
        required = set(IDENTIFIERS + ["method", "nmse"])
        if not required.issubset(frame.columns):
            raise ValueError(f"metrics schema missing {sorted(required - set(frame.columns))}")
        frame["nmse"] = pd.to_numeric(frame["nmse"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        summary["nonfinite_trial_rows"] = int(frame.nmse.isna().sum())
        summary["independent_training_seeds"] = sorted(frame["seed"].drop_duplicates().tolist())
        if "stable" in frame:
            stable = frame.stable.astype(str).str.lower().isin(("true", "1"))
            summary["rejected_or_failed_adaptations"] = int((~stable & frame.method.ne("frozen")).sum())
        grouped = frame.groupby(["seed", "environment", "category", "regime", "support_mode", "K", "method"], dropna=False)
        available = [column for column in ("nmse", "error_reduction", "recovery_ratio", "adaptation_seconds",
                                           "rmse_x", "rmse_xdot", "rmse_theta", "rmse_thetadot") if column in frame]
        per_seed = grouped[available].median().reset_index()
        per_seed["trial_count"] = grouped.size().to_numpy()
        per_seed.to_csv(directory / "per_seed_metrics.csv", index=False)
        all_pairs = _paired(frame)
        accepted = frame.copy()
        accepted["accepted"] = accepted.nmse.notna()
        if "stable" in accepted:
            accepted["accepted"] &= accepted.stable.astype(str).str.lower().isin(("true", "1"))
        accepted_pairs = _paired(accepted, "accepted").rename(columns={method: f"accepted_{method}" for method in METHODS})
        all_pairs = all_pairs.merge(accepted_pairs, on=IDENTIFIERS, how="left", validate="one_to_one")
        all_pairs.to_csv(directory / "paired_trial_metrics.csv", index=False)
        method_sets = frame.groupby(IDENTIFIERS, dropna=False)["method"].agg(lambda column: frozenset(column))
        if any(methods != frozenset(METHODS) for methods in method_sets):
            summary["missing"].append("Every matched condition must contain exactly frozen, ridge, and meta")
        if "support_hash" in frame:
            adapters = frame[frame.method.isin(("ridge", "meta"))]
            mismatched = adapters.groupby(IDENTIFIERS, dropna=False)["support_hash"].nunique(dropna=False)
            if bool((mismatched > 1).any()):
                raise AssertionError("Ridge and meta methods received different support observations")
            summary["matched_support_verified"] = True
        else:
            summary["matched_support_verified"] = False
            summary["missing"].append("Matched-support hashes")
        main_pairs = _main(all_pairs)
        main_rows = _main(frame)
        mismatch = _mismatch_evidence(main_rows, criteria)
        summary["mismatch_evidence"] = mismatch
        summary["sequential_ood_positive_frozen_to_id_gap_fraction"] = mismatch.get("positive_gap_fraction")
        summary["sequential_ood_meaningful_mismatch_fraction"] = mismatch["meaningful_fraction"]
        methods_present = set(frame.method)
        if not set(METHODS).issubset(methods_present):
            summary["missing"].append("One or more primary methods")
        if len(summary["independent_training_seeds"]) < criteria["minimum_independent_seeds"]:
            summary["missing"].append("Five independently trained seeds")
        if main_pairs.empty:
            summary["missing"].append("Sequential OOD trial pairs")
        elif {"meta", "ridge", "frozen"}.issubset(main_pairs.columns):
            summary["meta_vs_ridge"] = _direction(main_pairs, rollouts, "meta", "ridge", criteria)
            summary["ridge_vs_meta"] = _direction(main_pairs, rollouts, "ridge", "meta", criteria)
            summary["prediction_repair"] = {method: _repair_evidence(main_pairs, method, criteria)
                                            for method in ("meta", "ridge")}
        if rollouts.empty or not bool((rollouts.horizon == 100).any()):
            summary["missing"].append("100-step held-out rollouts")
        if not rollouts.empty:
            rollout_keys = IDENTIFIERS + ["method", "horizon"]
            if not set(rollout_keys + ["nmse"]).issubset(rollouts.columns):
                raise ValueError("rollout metrics schema does not match trial identifiers")
            if rollouts.duplicated(rollout_keys).any():
                raise ValueError("duplicate rollout method/trial/horizon metrics")
            trial_ids = frame[IDENTIFIERS + ["method"]].drop_duplicates()
            declared = _declared_registry(directory)
            requested_horizons = declared[1] if declared is not None else (5, 20, 50, 100)
            expected = trial_ids.merge(pd.DataFrame({"horizon": requested_horizons}), how="cross")
            correspondence = expected.merge(rollouts[rollout_keys], on=rollout_keys, how="outer", indicator=True)
            if bool((correspondence._merge != "both").any()):
                summary["missing"].append("Full declared-horizon rollout correspondence with every held-out trial row")
        if not status.get("complete", False):
            summary["missing"].append("Runner marked required work incomplete")
        summary["complete"] = not summary["missing"]
        if summary["complete"]:
            meta, ridge = summary.get("meta_vs_ridge", {}), summary.get("ridge_vs_meta", {})
            repair = summary.get("prediction_repair", {})
            if not mismatch["sufficient"]:
                summary["verdict"] = "INSUFFICIENT_MEANINGFUL_MISMATCH"
            elif not any(evidence["demonstrated"] for evidence in repair.values()):
                summary["verdict"] = "PREDICTION_REPAIR_NOT_DEMONSTRATED"
            elif meta.get("passes"):
                summary["verdict"] = "A: Meta outperforms Ridge"
            elif ridge.get("passes"):
                summary["verdict"] = "C: Ridge outperforms Meta"
            else:
                benefit = (main_pairs["ridge"] - main_pairs["meta"]) / main_pairs["ridge"].where(main_pairs["ridge"] > 1e-12)
                accepted_pair = (main_pairs["accepted_meta"].fillna(False).astype(bool)
                                 & main_pairs["accepted_ridge"].fillna(False).astype(bool))
                benefit = benefit.where(accepted_pair)
                seed_medians = benefit.groupby(main_pairs.seed).median()
                near_tie = bool(len(benefit) and (benefit.abs() <= criteria["paired_nmse_benefit"]).mean() >= criteria["trial_win_fraction"]
                                and (seed_medians.abs() <= criteria["paired_nmse_benefit"]).sum() >= criteria["minimum_seed_wins"]
                                and all(evidence["demonstrated"] for evidence in repair.values()))
                summary["verdict"] = "B: Meta approximately Ridge" if near_tie else "Mixed: no consistent winner"
        _comparison_figures(frame, rollouts, figures)
        summary["method_medians"] = frame.groupby("method")[available].median().reset_index().to_dict(orient="records")
    cleaned = _clean(summary)
    (directory / "summary.json").write_text(json.dumps(cleaned, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# CartPole V1 prediction adaptation", "", f"**{summary['verdict']}**", "",
             "This report uses saved held-out metrics. Training seeds are independent fitted models; trials are independent trajectories under each model.", "",
             "The main comparison uses sequential support in non-ID environments. IID is a separate scattered-data reference. All methods use the same six generic observable features; hidden physics is absent from predictor inputs.", "",
             "One-step NMSE uses fixed nominal delta scales. Rollout NMSE uses predeclared state scales and averages steps 1 through the named horizon. Angles wrap only when measuring errors. Recovery ratios remain unclamped and are undefined when the frozen-to-ID gap is nearly zero.", "",
             f"Saved trial rows: {len(frame)}. Saved rollout rows: {len(rollouts)}.", ""]
    if summary["missing"]:
        lines.extend(["Incomplete components:", "", *[f"- {item}" for item in summary["missing"]], ""])
    if "mismatch_evidence" in summary:
        lines.extend([f"Meaningful frozen-model mismatch is present in {summary['mismatch_evidence']['meaningful_fraction']} of matched sequential OOD conditions. A repair conclusion requires the frozen error to be at least {criteria['meaningful_ood_error_ratio']:g} times the nominal-physics counterfactual error in at least {criteria['minimum_meaningful_mismatch_fraction']:.0%} of conditions and at least {criteria['minimum_seed_wins']} training seeds. The counterfactual uses identical visited states/actions; these states need not belong to the nominal input distribution.", ""])
    if summary["verdict"] == "PREDICTION_REPAIR_NOT_DEMONSTRATED":
        lines.extend(["Neither adaptive method consistently demonstrated accepted, finite prediction improvement over frozen across the training seeds. Equal fallback predictions do not demonstrate successful residual repair or validate the low-dimensional residual hypothesis.", ""])
    elif summary["verdict"] == "INSUFFICIENT_MEANINGFUL_MISMATCH":
        lines.extend(["The benchmark did not consistently expose a meaningful frozen-model deficit relative to nominal physics. Saved comparisons remain available, but they do not establish repair of unseen-physics prediction failure.", ""])
    if "method_medians" in summary:
        lines.extend(["Aggregate medians across saved conditions (the paired seed/trial tables retain condition differences):", "",
                      _table(pd.DataFrame(summary["method_medians"])), ""])
    for label in ("meta_vs_ridge", "ridge_vs_meta"):
        if label in summary:
            result = summary[label]
            lines.extend([f"{result['candidate']} versus {result['comparator']}: median paired one-step benefit {result['median_paired_nmse_benefit']}; trial win fraction {result['trial_win_fraction']}; seed wins {result['seed_wins']}; no 100-step degradation {result['no_100_step_degradation']}.", ""])
    lines.extend([f"A consistent winner requires at least {criteria['paired_nmse_benefit']:.0%} paired NMSE benefit in {criteria['trial_win_fraction']:.0%} of trials, benefit in at least {criteria['minimum_seed_wins']} of {criteria['minimum_independent_seeds']} training seeds, actual improvement over frozen prediction, and no 100-step rollout degradation in any seed median. Incomplete work receives no final scientific verdict.", "",
                  "Prediction repair does not establish improved control or successful active diagnosis. CartPole V1 deliberately evaluates passive few-shot prediction adaptation first.", ""])
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return cleaned
