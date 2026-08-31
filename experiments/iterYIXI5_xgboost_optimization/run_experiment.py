"""Section 6e, part 1: sequential optimization of the promoted XGBoost ranker.

The fixed representation is iterYIXI4's promoted 5-day user-decay replacement.
No rejected 6d feature is present. Hyperparameter phases are deliberately
sequential: each phase is evaluated against the carried validation incumbent,
and a change is carried only when it clears the project's 0.0003 noise gate.

Run from the repository root:

    python3 experiments/iterYIXI5_xgboost_optimization/run_experiment.py
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
from typing import Any

import numpy as np
import xgboost as xgb


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
YIXI2_FEATURES_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI2_feature_depth", "features.py"
)
RESULTS_PATH = os.path.join(THIS_DIR, "results.json")

sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402


REFERENCE_VALID = 0.6664905548095703
REFERENCE_TEST = 0.6519709825515747
PRELIMINARY_DELTA = 0.0003
PROMOTION_DELTA = 0.001
SEEDS = (0, 1, 2, 3, 4)

BASE_CONFIG: dict[str, Any] = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@5-",
    "max_depth": 1,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_weight": 1.0,
    "gamma": 0.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "max_bin": 256,
    "enable_categorical": True,
    "max_cat_to_onehot": 4,
    "early_stopping_rounds": 30,
}

# These compact, ordered grids are fixed before validation is inspected.  They
# refine the 6a winner instead of forming a Cartesian product.
PHASES: tuple[tuple[str, tuple[tuple[str, dict[str, Any]], ...]], ...] = (
    (
        "tree_capacity",
        (("max_depth_2", {"max_depth": 2}),),
    ),
    (
        "boosting_rate_and_trees",
        (
            ("lr_0.10_trees_250", {"learning_rate": 0.10, "n_estimators": 250}),
            ("lr_0.075_trees_400", {"learning_rate": 0.075, "n_estimators": 400}),
            ("lr_0.025_trees_1000", {"learning_rate": 0.025, "n_estimators": 1000}),
            (
                "lr_0.025_trees_1000_patience_120",
                {
                    "learning_rate": 0.025,
                    "n_estimators": 1000,
                    "early_stopping_rounds": 120,
                },
            ),
        ),
    ),
    (
        "min_child_weight",
        (
            ("min_child_0.5", {"min_child_weight": 0.5}),
            ("min_child_2", {"min_child_weight": 2.0}),
            ("min_child_5", {"min_child_weight": 5.0}),
            ("min_child_10", {"min_child_weight": 10.0}),
        ),
    ),
    (
        "gamma",
        (
            ("gamma_0.01", {"gamma": 0.01}),
            ("gamma_0.05", {"gamma": 0.05}),
            ("gamma_0.1", {"gamma": 0.1}),
            ("gamma_0.5", {"gamma": 0.5}),
        ),
    ),
    (
        "reg_lambda",
        (
            ("lambda_0", {"reg_lambda": 0.0}),
            ("lambda_0.25", {"reg_lambda": 0.25}),
            ("lambda_0.5", {"reg_lambda": 0.5}),
            ("lambda_2", {"reg_lambda": 2.0}),
            ("lambda_4", {"reg_lambda": 4.0}),
            ("lambda_10", {"reg_lambda": 10.0}),
        ),
    ),
    (
        "reg_alpha",
        (
            ("alpha_0.01", {"reg_alpha": 0.01}),
            ("alpha_0.1", {"reg_alpha": 0.1}),
            ("alpha_0.5", {"reg_alpha": 0.5}),
            ("alpha_1", {"reg_alpha": 1.0}),
        ),
    ),
    (
        "reg_alpha_refinement",
        (
            ("alpha_0.05_refine", {"reg_alpha": 0.05}),
            ("alpha_0.075_refine", {"reg_alpha": 0.075}),
            ("alpha_0.15_refine", {"reg_alpha": 0.15}),
            ("alpha_0.2_refine", {"reg_alpha": 0.2}),
        ),
    ),
    (
        "row_stochasticity",
        (
            ("subsample_0.9", {"subsample": 0.9}),
            ("subsample_0.8", {"subsample": 0.8}),
        ),
    ),
    (
        "feature_stochasticity",
        (
            ("colsample_0.9", {"colsample_bytree": 0.9}),
            ("colsample_0.8", {"colsample_bytree": 0.8}),
        ),
    ),
)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


yixi2_features = load_module(YIXI2_FEATURES_PATH, "yixi2_features_for_yixi5")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def metric_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: jsonable(value) for key, value in metrics.items()}


def write_results(results: dict[str, Any]) -> None:
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(jsonable(results), f, indent=2, sort_keys=True)
        f.write("\n")


def stable_user_order(user_ids):
    user_ids = np.asarray(user_ids)
    order = np.argsort(user_ids, kind="stable")
    groups = np.unique(user_ids[order], return_counts=True)[1]
    return order, groups


def promoted_h5_columns(base_columns: list[str]) -> list[str]:
    """Exact iterYIXI4 selection: replace the 2.5-day pair with the 5-day pair."""
    columns = [
        column
        for column in base_columns
        if column not in ("decay_rate_2.5", "decay_act_2.5")
    ]
    columns.extend(("decay_rate_5", "decay_act_5"))
    expected = [
        "user_id",
        "video_id",
        "author_id",
        "tab",
        "last1",
        "duration_ms",
        "decay_tab_3",
        "lastk_rate",
        "gap",
        "decay_rate_5",
        "decay_act_5",
    ]
    if columns != expected:
        raise AssertionError(f"unexpected promoted h5 columns: {columns}")
    return columns


def load_fixed_representation():
    """Load the unchanged 6b builder and return only the promoted 6d columns."""
    frames, y, users, metadata = yixi2_features.prepare_feature_frames(DATA_DIR)
    if not metadata["causality"]["passed"]:
        raise AssertionError("imported 6b causal checks failed")
    columns = promoted_h5_columns(list(metadata["base_columns"]))
    return frames, y, users, metadata, columns


def make_ranker(config: dict[str, Any], seed: int) -> xgb.XGBRanker:
    return xgb.XGBRanker(
        **config,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )


def fit_config(
    frames,
    y,
    users,
    columns: list[str],
    config: dict[str, Any],
    seed: int,
    orders_groups=None,
):
    if orders_groups is None:
        train_order, train_groups = stable_user_order(users["train"])
        valid_order, valid_groups = stable_user_order(users["valid"])
    else:
        train_order, train_groups, valid_order, valid_groups = orders_groups
    Xtr = frames["train"][columns].iloc[train_order].reset_index(drop=True)
    ytr = y["train"][train_order]
    Xva = frames["valid"][columns].iloc[valid_order].reset_index(drop=True)
    yva = y["valid"][valid_order]
    model = make_ranker(config, seed)
    model.fit(
        Xtr,
        ytr,
        group=train_groups,
        eval_set=[(Xva, yva)],
        eval_group=[valid_groups],
        verbose=False,
    )
    scores = model.predict(frames["valid"][columns])
    metrics = evaluate(users["valid"], y["valid"], scores)
    raw_gain = model.get_booster().get_score(importance_type="gain")
    gain_total = float(sum(raw_gain.values()))
    gain_fraction = {
        column: (float(raw_gain.get(column, 0.0)) / gain_total if gain_total else 0.0)
        for column in columns
    }
    del Xtr, Xva
    gc.collect()
    return model, scores, metrics, gain_fraction


def record_fit(
    name: str,
    phase: str,
    config: dict[str, Any],
    seed: int,
    model,
    metrics,
    gain_fraction,
) -> dict[str, Any]:
    return {
        "name": name,
        "phase": phase,
        "seed": seed,
        "config": dict(config),
        "best_iteration": int(model.best_iteration),
        "best_internal_ndcg_at_5_minus": float(model.best_score),
        "valid": metric_dict(metrics),
        "feature_gain_fraction": gain_fraction,
    }


def tie_stats(scores, user_ids) -> dict[str, Any]:
    scores = np.asarray(scores)
    user_ids = np.asarray(user_ids)
    fractions = []
    for user_id in np.unique(user_ids):
        user_scores = scores[user_ids == user_id]
        fractions.append(len(np.unique(user_scores)) / len(user_scores))
    return {
        "unique_scores_overall": int(len(np.unique(scores))),
        "rows": int(len(scores)),
        "mean_per_user_unique_fraction": float(np.mean(fractions)),
    }


def main() -> None:
    print("=== loading fixed promoted h5 representation ===", flush=True)
    frames, y, users, feature_metadata, columns = load_fixed_representation()
    print(
        "causal gate PASSED: "
        f"max_error={feature_metadata['causality']['global_max_abs_error']:.3e}; "
        f"fixed_columns={len(columns)}",
        flush=True,
    )
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])
    orders_groups = (train_order, train_groups, valid_order, valid_groups)

    results: dict[str, Any] = {
        "experiment": "iterYIXI5_xgboost_optimization_standalone",
        "scope": "sequential XGBoost parameter tuning only; blend.py owns ensemble calibration",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
        },
        "provenance": {
            "feature_builder": os.path.relpath(YIXI2_FEATURES_PATH, REPO_ROOT),
            "feature_builder_sha256": file_sha256(YIXI2_FEATURES_PATH),
            "fixed_feature_change": {
                "added": ["decay_rate_5", "decay_act_5"],
                "removed": ["decay_rate_2.5", "decay_act_2.5"],
                "columns": columns,
            },
            "rejected_6d_features_included": False,
            "causality": feature_metadata["causality"],
        },
        "selection_policy": {
            "selector": "official validation primary only",
            "phase_carry_gate": PRELIMINARY_DELTA,
            "promotion_delta": PROMOTION_DELTA,
            "phase_order": [name for name, _ in PHASES],
            "test_access": "after all phases, validation selection, confirmation, and diagnostics",
        },
        "reference": {},
        "phases": [],
    }
    write_results(results)

    print("\n=== Stage 0: post-6d XGBoost harness reproduction ===", flush=True)
    ref_model, ref_scores, ref_metrics, ref_gain = fit_config(
        frames, y, users, columns, BASE_CONFIG, 0, orders_groups
    )
    reference = record_fit(
        "post_6d_h5_reference", "reference", BASE_CONFIG, 0,
        ref_model, ref_metrics, ref_gain
    )
    if abs(ref_metrics["primary"] - REFERENCE_VALID) > 1e-8:
        raise AssertionError(
            f"post-6d harness drift: {ref_metrics['primary']} vs {REFERENCE_VALID}"
        )
    results["reference"] = reference
    write_results(results)
    print(f"reference reproduction PASSED: {ref_metrics['primary']:.8f}", flush=True)

    incumbent_config = dict(BASE_CONFIG)
    incumbent_valid = float(ref_metrics["primary"])
    incumbent_name = reference["name"]
    del ref_model, ref_scores
    gc.collect()

    for phase_name, candidates in PHASES:
        print(f"\n=== phase: {phase_name} ===", flush=True)
        phase_start_name = incumbent_name
        phase_start_config = dict(incumbent_config)
        phase_start_valid = incumbent_valid
        candidate_records = []
        for candidate_name, overrides in candidates:
            candidate_config = dict(phase_start_config)
            candidate_config.update(overrides)
            model, scores, metrics, gain_fraction = fit_config(
                frames, y, users, columns, candidate_config, 0, orders_groups
            )
            record = record_fit(
                candidate_name, phase_name, candidate_config, 0,
                model, metrics, gain_fraction
            )
            record["delta_vs_phase_start"] = float(metrics["primary"] - phase_start_valid)
            record["delta_vs_reference"] = float(metrics["primary"] - REFERENCE_VALID)
            candidate_records.append(record)
            print(
                f"  {candidate_name}: valid={metrics['primary']:.8f} "
                f"phase_delta={record['delta_vs_phase_start']:+.8f} "
                f"best_iter={model.best_iteration}",
                flush=True,
            )
            del model, scores
            gc.collect()

        phase_best = max(candidate_records, key=lambda row: row["valid"]["primary"])
        best_gain = float(phase_best["valid"]["primary"] - phase_start_valid)
        carried = best_gain >= PRELIMINARY_DELTA
        if carried:
            incumbent_config = dict(phase_best["config"])
            incumbent_valid = float(phase_best["valid"]["primary"])
            incumbent_name = phase_best["name"]
        phase_record = {
            "name": phase_name,
            "start_incumbent": {
                "name": phase_start_name,
                "config": phase_start_config,
                "valid_primary": phase_start_valid,
            },
            "candidates": candidate_records,
            "raw_winner": phase_best["name"],
            "raw_winner_gain_vs_phase_start": best_gain,
            "carried": carried,
            "decision": "carry validation winner" if carried else "abandon axis as noise",
            "end_incumbent": {
                "name": incumbent_name,
                "config": dict(incumbent_config),
                "valid_primary": incumbent_valid,
            },
        }
        results["phases"].append(phase_record)
        write_results(results)
        print(
            f"  phase decision: {phase_record['decision']} "
            f"(best gain {best_gain:+.8f})",
            flush=True,
        )

    print("\n=== refitting frozen validation winner ===", flush=True)
    selected_model, selected_scores, selected_metrics, selected_gain = fit_config(
        frames, y, users, columns, incumbent_config, 0, orders_groups
    )
    if abs(selected_metrics["primary"] - incumbent_valid) > 1e-10:
        raise AssertionError("frozen seed-0 refit drifted")
    selected = record_fit(
        incumbent_name, "frozen_selection", incumbent_config, 0,
        selected_model, selected_metrics, selected_gain
    )
    selected["delta_vs_reference_valid"] = float(
        selected_metrics["primary"] - REFERENCE_VALID
    )
    results["selected_on_validation"] = selected

    changed_parameters = {
        key: {"reference": BASE_CONFIG[key], "selected": incumbent_config[key]}
        for key in BASE_CONFIG
        if BASE_CONFIG[key] != incumbent_config[key]
    }
    rng = np.random.default_rng(0)
    constant_metrics = evaluate(users["valid"], y["valid"], np.zeros(len(y["valid"])))
    random_metrics = evaluate(
        users["valid"], y["valid"], rng.uniform(size=len(y["valid"]))
    )
    results["artifact_and_confound_checks"] = {
        "exact_feature_columns": columns,
        "feature_width": len(columns),
        "rejected_6d_features_absent": all(
            name not in columns
            for name in (
                "decay_rate_per_duration",
                "decay_rate_x_log_activity",
                "activity_tier_x_tab",
                "author_decay_rate_2.5",
                "video_decay_rate_2.5",
            )
        ),
        "changed_parameters_only": changed_parameters,
        "constant_valid": metric_dict(constant_metrics),
        "random_valid": metric_dict(random_metrics),
        "selected_ties": tie_stats(selected_scores, users["valid"]),
        "metric_delta_vs_reference": {
            key: float(selected_metrics[key] - ref_metrics[key])
            for key in ("GAUC", "nDCG@5", "primary")
        },
    }
    write_results(results)

    confirmation_rows = []
    if selected["delta_vs_reference_valid"] >= PROMOTION_DELTA:
        print("\n=== paired five-seed standalone confirmation ===", flush=True)
        confirmation_rows.append(
            {
                "seed": 0,
                "reference_valid": reference["valid"],
                "candidate_valid": selected["valid"],
                "delta": selected["delta_vs_reference_valid"],
            }
        )
        for seed in SEEDS[1:]:
            baseline_model, baseline_scores, baseline_metrics, _ = fit_config(
                frames, y, users, columns, BASE_CONFIG, seed, orders_groups
            )
            candidate_model, candidate_scores, candidate_metrics, _ = fit_config(
                frames, y, users, columns, incumbent_config, seed, orders_groups
            )
            delta = float(candidate_metrics["primary"] - baseline_metrics["primary"])
            confirmation_rows.append(
                {
                    "seed": seed,
                    "reference_valid": metric_dict(baseline_metrics),
                    "candidate_valid": metric_dict(candidate_metrics),
                    "delta": delta,
                }
            )
            print(
                f"  seed={seed} reference={baseline_metrics['primary']:.8f} "
                f"candidate={candidate_metrics['primary']:.8f} delta={delta:+.8f}",
                flush=True,
            )
            del baseline_model, baseline_scores, candidate_model, candidate_scores
            gc.collect()

    confirmation_deltas = np.asarray(
        [row["delta"] for row in confirmation_rows], dtype=np.float64
    )
    standalone_confirmed = bool(
        len(confirmation_deltas) == len(SEEDS)
        and np.mean(confirmation_deltas) >= PROMOTION_DELTA
        and np.all(confirmation_deltas >= PROMOTION_DELTA)
    )
    results["five_seed_confirmation"] = {
        "performed": bool(confirmation_rows),
        "rows": confirmation_rows,
        "mean_delta": float(np.mean(confirmation_deltas)) if len(confirmation_deltas) else None,
        "std_delta": float(np.std(confirmation_deltas)) if len(confirmation_deltas) else None,
        "min_delta": float(np.min(confirmation_deltas)) if len(confirmation_deltas) else None,
        "confirmed": standalone_confirmed,
    }
    results["standalone_verdict"] = "PROMOTE" if standalone_confirmed else "REJECT"
    write_results(results)

    # Test is first accessed only after the full validation path above is frozen.
    print("\n=== FINAL STANDALONE TEST EVALUATION (selection frozen) ===", flush=True)
    test_scores = selected_model.predict(frames["test"][columns])
    test_metrics = evaluate(users["test"], y["test"], test_scores)
    results["selected_on_validation"]["test"] = metric_dict(test_metrics)
    results["selected_on_validation"]["delta_vs_reference_test"] = float(
        test_metrics["primary"] - REFERENCE_TEST
    )
    results["test_evaluation"] = {
        "performed": True,
        "timing": "after sequential validation selection, confirmation, and diagnostics",
    }
    write_results(results)
    print(
        f"selected standalone: valid={selected_metrics['primary']:.8f} "
        f"test={test_metrics['primary']:.8f} verdict={results['standalone_verdict']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
