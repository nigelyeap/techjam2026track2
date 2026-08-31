"""Section 6e, part 2: calibrate the tuned XGBoost in the current ensemble.

This runner is intentionally separate from ``run_experiment.py``.  It reads
that runner's frozen validation-selected XGBoost configuration, reproduces the
current iter63 FM + LightGBM reference, then compares global-score and
within-user-percentile three-model blends.  Weight and normalization selection
uses validation only; test predictions are first made after the choice and any
five-seed confirmation are frozen.

Run from the repository root after ``run_experiment.py``:

    python3 experiments/iterYIXI5_xgboost_optimization/blend.py
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import platform
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
TUNER_PATH = os.path.join(THIS_DIR, "run_experiment.py")
TUNING_RESULTS_PATH = os.path.join(THIS_DIR, "results.json")
BLEND_RESULTS_PATH = os.path.join(THIS_DIR, "blend_results.json")
ITER63_TRAIN_PATH = os.path.join(
    REPO_ROOT, "experiments", "iter63_decay_tab_rate", "train.py"
)
MAKE_SUBMISSION_PATH = os.path.join(REPO_ROOT, "make_submission.py")

sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402


SEEDS = (0, 1, 2, 3, 4)
CURRENT_LGB_VALID = 0.6716787219047546
CURRENT_FM_VALID = 0.6398779153823853
CURRENT_ENSEMBLE_VALID = 0.6760629415512085
CURRENT_ENSEMBLE_TEST = 0.659552812576294
CURRENT_FM_WEIGHT = 0.14
PRELIMINARY_DELTA = 0.0003
PROMOTION_DELTA = 0.001


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tuner = load_module(TUNER_PATH, "yixi5_tuner_for_blend")
iter63 = load_module(ITER63_TRAIN_PATH, "iter63_train_for_yixi5")


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
    with open(BLEND_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(jsonable(results), f, indent=2, sort_keys=True)
        f.write("\n")


def minmax(scores) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    lo, hi = float(values.min()), float(values.max())
    return (values - lo) / (hi - lo + 1e-12)


def within_user_percentile(scores, user_ids) -> np.ndarray:
    """Average-tie percentile ranks, independently within each user."""
    score_series = pd.Series(np.asarray(scores, dtype=np.float64), copy=False)
    user_series = pd.Series(np.asarray(user_ids), copy=False)
    ranked = score_series.groupby(user_series, sort=False).rank(method="average", pct=True)
    values = ranked.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise AssertionError("non-finite within-user percentile")
    return values


def normalize_components(
    fm_scores,
    lgb_scores,
    xgb_scores,
    users,
    method: str,
) -> dict[str, np.ndarray]:
    if method == "global":
        # FM is already a sigmoid probability.  The two tree-ranker score
        # ranges are normalized independently, matching the established blend.
        return {
            "fm": np.asarray(fm_scores, dtype=np.float64),
            "lgb": minmax(lgb_scores),
            "xgb": minmax(xgb_scores),
        }
    if method == "within_user_percentile":
        return {
            "fm": within_user_percentile(fm_scores, users),
            "lgb": within_user_percentile(lgb_scores, users),
            "xgb": within_user_percentile(xgb_scores, users),
        }
    raise ValueError(method)


def fit_current_lgb_validation():
    """Fit iter63's exact rate_only model without touching test predictions."""
    frames, y, users = iter63.prepare(DATA_DIR, "rate_only")
    Xtr, ytr, utr = iter63._sort_by_user(frames["train"], y["train"], users["train"])
    Xva, yva, uva = iter63._sort_by_user(frames["valid"], y["valid"], users["valid"])
    gtr = np.unique(utr, return_counts=True)[1]
    gva = np.unique(uva, return_counts=True)[1]
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[5],
        num_leaves=2,
        learning_rate=0.10,
        n_estimators=500,
        min_child_samples=200,
        reg_lambda=1.0,
        random_state=0,
        verbosity=-1,
        n_jobs=-1,
        linear_tree=True,
    )
    model.fit(
        Xtr,
        ytr,
        group=gtr,
        eval_set=[(Xva, yva)],
        eval_group=[gva],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    valid_scores = model.predict(frames["valid"])
    metrics = evaluate(users["valid"], y["valid"], valid_scores)
    del Xtr, Xva
    gc.collect()
    return model, valid_scores, metrics, (frames, y, users)


def fit_current_fm_validation(native_y, native_users):
    """Fit the unchanged iter38 five-seed FM ensemble, validation only."""
    submission = load_module(MAKE_SUBMISSION_PATH, "make_submission_for_yixi5")
    splits = submission.load_ext(
        DATA_DIR,
        halflives=submission.HALFLIVES,
        tab_halflives=submission.TAB_HALFLIVES,
    )
    encoded, dim = submission.encode_ext(
        splits,
        feature_set=submission.FEATURES,
        halflives=submission.HALFLIVES,
        tab_halflives=submission.TAB_HALFLIVES,
        alpha=0.5,
        n_buckets=20,
    )
    Xtr, ytr, utr = encoded["train"]
    Xva, yva, uva = encoded["valid"]
    if not np.array_equal(np.asarray(yva), native_y["valid"]):
        raise AssertionError("FM/native validation labels differ")
    if not np.array_equal(np.asarray(uva), np.asarray(native_users["valid"])):
        raise AssertionError("FM/native validation users differ")

    models = []
    seed_scores = []
    for seed in SEEDS:
        print(f"  fitting FM seed={seed}", flush=True)
        model = submission.train_one_fm(
            Xtr, ytr, utr, Xva, yva, uva, splits["train"], dim, seed
        )
        models.append(model)
        seed_scores.append(submission.sigmoid(model.predict(Xva)))
    valid_scores = np.mean(np.stack(seed_scores), axis=0)
    metrics = evaluate(uva, yva, valid_scores)
    return {
        "module": submission,
        "models": models,
        "encoded": encoded,
    }, valid_scores, metrics


def score_weight_grid(components, users, labels, method: str):
    """Coarse 0.10 simplex, then local 0.02 refinement around a 3-model point."""
    records = []
    seen = set()

    def assess(fm_weight: float, lgb_weight: float, xgb_weight: float, stage: str):
        key = (
            round(float(fm_weight), 8),
            round(float(lgb_weight), 8),
            round(float(xgb_weight), 8),
        )
        if key in seen:
            return
        seen.add(key)
        scores = (
            key[0] * components["fm"]
            + key[1] * components["lgb"]
            + key[2] * components["xgb"]
        )
        metrics = evaluate(users, labels, scores)
        records.append(
            {
                "normalization": method,
                "stage": stage,
                "weights": {"fm": key[0], "lgb": key[1], "xgb": key[2]},
                "valid": metric_dict(metrics),
            }
        )

    for fm_units in range(11):
        for xgb_units in range(11 - fm_units):
            lgb_units = 10 - fm_units - xgb_units
            assess(fm_units / 10, lgb_units / 10, xgb_units / 10, "coarse_0.10")

    coarse_three = max(
        (
            row
            for row in records
            if all(row["weights"][name] > 0 for name in ("fm", "lgb", "xgb"))
        ),
        key=lambda row: row["valid"]["primary"],
    )
    center_fm = int(round(coarse_three["weights"]["fm"] * 50))
    center_xgb = int(round(coarse_three["weights"]["xgb"] * 50))
    for fm_units in range(max(1, center_fm - 5), min(49, center_fm + 5) + 1):
        for xgb_units in range(
            max(1, center_xgb - 5),
            min(49 - fm_units, center_xgb + 5) + 1,
        ):
            lgb_units = 50 - fm_units - xgb_units
            if lgb_units >= 1:
                assess(fm_units / 50, lgb_units / 50, xgb_units / 50, "refine_0.02")

    best_overall = max(records, key=lambda row: row["valid"]["primary"])
    best_three = max(
        (
            row
            for row in records
            if all(row["weights"][name] > 0 for name in ("fm", "lgb", "xgb"))
        ),
        key=lambda row: row["valid"]["primary"],
    )
    records.sort(
        key=lambda row: (
            row["stage"], row["weights"]["fm"], row["weights"]["xgb"]
        )
    )
    return best_three, best_overall, records


def combined_scores(components, weights):
    return (
        weights["fm"] * components["fm"]
        + weights["lgb"] * components["lgb"]
        + weights["xgb"] * components["xgb"]
    )


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
    if not os.path.exists(TUNING_RESULTS_PATH):
        raise FileNotFoundError("run run_experiment.py before blend.py")
    with open(TUNING_RESULTS_PATH, encoding="utf-8") as f:
        tuning_results = json.load(f)
    selected_xgb = tuning_results.get("selected_on_validation")
    if not selected_xgb or "config" not in selected_xgb:
        raise RuntimeError("standalone tuning result is incomplete")
    tuned_config = selected_xgb["config"]

    results: dict[str, Any] = {
        "experiment": "iterYIXI5_ensemble_calibration",
        "scope": "current FM + LightGBM reference versus tuned-XGBoost three-model blends",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
        },
        "selection_policy": {
            "selector": "official validation primary only",
            "coarse_grid": 0.10,
            "refined_grid": 0.02,
            "preliminary_delta": PRELIMINARY_DELTA,
            "promotion_delta": PROMOTION_DELTA,
            "promotion_reference": "iter63 14% FM / 86% rate_only LightGBM",
            "test_access": "after normalization, weights, confirmation, and diagnostics are frozen",
        },
        "tuned_xgb_source": {
            "config": tuned_config,
            "standalone_valid": selected_xgb["valid"],
            "standalone_delta_vs_post_6d": selected_xgb["delta_vs_reference_valid"],
            "standalone_verdict": tuning_results["standalone_verdict"],
        },
        "normalization_sweeps": {},
    }
    write_results(results)

    print("=== loading fixed h5 XGBoost representation ===", flush=True)
    xgb_frames, xgb_y, xgb_users, _, xgb_columns = tuner.load_fixed_representation()
    train_order, train_groups = tuner.stable_user_order(xgb_users["train"])
    valid_order, valid_groups = tuner.stable_user_order(xgb_users["valid"])
    orders_groups = (train_order, train_groups, valid_order, valid_groups)
    xgb_model, xgb_valid_scores, xgb_valid_metrics, _ = tuner.fit_config(
        xgb_frames,
        xgb_y,
        xgb_users,
        xgb_columns,
        tuned_config,
        0,
        orders_groups,
    )
    if abs(xgb_valid_metrics["primary"] - selected_xgb["valid"]["primary"]) > 1e-8:
        raise AssertionError("tuned XGBoost validation reproduction drifted")
    print(f"tuned XGBoost reproduction PASSED: {xgb_valid_metrics['primary']:.8f}", flush=True)

    print("\n=== reproducing current iter63 LightGBM validation component ===", flush=True)
    lgb_model, lgb_valid_scores, lgb_valid_metrics, lgb_context = (
        fit_current_lgb_validation()
    )
    lgb_frames, lgb_y, lgb_users = lgb_context
    if abs(lgb_valid_metrics["primary"] - CURRENT_LGB_VALID) > 1e-8:
        raise AssertionError(
            f"current LightGBM drift: {lgb_valid_metrics['primary']} vs {CURRENT_LGB_VALID}"
        )
    if not np.array_equal(lgb_y["valid"], xgb_y["valid"]):
        raise AssertionError("LightGBM/XGBoost validation labels differ")
    if not np.array_equal(np.asarray(lgb_users["valid"]), np.asarray(xgb_users["valid"])):
        raise AssertionError("LightGBM/XGBoost validation users differ")
    print(f"current LightGBM reproduction PASSED: {lgb_valid_metrics['primary']:.8f}", flush=True)

    print("\n=== fitting unchanged FM five-seed ensemble ===", flush=True)
    fm_context, fm_valid_scores, fm_valid_metrics = fit_current_fm_validation(
        xgb_y, xgb_users
    )
    if abs(fm_valid_metrics["primary"] - CURRENT_FM_VALID) > 1e-8:
        raise AssertionError(
            f"current FM drift: {fm_valid_metrics['primary']} vs {CURRENT_FM_VALID}"
        )
    print(f"FM reproduction PASSED: {fm_valid_metrics['primary']:.8f}", flush=True)

    current_valid_scores = (
        CURRENT_FM_WEIGHT * fm_valid_scores
        + (1.0 - CURRENT_FM_WEIGHT) * minmax(lgb_valid_scores)
    )
    current_valid_metrics = evaluate(
        xgb_users["valid"], xgb_y["valid"], current_valid_scores
    )
    # The published iter63 JSON was produced in a slightly different numeric
    # environment.  Its component scores reproduce exactly here; the final
    # blend differs by only 1.6e-6 due to ranking ties, within the instruction's
    # allowed floating-point noise.
    if abs(current_valid_metrics["primary"] - CURRENT_ENSEMBLE_VALID) > 1e-5:
        raise AssertionError(
            f"current ensemble drift: {current_valid_metrics['primary']} "
            f"vs {CURRENT_ENSEMBLE_VALID}"
        )
    print(
        f"current FM + LightGBM reference PASSED: {current_valid_metrics['primary']:.8f}",
        flush=True,
    )
    results["components"] = {
        "tuned_xgb_valid": metric_dict(xgb_valid_metrics),
        "current_lgb_valid": metric_dict(lgb_valid_metrics),
        "current_fm_valid": metric_dict(fm_valid_metrics),
        "current_ensemble_valid": metric_dict(current_valid_metrics),
        "published_current_ensemble_valid": CURRENT_ENSEMBLE_VALID,
        "current_ensemble_weights": {"fm": 0.14, "lgb": 0.86, "xgb": 0.0},
    }
    write_results(results)

    family_winners = []
    validation_components = {}
    for method in ("global", "within_user_percentile"):
        print(f"\n=== {method} validation weight sweep ===", flush=True)
        components = normalize_components(
            fm_valid_scores,
            lgb_valid_scores,
            xgb_valid_scores,
            xgb_users["valid"],
            method,
        )
        best_three, best_overall, records = score_weight_grid(
            components, xgb_users["valid"], xgb_y["valid"], method
        )
        best_three["delta_vs_current_reference"] = float(
            best_three["valid"]["primary"] - current_valid_metrics["primary"]
        )
        results["normalization_sweeps"][method] = {
            "best_three_model": best_three,
            "best_including_ablations": best_overall,
            "records": records,
        }
        family_winners.append(best_three)
        validation_components[method] = components
        write_results(results)
        print(
            f"  best three-model weights={best_three['weights']} "
            f"valid={best_three['valid']['primary']:.8f} "
            f"delta={best_three['delta_vs_current_reference']:+.8f}",
            flush=True,
        )

    selected = max(family_winners, key=lambda row: row["valid"]["primary"])
    selected_method = selected["normalization"]
    selected_weights = selected["weights"]
    selected_valid_scores = combined_scores(
        validation_components[selected_method], selected_weights
    )
    results["selected_on_validation"] = selected
    results["selected_reason"] = (
        "highest official validation primary among the two predeclared normalization families"
    )
    results["diagnostics"] = {
        "selected_blend_ties": tie_stats(selected_valid_scores, xgb_users["valid"]),
        "tuned_xgb_ties": tie_stats(xgb_valid_scores, xgb_users["valid"]),
        "metric_delta_vs_current_reference": {
            key: float(selected["valid"][key] - current_valid_metrics[key])
            for key in ("GAUC", "nDCG@5", "primary")
        },
        "top_nearby_three_model_points": sorted(
            (
                row
                for row in results["normalization_sweeps"][selected_method]["records"]
                if all(row["weights"][name] > 0 for name in ("fm", "lgb", "xgb"))
            ),
            key=lambda row: row["valid"]["primary"],
            reverse=True,
        )[:10],
    }
    write_results(results)
    print(
        f"\nselected normalization={selected_method} weights={selected_weights} "
        f"valid={selected['valid']['primary']:.8f} "
        f"delta={selected['delta_vs_current_reference']:+.8f}",
        flush=True,
    )

    confirmation_rows = []
    if selected["delta_vs_current_reference"] >= PROMOTION_DELTA:
        print("\n=== five-seed fixed-calibration ensemble confirmation ===", flush=True)
        confirmation_rows.append(
            {
                "seed": 0,
                "valid": selected["valid"],
                "delta_vs_current_reference": selected["delta_vs_current_reference"],
            }
        )
        for seed in SEEDS[1:]:
            seed_model, seed_xgb_scores, _, _ = tuner.fit_config(
                xgb_frames,
                xgb_y,
                xgb_users,
                xgb_columns,
                tuned_config,
                seed,
                orders_groups,
            )
            seed_components = normalize_components(
                fm_valid_scores,
                lgb_valid_scores,
                seed_xgb_scores,
                xgb_users["valid"],
                selected_method,
            )
            seed_scores = combined_scores(seed_components, selected_weights)
            seed_metrics = evaluate(xgb_users["valid"], xgb_y["valid"], seed_scores)
            delta = float(seed_metrics["primary"] - current_valid_metrics["primary"])
            confirmation_rows.append(
                {
                    "seed": seed,
                    "valid": metric_dict(seed_metrics),
                    "delta_vs_current_reference": delta,
                }
            )
            print(
                f"  seed={seed} valid={seed_metrics['primary']:.8f} delta={delta:+.8f}",
                flush=True,
            )
            del seed_model, seed_xgb_scores, seed_components, seed_scores
            gc.collect()

    confirmation_deltas = np.asarray(
        [row["delta_vs_current_reference"] for row in confirmation_rows],
        dtype=np.float64,
    )
    ensemble_confirmed = bool(
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
        "confirmed": ensemble_confirmed,
    }
    standalone_confirmed = tuning_results["standalone_verdict"] == "PROMOTE"
    if ensemble_confirmed and standalone_confirmed:
        verdict = "PROMOTE_BOTH"
    elif ensemble_confirmed:
        verdict = "PROMOTE_ENSEMBLE_ONLY"
    elif standalone_confirmed:
        verdict = "PROMOTE_STANDALONE_ONLY"
    else:
        verdict = "REJECT"
    results["verdict"] = verdict
    write_results(results)

    # First test predictions in this runner: all validation choices are frozen.
    print("\n=== FINAL ENSEMBLE TEST EVALUATION (selection frozen) ===", flush=True)
    xgb_test_scores = xgb_model.predict(xgb_frames["test"][xgb_columns])
    lgb_test_scores = lgb_model.predict(lgb_frames["test"])
    Xte_fm, yte_fm, ute_fm = fm_context["encoded"]["test"]
    if not np.array_equal(np.asarray(yte_fm), xgb_y["test"]):
        raise AssertionError("FM/XGBoost test labels differ")
    if not np.array_equal(np.asarray(ute_fm), np.asarray(xgb_users["test"])):
        raise AssertionError("FM/XGBoost test users differ")
    if not np.array_equal(lgb_y["test"], xgb_y["test"]):
        raise AssertionError("LightGBM/XGBoost test labels differ")
    if not np.array_equal(np.asarray(lgb_users["test"]), np.asarray(xgb_users["test"])):
        raise AssertionError("LightGBM/XGBoost test users differ")
    fm_test_scores = np.mean(
        np.stack(
            [
                fm_context["module"].sigmoid(model.predict(Xte_fm))
                for model in fm_context["models"]
            ]
        ),
        axis=0,
    )
    current_test_scores = (
        CURRENT_FM_WEIGHT * fm_test_scores
        + (1.0 - CURRENT_FM_WEIGHT) * minmax(lgb_test_scores)
    )
    current_test_metrics = evaluate(
        xgb_users["test"], xgb_y["test"], current_test_scores
    )
    if abs(current_test_metrics["primary"] - CURRENT_ENSEMBLE_TEST) > 1e-5:
        raise AssertionError(
            f"current test ensemble drift: {current_test_metrics['primary']} "
            f"vs {CURRENT_ENSEMBLE_TEST}"
        )
    test_components = normalize_components(
        fm_test_scores,
        lgb_test_scores,
        xgb_test_scores,
        xgb_users["test"],
        selected_method,
    )
    selected_test_scores = combined_scores(test_components, selected_weights)
    selected_test_metrics = evaluate(
        xgb_users["test"], xgb_y["test"], selected_test_scores
    )
    results["components"]["tuned_xgb_test"] = metric_dict(
        evaluate(xgb_users["test"], xgb_y["test"], xgb_test_scores)
    )
    results["components"]["current_lgb_test"] = metric_dict(
        evaluate(xgb_users["test"], xgb_y["test"], lgb_test_scores)
    )
    results["components"]["current_fm_test"] = metric_dict(
        evaluate(xgb_users["test"], xgb_y["test"], fm_test_scores)
    )
    results["components"]["current_ensemble_test"] = metric_dict(current_test_metrics)
    results["components"]["published_current_ensemble_test"] = CURRENT_ENSEMBLE_TEST
    results["selected_on_validation"]["test"] = metric_dict(selected_test_metrics)
    results["selected_on_validation"]["delta_vs_current_reference_test"] = float(
        selected_test_metrics["primary"] - current_test_metrics["primary"]
    )
    results["test_evaluation"] = {
        "performed": True,
        "timing": "after normalization, fixed weights, confirmation, diagnostics, and verdict",
    }
    write_results(results)
    print(
        f"selected ensemble: valid={selected['valid']['primary']:.8f} "
        f"test={selected_test_metrics['primary']:.8f} verdict={verdict}",
        flush=True,
    )
    print(f"wrote {BLEND_RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
