"""Section 6a, part 2: FM + LightGBM + XGBoost score blend.

This follows iter44's train.py/blend.py separation. The standalone XGBoost
configuration is defined in ``run_experiment.py``; this file retrains that
fixed winner with the two published reference components, selects weights on
validation only, confirms the fixed weights across five XGBoost seeds, and
evaluates the frozen blend on test once.

The three-model blend produced here is the promoted 6a result. Standalone
XGBoost is explicitly rejected as a replacement for iter44 LightGBM.

Run from the repository root:

    python3 experiments/iterYIXI1_xgboost_native/blend.py
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
import xgboost as xgb


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
BLEND_RESULTS_PATH = os.path.join(THIS_DIR, "blend_results.json")
STANDALONE_RUNNER_PATH = os.path.join(THIS_DIR, "run_experiment.py")

sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402


SEEDS = (0, 1, 2, 3, 4)
XGB_OBJECTIVE = "rank:ndcg"
XGB_MAX_DEPTH = 1
XGB_REFERENCE_VALID = 0.6586387157440186
LGB_REFERENCE_VALID = 0.6613548994064331
FM_REFERENCE_VALID = 0.6398779153823853
CURRENT_BLEND_REFERENCE_VALID = 0.6647330522537231
PROMOTION_DELTA = 0.001


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


standalone = load_module(STANDALONE_RUNNER_PATH, "yixi1_standalone_for_blend")


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


def minmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    lo, hi = float(scores.min()), float(scores.max())
    return (scores - lo) / (hi - lo + 1e-12)


def fit_lgb_reference(dfs, y, users):
    Xtr, ytr, gtr = standalone.grouped_view(dfs["train"], y["train"], users["train"])
    Xva, yva, gva = standalone.grouped_view(dfs["valid"], y["valid"], users["valid"])
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[5],
        num_leaves=2,
        learning_rate=0.05,
        n_estimators=500,
        min_child_samples=200,
        reg_lambda=1.0,
        random_state=0,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(
        Xtr,
        ytr,
        group=gtr,
        eval_set=[(Xva, yva)],
        eval_group=[gva],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    valid_scores = model.predict(dfs["valid"])
    valid_metrics = evaluate(users["valid"], y["valid"], valid_scores)
    return model, valid_scores, valid_metrics


def fit_fm_reference(y, users):
    submission_ref = load_module(
        os.path.join(REPO_ROOT, "make_submission.py"), "make_submission_for_yixi1_blend"
    )
    splits = submission_ref.load_ext(
        DATA_DIR,
        halflives=submission_ref.HALFLIVES,
        tab_halflives=submission_ref.TAB_HALFLIVES,
    )
    encoded, dim = submission_ref.encode_ext(
        splits,
        feature_set=submission_ref.FEATURES,
        halflives=submission_ref.HALFLIVES,
        tab_halflives=submission_ref.TAB_HALFLIVES,
        alpha=0.5,
        n_buckets=20,
    )
    Xtr, ytr, utr = encoded["train"]
    Xva, yva, uva = encoded["valid"]
    assert np.array_equal(np.asarray(yva), y["valid"]), "FM/native valid labels differ"
    assert np.array_equal(
        np.asarray(uva), np.asarray(users["valid"])
    ), "FM/native valid users differ"

    models = []
    valid_seed_scores = []
    for seed in SEEDS:
        print(f"  fitting FM reference seed={seed}", flush=True)
        model = submission_ref.train_one_fm(
            Xtr, ytr, utr, Xva, yva, uva, splits["train"], dim, seed
        )
        models.append(model)
        valid_seed_scores.append(submission_ref.sigmoid(model.predict(Xva)))
    valid_scores = np.mean(np.stack(valid_seed_scores), axis=0)
    valid_metrics = evaluate(uva, yva, valid_scores)
    context = {
        "submission_ref": submission_ref,
        "models": models,
        "encoded": encoded,
    }
    return context, valid_scores, valid_metrics


def score_weight_grid(fm_scores, lgb_scores, xgb_scores, users, labels):
    components = {
        "fm": np.asarray(fm_scores, dtype=np.float64),
        "lgb": minmax(lgb_scores),
        "xgb": minmax(xgb_scores),
    }
    seen = set()
    records = []
    best = None

    def assess(fm_weight: float, lgb_weight: float, xgb_weight: float, stage: str):
        nonlocal best
        key = (round(fm_weight, 8), round(lgb_weight, 8), round(xgb_weight, 8))
        if key in seen:
            return
        seen.add(key)
        scores = (
            key[0] * components["fm"]
            + key[1] * components["lgb"]
            + key[2] * components["xgb"]
        )
        metrics = evaluate(users, labels, scores)
        record = {
            "stage": stage,
            "weights": {"fm": key[0], "lgb": key[1], "xgb": key[2]},
            "valid": metric_dict(metrics),
        }
        records.append(record)
        if best is None or metrics["primary"] > best["valid"]["primary"] + 1e-12:
            best = record

    for fm_units in range(11):
        for xgb_units in range(11 - fm_units):
            lgb_units = 10 - fm_units - xgb_units
            assess(fm_units / 10, lgb_units / 10, xgb_units / 10, "coarse_0.10")

    assert best is not None
    coarse_weights = best["weights"].copy()
    coarse_fm_units = int(round(coarse_weights["fm"] * 50))
    coarse_xgb_units = int(round(coarse_weights["xgb"] * 50))
    for fm_units in range(max(0, coarse_fm_units - 5), min(50, coarse_fm_units + 5) + 1):
        for xgb_units in range(
            max(0, coarse_xgb_units - 5),
            min(50 - fm_units, coarse_xgb_units + 5) + 1,
        ):
            lgb_units = 50 - fm_units - xgb_units
            if lgb_units >= 0:
                assess(fm_units / 50, lgb_units / 50, xgb_units / 50, "refine_0.02")

    records.sort(
        key=lambda record: (
            record["stage"],
            record["weights"]["fm"],
            record["weights"]["xgb"],
        )
    )
    return best, records, components


def fit_fixed_xgb(dfs, y, users, seed: int):
    Xtr, ytr, gtr = standalone.grouped_view(dfs["train"], y["train"], users["train"])
    Xva, yva, gva = standalone.grouped_view(dfs["valid"], y["valid"], users["valid"])
    return standalone.fit_xgb_candidate(
        XGB_OBJECTIVE,
        XGB_MAX_DEPTH,
        seed,
        Xtr,
        ytr,
        gtr,
        Xva,
        yva,
        gva,
        dfs["valid"],
        users["valid"],
        y["valid"],
    )


def main() -> None:
    print("=== loading fixed 6a component representations ===", flush=True)
    dfs, y, users = standalone.iter44.prepare(DATA_DIR)
    assert list(dfs["train"].columns) == standalone.iter44.CAT_COLS + standalone.iter44.NUM_COLS
    results: dict[str, Any] = {
        "experiment": "iterYIXI1_three_model_blend",
        "scope": "FM + native LightGBM + native XGBoost combination only",
        "promotion_target": "three_model_blend",
        "standalone_xgb_verdict": "REJECT as an iter44 LightGBM replacement",
        "selection_policy": {
            "selector": "official validation primary only",
            "test_access": "after fixed component configuration and blend weights are frozen",
            "promotion_reference": "current 10% FM / 90% LightGBM blend",
            "promotion_delta": PROMOTION_DELTA,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
            "lightgbm": lgb.__version__,
        },
        "fixed_xgb": {
            "objective": XGB_OBJECTIVE,
            "max_depth": XGB_MAX_DEPTH,
            "source": "run_experiment.py validation winner",
        },
    }
    write_results(results)

    print("\n=== fitting fixed standalone components on validation ===", flush=True)
    xgb_model, xgb_valid_scores, xgb_valid_metrics, xgb_record = fit_fixed_xgb(
        dfs, y, users, seed=0
    )
    if abs(xgb_valid_metrics["primary"] - XGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError(
            f"standalone XGBoost drift: {xgb_valid_metrics['primary']} vs {XGB_REFERENCE_VALID}"
        )
    print(f"  XGBoost reference valid={xgb_valid_metrics['primary']:.5f}", flush=True)

    lgb_model, lgb_valid_scores, lgb_valid_metrics = fit_lgb_reference(dfs, y, users)
    if abs(lgb_valid_metrics["primary"] - LGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError(
            f"LightGBM reference drift: {lgb_valid_metrics['primary']} vs {LGB_REFERENCE_VALID}"
        )
    print(f"  LightGBM reference valid={lgb_valid_metrics['primary']:.5f}", flush=True)
    fm_context, fm_valid_scores, fm_valid_metrics = fit_fm_reference(y, users)
    if abs(fm_valid_metrics["primary"] - FM_REFERENCE_VALID) > 1e-8:
        raise AssertionError(
            f"FM reference drift: {fm_valid_metrics['primary']} vs {FM_REFERENCE_VALID}"
        )
    print(f"  FM reference valid={fm_valid_metrics['primary']:.5f}", flush=True)

    current_valid_scores = 0.1 * fm_valid_scores + 0.9 * minmax(lgb_valid_scores)
    current_valid_metrics = evaluate(users["valid"], y["valid"], current_valid_scores)
    if abs(current_valid_metrics["primary"] - CURRENT_BLEND_REFERENCE_VALID) > 1e-8:
        raise AssertionError(
            "current blend reference drift: "
            f"{current_valid_metrics['primary']} vs {CURRENT_BLEND_REFERENCE_VALID}"
        )
    print(
        f"  current 10% FM / 90% LightGBM valid={current_valid_metrics['primary']:.5f}",
        flush=True,
    )
    blend_best, blend_sweep, blend_components = score_weight_grid(
        fm_valid_scores,
        lgb_valid_scores,
        xgb_valid_scores,
        users["valid"],
        y["valid"],
    )
    blend_delta = float(blend_best["valid"]["primary"] - current_valid_metrics["primary"])
    print(
        f"  best weights={blend_best['weights']} "
        f"valid={blend_best['valid']['primary']:.5f} delta={blend_delta:+.5f}",
        flush=True,
    )
    results["components"] = {
        "xgb_valid": metric_dict(xgb_valid_metrics),
        "xgb_record": xgb_record,
        "lgb_valid": metric_dict(lgb_valid_metrics),
        "fm_valid": metric_dict(fm_valid_metrics),
        "current_10pct_fm_90pct_lgb_valid": metric_dict(current_valid_metrics),
    }
    results["weight_sweep"] = blend_sweep
    results["winner"] = blend_best
    results["delta_vs_current_blend_valid"] = blend_delta
    write_results(results)

    weights = blend_best["weights"]
    confirmation = []
    if blend_delta >= PROMOTION_DELTA:
        print("\n=== five-seed fixed-weight blend validation confirmation ===", flush=True)
        confirmation.append(
            {
                "seed": 0,
                "valid": blend_best["valid"],
                "delta_vs_current_blend": blend_delta,
            }
        )
        for seed in SEEDS[1:]:
            model, seed_scores, _, _ = fit_fixed_xgb(dfs, y, users, seed)
            seed_blend_scores = (
                weights["fm"] * blend_components["fm"]
                + weights["lgb"] * blend_components["lgb"]
                + weights["xgb"] * minmax(seed_scores)
            )
            seed_metrics = evaluate(users["valid"], y["valid"], seed_blend_scores)
            seed_delta = float(seed_metrics["primary"] - current_valid_metrics["primary"])
            confirmation.append(
                {
                    "seed": seed,
                    "valid": metric_dict(seed_metrics),
                    "delta_vs_current_blend": seed_delta,
                }
            )
            print(
                f"  seed={seed} blend_valid={seed_metrics['primary']:.5f} "
                f"delta={seed_delta:+.5f}",
                flush=True,
            )
            del model, seed_scores, seed_blend_scores
            gc.collect()

    deltas = np.asarray(
        [row["delta_vs_current_blend"] for row in confirmation], dtype=np.float64
    )
    confirmed = bool(
        len(deltas) == len(SEEDS)
        and np.mean(deltas) >= PROMOTION_DELTA
        and np.all(deltas >= PROMOTION_DELTA)
    )
    results["five_seed_confirmation"] = {
        "rows": confirmation,
        "mean_delta": float(np.mean(deltas)) if len(deltas) else None,
        "std_delta": float(np.std(deltas)) if len(deltas) else None,
        "min_delta": float(np.min(deltas)) if len(deltas) else None,
        "confirmed": confirmed,
    }
    selected_valid_scores = (
        weights["fm"] * blend_components["fm"]
        + weights["lgb"] * blend_components["lgb"]
        + weights["xgb"] * blend_components["xgb"]
    )
    results["tie_diagnostic"] = {
        "xgb": standalone.tie_stats(xgb_valid_scores, users["valid"]),
        "blend": standalone.tie_stats(selected_valid_scores, users["valid"]),
    }
    results["verdict"] = "PROMOTE_THREE_MODEL_BLEND" if confirmed else "REJECT"
    write_results(results)

    print("\n=== FINAL BLEND TEST EVALUATION (selection frozen) ===", flush=True)
    xgb_test_scores = xgb_model.predict(dfs["test"])
    lgb_test_scores = lgb_model.predict(dfs["test"])
    Xte_fm, yte_fm, ute_fm = fm_context["encoded"]["test"]
    fm_test_seed_scores = [
        fm_context["submission_ref"].sigmoid(model.predict(Xte_fm))
        for model in fm_context["models"]
    ]
    fm_test_scores = np.mean(np.stack(fm_test_seed_scores), axis=0)
    assert np.array_equal(np.asarray(yte_fm), y["test"]), "FM/native test labels differ"
    assert np.array_equal(
        np.asarray(ute_fm), np.asarray(users["test"])
    ), "FM/native test users differ"

    current_test_scores = 0.1 * fm_test_scores + 0.9 * minmax(lgb_test_scores)
    current_test_metrics = evaluate(users["test"], y["test"], current_test_scores)
    blend_test_scores = (
        weights["fm"] * fm_test_scores
        + weights["lgb"] * minmax(lgb_test_scores)
        + weights["xgb"] * minmax(xgb_test_scores)
    )
    blend_test_metrics = evaluate(users["test"], y["test"], blend_test_scores)
    results["components"]["xgb_test"] = metric_dict(
        evaluate(users["test"], y["test"], xgb_test_scores)
    )
    results["components"]["lgb_test"] = metric_dict(
        evaluate(users["test"], y["test"], lgb_test_scores)
    )
    results["components"]["fm_test"] = metric_dict(
        evaluate(users["test"], y["test"], fm_test_scores)
    )
    results["components"]["current_10pct_fm_90pct_lgb_test"] = metric_dict(
        current_test_metrics
    )
    results["winner"]["test"] = metric_dict(blend_test_metrics)
    results["delta_vs_current_blend_test"] = float(
        blend_test_metrics["primary"] - current_test_metrics["primary"]
    )
    results["test_evaluation"] = {
        "performed": True,
        "timing": "after component configuration, validation weights, and confirmation were frozen",
    }
    write_results(results)
    print(
        f"three-model blend: valid={blend_best['valid']['primary']:.5f} "
        f"test={blend_test_metrics['primary']:.5f} verdict={results['verdict']}",
        flush=True,
    )
    print(f"wrote {BLEND_RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
