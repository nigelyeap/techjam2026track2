"""Section 6a, part 1: standalone XGBoost-native ranker experiment.

This file intentionally tests only XGBoost models. It selects the objective
and max depth using official validation primary, checks the selected model for
tie artifacts, and evaluates that frozen standalone winner on test once.

The separate ``blend.py`` imports the fixed winner from this module and tests
the FM + LightGBM + XGBoost combination. The blend, not this standalone model,
is the promoted 6a result.

Run from the repository root:

    python3 experiments/iterYIXI1_xgboost_native/run_experiment.py
"""

from __future__ import annotations

import gc
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
ITER44_DIR = os.path.join(REPO_ROOT, "experiments", "iter44_gbm_native_features")
RESULTS_PATH = os.path.join(THIS_DIR, "results.json")

sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402


OBJECTIVES = ("rank:pairwise", "rank:ndcg")
MAX_DEPTHS = (1, 2, 3, 4, 5, 7)
SEEDS = (0, 1, 2, 3, 4)

LEARNING_RATE = 0.05
N_ESTIMATORS = 500
EARLY_STOPPING_ROUNDS = 30
REG_LAMBDA = 1.0

LGB_REFERENCE_VALID = 0.6613549
PROMOTION_DELTA = 0.001


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


iter44 = _load_module(os.path.join(ITER44_DIR, "train.py"), "iter44_train_for_yixi1")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_results(payload: dict[str, Any]) -> None:
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def _metric_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in metrics.items()}


def grouped_view(df, labels, user_ids):
    """Stable user sort and group sizes required by XGBRanker."""
    user_ids = np.asarray(user_ids)
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    groups = np.unique(sorted_users, return_counts=True)[1]
    return df.iloc[order].reset_index(drop=True), labels[order], groups


def make_xgb_ranker(objective: str, max_depth: int, seed: int) -> xgb.XGBRanker:
    """Construct the exact 6a XGBoost model; also reused by later experiments."""
    return xgb.XGBRanker(
        objective=objective,
        eval_metric="ndcg@5-",
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=max_depth,
        min_child_weight=1.0,
        reg_lambda=REG_LAMBDA,
        reg_alpha=0.0,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        max_bin=256,
        enable_categorical=True,
        max_cat_to_onehot=4,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )


def fit_xgb_candidate(
    objective: str,
    max_depth: int,
    seed: int,
    Xtr,
    ytr,
    gtr,
    Xva,
    yva,
    gva,
    valid_df,
    valid_users,
    valid_labels,
):
    model = make_xgb_ranker(objective, max_depth, seed)
    model.fit(
        Xtr,
        ytr,
        group=gtr,
        eval_set=[(Xva, yva)],
        eval_group=[gva],
        verbose=False,
    )
    valid_scores = model.predict(valid_df)
    valid_metrics = evaluate(valid_users, valid_labels, valid_scores)
    record = {
        "objective": objective,
        "max_depth": max_depth,
        "seed": seed,
        "best_iteration": int(model.best_iteration),
        "best_internal_ndcg_at_5_minus": float(model.best_score),
        "valid": _metric_dict(valid_metrics),
    }
    return model, valid_scores, valid_metrics, record


def tie_stats(scores, user_ids) -> dict[str, Any]:
    scores = np.asarray(scores)
    user_ids = np.asarray(user_ids)
    per_user_unique_fraction = []
    for user_id in np.unique(user_ids):
        group_scores = scores[user_ids == user_id]
        per_user_unique_fraction.append(len(np.unique(group_scores)) / len(group_scores))
    return {
        "unique_scores_overall": int(len(np.unique(scores))),
        "rows": int(len(scores)),
        "mean_per_user_unique_fraction": float(np.mean(per_user_unique_fraction)),
    }


def main() -> None:
    print("=== loading iter44 native features (train-fitted categoricals) ===", flush=True)
    dfs, y, users = iter44.prepare(DATA_DIR)
    print(
        f"rows: train={len(dfs['train'])} valid={len(dfs['valid'])} "
        f"test={len(dfs['test'])}; test is not scored during selection",
        flush=True,
    )
    assert list(dfs["train"].columns) == iter44.CAT_COLS + iter44.NUM_COLS

    Xtr, ytr, gtr = grouped_view(dfs["train"], y["train"], users["train"])
    Xva, yva_grouped, gva = grouped_view(dfs["valid"], y["valid"], users["valid"])
    results: dict[str, Any] = {
        "experiment": "iterYIXI1_xgboost_native_standalone",
        "scope": "standalone XGBoost only; blend.py owns the promoted three-model blend",
        "selection_policy": {
            "selector": "official validation primary only",
            "test_access": "after the standalone objective/depth configuration is frozen",
            "promotion_reference": "iter44 standalone native LightGBM",
            "promotion_delta": PROMOTION_DELTA,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
        },
        "features": {
            "categorical": list(iter44.CAT_COLS),
            "numeric": list(iter44.NUM_COLS),
        },
        "fixed_xgb_parameters": {
            "learning_rate": LEARNING_RATE,
            "n_estimators": N_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "eval_metric": "ndcg@5-",
            "min_child_weight": 1.0,
            "reg_lambda": REG_LAMBDA,
            "reg_alpha": 0.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "max_bin": 256,
            "enable_categorical": True,
            "max_cat_to_onehot": 4,
        },
        "sweep": [],
    }
    _write_results(results)

    print("\n=== validation-only XGBoost objective/depth sweep ===", flush=True)
    best_model = None
    best_scores = None
    best_metrics = None
    best_record = None
    for objective in OBJECTIVES:
        for max_depth in MAX_DEPTHS:
            model, scores, metrics, record = fit_xgb_candidate(
                objective,
                max_depth,
                0,
                Xtr,
                ytr,
                gtr,
                Xva,
                yva_grouped,
                gva,
                dfs["valid"],
                users["valid"],
                y["valid"],
            )
            results["sweep"].append(record)
            print(
                f"  objective={objective:13s} depth={max_depth} "
                f"best_iter={record['best_iteration']:3d} "
                f"valid={metrics['primary']:.5f} "
                f"(GAUC={metrics['GAUC']:.5f}, nDCG@5={metrics['nDCG@5']:.5f})",
                flush=True,
            )
            if best_metrics is None or metrics["primary"] > best_metrics["primary"] + 1e-12:
                if best_model is not None:
                    del best_model
                best_model = model
                best_scores = scores
                best_metrics = metrics
                best_record = record
            else:
                del model, scores
            gc.collect()
            results["sweep_winner_so_far"] = best_record
            _write_results(results)

    assert best_model is not None and best_scores is not None
    assert best_metrics is not None and best_record is not None
    results["standalone_winner"] = best_record
    print(
        f"\nstandalone winner: objective={best_record['objective']} "
        f"depth={best_record['max_depth']} valid={best_metrics['primary']:.5f}",
        flush=True,
    )

    rng = np.random.default_rng(0)
    constant_metrics = evaluate(users["valid"], y["valid"], np.zeros(len(y["valid"])))
    random_metrics = evaluate(
        users["valid"], y["valid"], rng.uniform(size=len(y["valid"]))
    )
    results["tie_diagnostic"] = {
        "constant_valid": _metric_dict(constant_metrics),
        "random_valid": _metric_dict(random_metrics),
        "winner": tie_stats(best_scores, users["valid"]),
    }
    print(
        "tie diagnostic: "
        f"constant={constant_metrics['primary']:.5f} "
        f"random={random_metrics['primary']:.5f} "
        f"winner_mean_user_unique="
        f"{results['tie_diagnostic']['winner']['mean_per_user_unique_fraction']:.4f}",
        flush=True,
    )

    standalone_delta = float(best_metrics["primary"] - LGB_REFERENCE_VALID)
    seed_confirmation = []
    if standalone_delta >= PROMOTION_DELTA:
        print("\n=== five-seed standalone validation confirmation ===", flush=True)
        seed_confirmation.append({"seed": 0, "valid": _metric_dict(best_metrics)})
        for seed in SEEDS[1:]:
            model, scores, metrics, _ = fit_xgb_candidate(
                best_record["objective"],
                best_record["max_depth"],
                seed,
                Xtr,
                ytr,
                gtr,
                Xva,
                yva_grouped,
                gva,
                dfs["valid"],
                users["valid"],
                y["valid"],
            )
            seed_confirmation.append({"seed": seed, "valid": _metric_dict(metrics)})
            print(f"  seed={seed} valid={metrics['primary']:.5f}", flush=True)
            del model, scores
            gc.collect()

    confirmed = bool(
        standalone_delta >= PROMOTION_DELTA
        and len(seed_confirmation) == len(SEEDS)
        and all(
            row["valid"]["primary"] - LGB_REFERENCE_VALID >= PROMOTION_DELTA
            for row in seed_confirmation
        )
    )
    results["standalone_comparison"] = {
        "lgb_reference_valid": LGB_REFERENCE_VALID,
        "delta_vs_lgb_valid": standalone_delta,
        "seed_confirmation": seed_confirmation,
        "confirmed": confirmed,
        "verdict": "PROMOTE" if confirmed else "REJECT",
        "note": "This verdict applies only to standalone XGBoost.",
    }
    _write_results(results)

    print("\n=== FINAL STANDALONE TEST EVALUATION (selection frozen) ===", flush=True)
    test_scores = best_model.predict(dfs["test"])
    test_metrics = evaluate(users["test"], y["test"], test_scores)
    results["standalone_winner"]["test"] = _metric_dict(test_metrics)
    results["test_evaluation"] = {
        "performed": True,
        "timing": "after standalone objective/depth selection was frozen",
    }
    _write_results(results)
    print(
        f"XGBoost standalone: valid={best_metrics['primary']:.5f} "
        f"test={test_metrics['primary']:.5f} verdict={results['standalone_comparison']['verdict']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
