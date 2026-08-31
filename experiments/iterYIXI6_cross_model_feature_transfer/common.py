"""Shared fixed configurations and fitting helpers for YIXI Section 6f."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import platform
from typing import Any

import lightgbm as lgb
import numpy as np
import xgboost as xgb


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
YIXI5_RESULTS_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "results.json"
)

import sys

sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402

from features import CAT_COLS, load_frames  # noqa: E402


SEEDS = (0, 1, 2, 3, 4)
PRELIMINARY_DELTA = 0.0003
PROMOTION_DELTA = 0.001

XGB_REFERENCE_VALID = 0.6675541996955872
LGB_REFERENCE_VALID = 0.6716787219047546
FM_REFERENCE_VALID = 0.6398779153823853
BLEND_REFERENCE_VALID = 0.6823752522468567
BLEND_REFERENCE_TEST = 0.6694862246513367

XGB_A0_COLUMNS = CAT_COLS + [
    "duration_ms",
    "decay_tab_3",
    "lastk_rate",
    "gap",
    "decay_rate_5",
    "decay_act_5",
]
XGB_A1_COLUMNS = CAT_COLS + [
    "duration_ms",
    "decay_tab_rate_3",
    "lastk_rate",
    "gap",
    "decay_rate_5",
    "decay_act_5",
]
XGB_A2_COLUMNS = CAT_COLS + [
    "duration_ms",
    "decay_tab_3",
    "decay_tab_rate_3",
    "lastk_rate",
    "gap",
    "decay_rate_5",
    "decay_act_5",
]

LGB_B0_COLUMNS = CAT_COLS + [
    "duration_ms",
    "decay_rate_2.5",
    "decay_act_2.5",
    "lastk_rate",
    "gap",
    "decay_tab_rate_3",
]
LGB_B1_COLUMNS = CAT_COLS + [
    "duration_ms",
    "decay_rate_5",
    "decay_act_5",
    "lastk_rate",
    "gap",
    "decay_tab_rate_3",
]
LGB_B2_COLUMNS = LGB_B0_COLUMNS + ["decay_rate_5", "decay_act_5"]
LGB_C1_COLUMNS = LGB_B0_COLUMNS + ["decay_rate_x_log_activity"]


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "lightgbm": lgb.__version__,
        "xgboost": xgb.__version__,
    }


def selected_xgb_config() -> dict[str, Any]:
    with open(YIXI5_RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    config = results["selected_on_validation"]["config"]
    expected = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@5-",
        "max_depth": 1,
        "learning_rate": 0.025,
        "n_estimators": 1000,
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
        "early_stopping_rounds": 120,
    }
    if config != expected:
        raise AssertionError(f"YIXI5 selected XGBoost config drifted: {config}")
    return config


def stable_user_order(user_ids):
    values = np.asarray(user_ids)
    order = np.argsort(values, kind="stable")
    groups = np.unique(values[order], return_counts=True)[1]
    return order, groups


def fit_xgb(frames, y, users, columns: list[str], seed: int):
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])
    Xtr = frames["train"][columns].iloc[train_order].reset_index(drop=True)
    Xva = frames["valid"][columns].iloc[valid_order].reset_index(drop=True)
    model = xgb.XGBRanker(
        **selected_xgb_config(), random_state=seed, n_jobs=-1, verbosity=0
    )
    model.fit(
        Xtr,
        y["train"][train_order],
        group=train_groups,
        eval_set=[(Xva, y["valid"][valid_order])],
        eval_group=[valid_groups],
        verbose=False,
    )
    scores = model.predict(frames["valid"][columns])
    metrics = evaluate(users["valid"], y["valid"], scores)
    raw_gain = model.get_booster().get_score(importance_type="gain")
    total_gain = float(sum(raw_gain.values()))
    gain_fraction = {
        column: (float(raw_gain.get(column, 0.0)) / total_gain if total_gain else 0.0)
        for column in columns
    }
    del Xtr, Xva
    gc.collect()
    return model, scores, metrics, gain_fraction


def fit_lgb(frames, y, users, columns: list[str], seed: int):
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])
    Xtr = frames["train"][columns].iloc[train_order].reset_index(drop=True)
    Xva = frames["valid"][columns].iloc[valid_order].reset_index(drop=True)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[5],
        num_leaves=2,
        learning_rate=0.10,
        n_estimators=500,
        min_child_samples=200,
        reg_lambda=1.0,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1,
        linear_tree=True,
    )
    model.fit(
        Xtr,
        y["train"][train_order],
        group=train_groups,
        eval_set=[(Xva, y["valid"][valid_order])],
        eval_group=[valid_groups],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    scores = model.predict(frames["valid"][columns])
    metrics = evaluate(users["valid"], y["valid"], scores)
    gain = model.booster_.feature_importance(importance_type="gain")
    total_gain = float(np.sum(gain))
    gain_fraction = {
        column: (float(value) / total_gain if total_gain else 0.0)
        for column, value in zip(columns, gain)
    }
    del Xtr, Xva
    gc.collect()
    return model, scores, metrics, gain_fraction


def fit_record(name, model_family, columns, seed, model, metrics, gain_fraction):
    best_iteration = (
        int(model.best_iteration)
        if model_family == "xgb"
        else int(model.best_iteration_)
    )
    return {
        "name": name,
        "model_family": model_family,
        "columns": list(columns),
        "seed": seed,
        "best_iteration": best_iteration,
        "valid": metric_dict(metrics),
        "feature_gain_fraction": gain_fraction,
    }


def tie_stats(scores, user_ids) -> dict[str, Any]:
    scores = np.asarray(scores)
    user_ids = np.asarray(user_ids)
    fractions = []
    for user_id in np.unique(user_ids):
        values = scores[user_ids == user_id]
        fractions.append(len(np.unique(values)) / len(values))
    return {
        "unique_scores_overall": int(len(np.unique(scores))),
        "rows": int(len(scores)),
        "mean_per_user_unique_fraction": float(np.mean(fractions)),
    }


def confirm_candidate(
    model_family: str,
    frames,
    y,
    users,
    reference_columns,
    candidate_columns,
    seed0_reference,
    seed0_candidate,
):
    fit = fit_xgb if model_family == "xgb" else fit_lgb
    rows = [
        {
            "seed": 0,
            "reference_valid": seed0_reference,
            "candidate_valid": seed0_candidate,
            "delta": float(
                seed0_candidate["primary"] - seed0_reference["primary"]
            ),
        }
    ]
    for seed in SEEDS[1:]:
        ref_model, ref_scores, ref_metrics, _ = fit(
            frames, y, users, reference_columns, seed
        )
        cand_model, cand_scores, cand_metrics, _ = fit(
            frames, y, users, candidate_columns, seed
        )
        rows.append(
            {
                "seed": seed,
                "reference_valid": metric_dict(ref_metrics),
                "candidate_valid": metric_dict(cand_metrics),
                "delta": float(cand_metrics["primary"] - ref_metrics["primary"]),
            }
        )
        print(
            f"  seed={seed} reference={ref_metrics['primary']:.8f} "
            f"candidate={cand_metrics['primary']:.8f} "
            f"delta={rows[-1]['delta']:+.8f}",
            flush=True,
        )
        del ref_model, ref_scores, cand_model, cand_scores
        gc.collect()
    deltas = np.asarray([row["delta"] for row in rows], dtype=np.float64)
    return {
        "performed": True,
        "rows": rows,
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas)),
        "min_delta": float(np.min(deltas)),
        "confirmed": bool(
            np.mean(deltas) >= PROMOTION_DELTA
            and np.all(deltas >= PROMOTION_DELTA)
        ),
    }
