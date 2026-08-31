"""Shared fixed data, model architectures, and checks for YIXI Section 6g."""

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
YIXI6_DIR = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI6_cross_model_feature_transfer"
)
YIXI6_COMMON_PATH = os.path.join(YIXI6_DIR, "common.py")
YIXI5_BLEND_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "blend.py"
)

sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402


SEEDS = (0, 1, 2, 3, 4)
PRELIMINARY_DELTA = 0.0003
PROMOTION_DELTA = 0.001

LGB_POST6F_VALID = 0.6731172204017639
XGB_POST6F_VALID = 0.6697614192962646
CURRENT_LGB_VALID = 0.6716787219047546
CURRENT_XGB_VALID = 0.6675541996955872
FM_VALID = 0.6398779153823853
ENSEMBLE_VALID = 0.6823752522468567
ENSEMBLE_TEST = 0.6694862246513367
CURRENT_WEIGHTS = {"fm": 0.24, "lgb": 0.40, "xgb": 0.36}


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# YIXI6 owns the exact causal union of the post-6f feature representations.
# It imports its sibling ``features.py`` by name, so expose only that directory
# while loading it.  No shared source or feature cache is written here.
if YIXI6_DIR not in sys.path:
    sys.path.insert(0, YIXI6_DIR)
yixi6 = load_module(YIXI6_COMMON_PATH, "yixi6_common_for_yixi7")

LGB_TUNING_COLUMNS = list(yixi6.LGB_B1_COLUMNS)
XGB_TUNING_COLUMNS = list(yixi6.XGB_A1_COLUMNS)
CURRENT_LGB_COLUMNS = list(yixi6.LGB_B0_COLUMNS)
CURRENT_XGB_COLUMNS = list(yixi6.XGB_A0_COLUMNS)


LGB_TREE_CONFIG = {
    "metric": "ndcg",
    "eval_at": [5],
    "num_leaves": 2,
    "learning_rate": 0.10,
    "n_estimators": 500,
    "min_child_samples": 200,
    "reg_lambda": 1.0,
    "verbosity": -1,
    "n_jobs": -1,
    "linear_tree": True,
}


def xgb_tree_config() -> dict[str, Any]:
    return dict(yixi6.selected_xgb_config())


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


def read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "lightgbm": lgb.__version__,
        "xgboost": xgb.__version__,
    }


def load_frames():
    return yixi6.load_frames(DATA_DIR)


def stable_user_order(user_ids):
    values = np.asarray(user_ids)
    order = np.argsort(values, kind="stable")
    groups = np.unique(values[order], return_counts=True)[1]
    return order, groups


def fit_lgb(frames, y, users, rank_config: dict[str, Any], seed: int = 0):
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])
    columns = LGB_TUNING_COLUMNS
    Xtr = frames["train"][columns].iloc[train_order].reset_index(drop=True)
    Xva = frames["valid"][columns].iloc[valid_order].reset_index(drop=True)
    config = dict(LGB_TREE_CONFIG)
    config.update(rank_config)
    model = lgb.LGBMRanker(**config, random_state=seed)
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


def fit_lgb_current(frames, y, users, seed: int = 0):
    """Unchanged YIXI5 ensemble LightGBM, not the post-6f tuning reference."""
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])
    columns = CURRENT_LGB_COLUMNS
    Xtr = frames["train"][columns].iloc[train_order].reset_index(drop=True)
    Xva = frames["valid"][columns].iloc[valid_order].reset_index(drop=True)
    model = lgb.LGBMRanker(
        objective="lambdarank", **LGB_TREE_CONFIG, random_state=seed
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
    del Xtr, Xva
    gc.collect()
    return model, scores, metrics


def fit_xgb(frames, y, users, rank_config: dict[str, Any], seed: int = 0):
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])
    columns = XGB_TUNING_COLUMNS
    Xtr = frames["train"][columns].iloc[train_order].reset_index(drop=True)
    Xva = frames["valid"][columns].iloc[valid_order].reset_index(drop=True)
    config = xgb_tree_config()
    config.update(rank_config)
    model = xgb.XGBRanker(**config, random_state=seed, n_jobs=-1, verbosity=0)
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


def fit_xgb_current(frames, y, users, seed: int = 0):
    """Unchanged YIXI5 ensemble XGBoost, not the post-6f tuning reference."""
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])
    columns = CURRENT_XGB_COLUMNS
    Xtr = frames["train"][columns].iloc[train_order].reset_index(drop=True)
    Xva = frames["valid"][columns].iloc[valid_order].reset_index(drop=True)
    model = xgb.XGBRanker(
        **xgb_tree_config(), random_state=seed, n_jobs=-1, verbosity=0
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
    del Xtr, Xva
    gc.collect()
    return model, scores, metrics


def xgb_effective_objective(model) -> dict[str, Any]:
    config = json.loads(model.get_booster().save_config())
    return config["learner"]["objective"]


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


def record(name, rank_config, model, metrics, gain_fraction, family):
    best_iteration = (
        int(model.best_iteration_) if family == "lgb" else int(model.best_iteration)
    )
    return {
        "name": name,
        "rank_config": rank_config,
        "best_iteration": best_iteration,
        "valid": metric_dict(metrics),
        "feature_gain_fraction": gain_fraction,
    }


def confirmation_summary(rows):
    deltas = np.asarray([row["delta"] for row in rows], dtype=np.float64)
    return {
        "performed": bool(rows),
        "rows": rows,
        "mean_delta": float(np.mean(deltas)) if len(deltas) else None,
        "std_delta": float(np.std(deltas)) if len(deltas) else None,
        "min_delta": float(np.min(deltas)) if len(deltas) else None,
        "confirmed": bool(
            len(deltas) == len(SEEDS)
            and np.mean(deltas) >= PROMOTION_DELTA
            and np.all(deltas >= PROMOTION_DELTA)
        ),
    }
