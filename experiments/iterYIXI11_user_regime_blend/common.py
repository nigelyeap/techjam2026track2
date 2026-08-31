"""Fixed YIXI10 system and rank-space helpers for adaptive user blending."""

from __future__ import annotations

import gc
import hashlib
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

import features
import regimes


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
PREDICTIONS_PATH = os.path.join(THIS_DIR, ".official_valid_predictions.npz")
PREDICTIONS_METADATA_PATH = os.path.join(THIS_DIR, "harness_results.json")
YIXI5_RESULTS_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "results.json"
)
YIXI5_BLEND_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "blend.py"
)
YIXI8_COMMON_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI8_rank_space_calibration", "common.py"
)

sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402


SEEDS = (0, 1, 2, 3, 4)
MODELS = ("fm", "lgb", "xgb")
PRELIMINARY_DELTA = 0.0003
PROMOTION_DELTA = 0.001
MEANINGFUL_ORDER_MARGIN = 0.001
MIN_REGIME_USERS = 200
LGB_REFERENCE_VALID = 0.6883414387702942
XGB_REFERENCE_VALID = 0.6675541996955872
FM_REFERENCE_VALID = 0.6398779153823853
ENSEMBLE_REFERENCE_VALID = 0.6994343996047974
ENSEMBLE_REFERENCE_TEST = 0.6843225955963135
GLOBAL_WEIGHTS = {"fm": 0.10, "lgb": 0.52, "xgb": 0.38}
LOCAL_WEIGHT_RADIUS = 0.10
LOCAL_WEIGHT_STEP = 0.02
MIN_MODEL_WEIGHT = 0.02

CAT_COLS = ["user_id", "video_id", "author_id", "tab", "last1"]
LGB_COLUMNS = CAT_COLS + [
    "duration_ms", "decay_rate_5", "decay_act_5", "lastk_rate", "gap",
    "decay_tab_rate_3", "hist_watch_decay_mean_5", "meta_upload_type",
]
XGB_COLUMNS = CAT_COLS + [
    "duration_ms", "decay_tab_3", "lastk_rate", "gap", "decay_rate_5",
    "decay_act_5",
]
LGB_CONFIG = {
    "objective": "lambdarank",
    "lambdarank_truncation_level": 50,
    "sigmoid": 2.0,
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


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def metric_dict(metrics):
    return {key: jsonable(value) for key, value in metrics.items()}


def write_json(path: str, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def environment():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": lgb.__version__,
        "xgboost": xgb.__version__,
    }


def array_sha256(values):
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def selected_xgb_config():
    config = read_json(YIXI5_RESULTS_PATH)["selected_on_validation"]["config"]
    expected = {
        "objective": "rank:ndcg", "eval_metric": "ndcg@5-", "max_depth": 1,
        "learning_rate": 0.025, "n_estimators": 1000,
        "min_child_weight": 1.0, "gamma": 0.0, "reg_lambda": 1.0,
        "reg_alpha": 0.0, "subsample": 1.0, "colsample_bytree": 1.0,
        "tree_method": "hist", "max_bin": 256, "enable_categorical": True,
        "max_cat_to_onehot": 4, "early_stopping_rounds": 120,
    }
    if config != expected:
        raise AssertionError(f"XGBoost config drift: {config}")
    return config


def stable_user_order(user_ids):
    values = np.asarray(user_ids)
    order = np.argsort(values, kind="stable")
    groups = np.unique(values[order], return_counts=True)[1]
    return order, groups


def fit_lgb(frames, y, users, seed=0):
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])
    Xtr = frames["train"][LGB_COLUMNS].iloc[train_order].reset_index(drop=True)
    Xva = frames["valid"][LGB_COLUMNS].iloc[valid_order].reset_index(drop=True)
    model = lgb.LGBMRanker(**LGB_CONFIG, random_state=seed)
    model.fit(
        Xtr, y["train"][train_order], group=train_groups,
        eval_set=[(Xva, y["valid"][valid_order])], eval_group=[valid_groups],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    scores = model.predict(frames["valid"][LGB_COLUMNS])
    metrics = evaluate(users["valid"], y["valid"], scores)
    del Xtr, Xva
    gc.collect()
    return model, scores, metrics


def fit_xgb(frames, y, users, seed=0):
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])
    Xtr = frames["train"][XGB_COLUMNS].iloc[train_order].reset_index(drop=True)
    Xva = frames["valid"][XGB_COLUMNS].iloc[valid_order].reset_index(drop=True)
    model = xgb.XGBRanker(
        **selected_xgb_config(), random_state=seed, n_jobs=-1, verbosity=0
    )
    model.fit(
        Xtr, y["train"][train_order], group=train_groups,
        eval_set=[(Xva, y["valid"][valid_order])], eval_group=[valid_groups],
        verbose=False,
    )
    scores = model.predict(frames["valid"][XGB_COLUMNS])
    metrics = evaluate(users["valid"], y["valid"], scores)
    del Xtr, Xva
    gc.collect()
    return model, scores, metrics


def percentile(scores, users):
    score_series = pd.Series(np.asarray(scores, dtype=np.float64), copy=False)
    user_series = pd.Series(np.asarray(users), copy=False)
    values = score_series.groupby(user_series, sort=False).rank(
        method="average", pct=True
    ).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise AssertionError("non-finite percentile")
    return values


def percentile_components(raw, users):
    return {name: percentile(raw[name], users) for name in MODELS}


def combined(components, weights):
    return sum(float(weights[name]) * components[name] for name in MODELS)


def score_components(components, weights, users, labels):
    scores = combined(components, weights)
    return scores, evaluate(users, labels, scores)


def local_weight_grid():
    """Three-model weights within L-infinity 0.10 of the global reference."""
    records = []
    step_units = int(round(1.0 / LOCAL_WEIGHT_STEP))
    radius_units = int(round(LOCAL_WEIGHT_RADIUS / LOCAL_WEIGHT_STEP))
    centers = {
        name: int(round(GLOBAL_WEIGHTS[name] * step_units)) for name in MODELS
    }
    min_units = int(round(MIN_MODEL_WEIGHT * step_units))
    for fm_units in range(
        max(min_units, centers["fm"] - radius_units),
        centers["fm"] + radius_units + 1,
    ):
        for lgb_units in range(
            max(min_units, centers["lgb"] - radius_units),
            centers["lgb"] + radius_units + 1,
        ):
            xgb_units = step_units - fm_units - lgb_units
            if xgb_units < min_units:
                continue
            if abs(xgb_units - centers["xgb"]) > radius_units:
                continue
            weights = {
                "fm": fm_units / step_units,
                "lgb": lgb_units / step_units,
                "xgb": xgb_units / step_units,
            }
            if any(
                abs(weights[name] - GLOBAL_WEIGHTS[name])
                > LOCAL_WEIGHT_RADIUS + 1e-12
                for name in MODELS
            ):
                continue
            records.append(weights)
    if GLOBAL_WEIGHTS not in records:
        raise AssertionError("local grid omits global reference")
    return records


def adaptive_scores(components, row_regimes, weights_by_regime):
    output = np.empty(len(row_regimes), dtype=np.float64)
    assigned = np.zeros(len(row_regimes), dtype=bool)
    for regime in regimes.REGIME_NAMES:
        mask = row_regimes == regime
        output[mask] = combined(
            {name: values[mask] for name, values in components.items()},
            weights_by_regime[regime],
        )
        assigned[mask] = True
    if not np.all(assigned) or not np.all(np.isfinite(output)):
        raise AssertionError("invalid adaptive score assignment")
    return output


def load_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError("run harness.py first")
    payload = {key: value for key, value in np.load(PREDICTIONS_PATH).items()}
    metadata = read_json(PREDICTIONS_METADATA_PATH)
    for key, expected in metadata["array_sha256"].items():
        if array_sha256(payload[key]) != expected:
            raise AssertionError(f"prediction hash drift for {key}")
    return payload


def unique_stats(scores, users):
    frame = pd.DataFrame({"user": np.asarray(users), "score": np.asarray(scores)})
    grouped = frame.groupby("user", sort=False)["score"]
    fractions = grouped.nunique(dropna=False) / grouped.size()
    return {
        "rows": int(len(frame)),
        "unique_scores_overall": int(frame["score"].nunique(dropna=False)),
        "mean_per_user_unique_fraction": float(fractions.mean()),
    }
