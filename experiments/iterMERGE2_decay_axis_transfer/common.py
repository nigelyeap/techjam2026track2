"""Shared references and fitting helpers for iterMERGE2.

Tests three causal decay-rate feature axes discovered by this project's own
research track (iter71 author, iter72 duration-bucket, iter73 hour-of-day --
all previously REJECTed as exact-zero on the old iter63 rate_only harness)
against teammate yixi's current LightGBM/XGBoost reference representation and
her current-best 3-model blend. Nothing outside this directory is modified;
every fixed reference value and column list below is copied (not imported by
mutation) from experiments/iterYIXI9_watch_depth_history/common.py and
experiments/iterYIXI10_video_metadata/{common.py,RESULT.md}.
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

YIXI9_FEATURES_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI9_watch_depth_history", "features.py"
)
YIXI10_FEATURES_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI10_video_metadata", "features.py"
)
YIXI5_RESULTS_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "results.json"
)
YIXI5_BLEND_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "blend.py"
)
YIXI8_COMMON_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI8_rank_space_calibration", "common.py"
)
AXIS_DATA_EXT_PATHS = {
    "author": os.path.join(
        REPO_ROOT, "experiments", "iter71_decay_author_rate", "data_ext.py"
    ),
    "durbucket": os.path.join(
        REPO_ROOT, "experiments", "iter72_decay_durbucket_rate", "data_ext.py"
    ),
    "hour": os.path.join(
        REPO_ROOT, "experiments", "iter73_decay_hour_rate", "data_ext.py"
    ),
}

sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402


SEEDS = (0, 1, 2, 3, 4)
PRELIMINARY_DELTA = 0.0003
PROMOTION_DELTA = 0.001
ALPHA = 0.5  # matches iter63/iter71/iter72/iter73's Laplace-smoothing constant
HALFLIFE = 3  # matches decay_tab_rate_3's own halflife; also present for all 3 axes

# --- Exact fidelity targets, copied from iterYIXI9_watch_depth_history/common.py ---
LGB_REFERENCE_VALID = 0.6768913269042969
XGB_REFERENCE_VALID = 0.6697614192962646
CURRENT_XGB_VALID = 0.6675541996955872

CAT_COLS = ["user_id", "video_id", "author_id", "tab", "last1"]
LGB_REFERENCE_COLUMNS = CAT_COLS + [
    "duration_ms",
    "decay_rate_5",
    "decay_act_5",
    "lastk_rate",
    "gap",
    "decay_tab_rate_3",
]
XGB_REFERENCE_COLUMNS = CAT_COLS + [
    "duration_ms",
    "decay_tab_rate_3",
    "lastk_rate",
    "gap",
    "decay_rate_5",
    "decay_act_5",
]
CURRENT_XGB_COLUMNS = CAT_COLS + [
    "duration_ms",
    "decay_tab_3",
    "lastk_rate",
    "gap",
    "decay_rate_5",
    "decay_act_5",
]

# --- Current best 3-model blend, copied from iterYIXI10_video_metadata/RESULT.md ---
BEST_BLEND_VALID = 0.69943440
BEST_BLEND_TEST = 0.68432260
BEST_WEIGHTS = {"fm": 0.10, "lgb": 0.52, "xgb": 0.38}
LGB_BEST_COLUMNS = CAT_COLS + [
    "duration_ms",
    "decay_rate_5",
    "decay_act_5",
    "lastk_rate",
    "gap",
    "decay_tab_rate_3",
    "hist_watch_decay_mean_5",
    "meta_upload_type",
]
XGB_BEST_COLUMNS = CAT_COLS + [
    "duration_ms",
    "decay_tab_3",
    "lastk_rate",
    "gap",
    "decay_rate_5",
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


def metric_dict(metrics):
    return {key: jsonable(value) for key, value in metrics.items()}


def write_json(path: str, payload: dict[str, Any]) -> None:
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


def selected_xgb_config():
    results = read_json(YIXI5_RESULTS_PATH)
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
        raise AssertionError(f"YIXI5 XGBoost config drift: {config}")
    return config


def stable_user_order(user_ids):
    values = np.asarray(user_ids)
    order = np.argsort(values, kind="stable")
    groups = np.unique(values[order], return_counts=True)[1]
    return order, groups


def fit_lgb(frames, y, users, columns, seed=0):
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])
    Xtr = frames["train"][columns].iloc[train_order].reset_index(drop=True)
    Xva = frames["valid"][columns].iloc[valid_order].reset_index(drop=True)
    model = lgb.LGBMRanker(**LGB_CONFIG, random_state=seed)
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
    fractions = {
        column: (float(value) / total_gain if total_gain else 0.0)
        for column, value in zip(columns, gain)
    }
    del Xtr, Xva
    gc.collect()
    return model, scores, metrics, fractions


def fit_xgb(frames, y, users, columns, seed=0):
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
    fractions = {
        column: (float(raw_gain.get(column, 0.0)) / total_gain if total_gain else 0.0)
        for column in columns
    }
    del Xtr, Xva
    gc.collect()
    return model, scores, metrics, fractions


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


def unique_stats(scores, users):
    frame = pd.DataFrame({"user": np.asarray(users), "score": np.asarray(scores)})
    grouped = frame.groupby("user", sort=False)["score"]
    fractions = grouped.nunique(dropna=False) / grouped.size()
    return {
        "rows": int(len(frame)),
        "unique_scores_overall": int(frame["score"].nunique(dropna=False)),
        "mean_per_user_unique_fraction": float(fractions.mean()),
    }


def load_reference_frames(source: str):
    """source in {'yixi9', 'yixi10'}. Returns frames, y, users, metadata."""
    path = YIXI9_FEATURES_PATH if source == "yixi9" else YIXI10_FEATURES_PATH
    mod = load_module(path, f"features_for_merge2_{source}")
    return mod.load_frames(DATA_DIR)


_AXIS_COL_FN_NAMES = {
    "author": ("_author_halflife_col", "_author_halflife_total_col"),
    "durbucket": ("_dur_halflife_col", "_dur_halflife_total_col"),
    "hour": ("_hour_halflife_col", "_hour_halflife_total_col"),
}


def axis_column_name(axis: str, halflife: int = HALFLIFE) -> str:
    return f"decay_{axis}_rate_{halflife}"


def attach_axis_rate(frames, y, users, axis: str, halflife: int = HALFLIFE) -> str:
    """Attach the iter71/72/73 causal decayed-rate feature for one axis onto
    an already-loaded reference frames dict, verifying row-for-row alignment
    (user_id, label) against the reference split before trusting anything."""
    mod = load_module(AXIS_DATA_EXT_PATHS[axis], f"data_ext_for_merge2_{axis}")
    ext = mod.load_ext(DATA_DIR, use_cache=True)
    pos_fn_name, tot_fn_name = _AXIS_COL_FN_NAMES[axis]
    pos_col = getattr(mod, pos_fn_name)(halflife)
    tot_col = getattr(mod, tot_fn_name)(halflife)
    col_name = axis_column_name(axis, halflife)
    for split in ("train", "valid", "test"):
        rows = ext[split]
        if len(rows) != len(frames[split]):
            raise AssertionError(
                f"row count mismatch axis={axis} split={split}: "
                f"{len(rows)} vs {len(frames[split])}"
            )
        row_users = np.asarray([r[mod.IDX["user_id"]] for r in rows])
        row_labels = np.asarray([r[mod.IDX["label"]] for r in rows], dtype=np.float32)
        if not np.array_equal(row_users, np.asarray(users[split])):
            raise AssertionError(f"user alignment failed axis={axis} split={split}")
        if not np.array_equal(row_labels, y[split]):
            raise AssertionError(f"label alignment failed axis={axis} split={split}")
        pos = np.asarray([r[pos_col] for r in rows], dtype=np.float64)
        tot = np.asarray([r[tot_col] for r in rows], dtype=np.float64)
        rate = (pos + ALPHA) / (tot + 2 * ALPHA)
        frames[split] = frames[split].copy()
        frames[split][col_name] = rate
    return col_name
