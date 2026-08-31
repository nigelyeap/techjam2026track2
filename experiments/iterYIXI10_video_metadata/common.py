"""Fixed YIXI9 references and fitting helpers for Section 6j."""

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

import features


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
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
PRELIMINARY_DELTA = 0.0003
PROMOTION_DELTA = 0.001
LGB_REFERENCE_VALID = 0.6782224178314209
XGB_REFERENCE_VALID = 0.6675541996955872
FM_REFERENCE_VALID = 0.6398779153823853
ENSEMBLE_REFERENCE_VALID = 0.690884530544281
ENSEMBLE_REFERENCE_TEST = 0.6784203052520752
CURRENT_WEIGHTS = {"fm": 0.18, "lgb": 0.46, "xgb": 0.36}

CAT_COLS = ["user_id", "video_id", "author_id", "tab", "last1"]
LGB_REFERENCE_COLUMNS = CAT_COLS + [
    "duration_ms",
    "decay_rate_5",
    "decay_act_5",
    "lastk_rate",
    "gap",
    "decay_tab_rate_3",
    "hist_watch_decay_mean_5",
]
XGB_REFERENCE_COLUMNS = CAT_COLS + [
    "duration_ms",
    "decay_tab_3",
    "lastk_rate",
    "gap",
    "decay_rate_5",
    "decay_act_5",
]

# Predeclared independent tests.  music_id stays isolated as the final group.
FEATURE_GROUPS = {
    "video_age": ["video_age_days"],
    "video_type": ["meta_video_type"],
    "upload_type": ["meta_upload_type"],
    "music_type": ["meta_music_type"],
    "tag": ["meta_tag"],
    "aspect_ratio": ["aspect_ratio"],
    "music_id_high_cardinality": ["meta_music_id"],
}

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


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def environment():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": lgb.__version__,
        "xgboost": xgb.__version__,
    }


def selected_xgb_config():
    config = read_json(YIXI5_RESULTS_PATH)["selected_on_validation"]["config"]
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
    total = float(np.sum(gain))
    fractions = {
        column: (float(value) / total if total else 0.0)
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
    gain = model.get_booster().get_score(importance_type="gain")
    total = float(sum(gain.values()))
    fractions = {
        column: (float(gain.get(column, 0.0)) / total if total else 0.0)
        for column in columns
    }
    del Xtr, Xva
    gc.collect()
    return model, scores, metrics, fractions


def record(name, groups, columns, family, model, metrics, fractions):
    best_iteration = (
        int(model.best_iteration_) if family == "lgb" else int(model.best_iteration)
    )
    return {
        "name": name,
        "groups": list(groups),
        "columns": list(columns),
        "seed": 0,
        "best_iteration": best_iteration,
        "valid": metric_dict(metrics),
        "feature_gain_fraction": fractions,
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


def confirm_representation(
    family,
    frames,
    y,
    users,
    reference_columns,
    candidate_columns,
    seed0_reference,
    seed0_candidate,
):
    fit = fit_lgb if family == "lgb" else fit_xgb
    rows = [
        {
            "seed": 0,
            "reference_valid": metric_dict(seed0_reference),
            "candidate_valid": metric_dict(seed0_candidate),
            "delta": float(seed0_candidate["primary"] - seed0_reference["primary"]),
        }
    ]
    for seed in SEEDS[1:]:
        ref_model, ref_scores, ref_metrics, _ = fit(
            frames, y, users, reference_columns, seed
        )
        cand_model, cand_scores, cand_metrics, _ = fit(
            frames, y, users, candidate_columns, seed
        )
        delta = float(cand_metrics["primary"] - ref_metrics["primary"])
        rows.append(
            {
                "seed": seed,
                "reference_valid": metric_dict(ref_metrics),
                "candidate_valid": metric_dict(cand_metrics),
                "delta": delta,
            }
        )
        print(
            f"    seed={seed} ref={ref_metrics['primary']:.8f} "
            f"candidate={cand_metrics['primary']:.8f} delta={delta:+.8f}",
            flush=True,
        )
        del ref_model, ref_scores, cand_model, cand_scores
        gc.collect()
    return confirmation_summary(rows)
