"""Inspect installed ranking-objective support before the Section 6g sweeps.

This is deliberately a tiny synthetic fit.  It verifies that each requested
parameter is accepted by the installed native library and records the
effective XGBoost objective configuration rather than relying on API memory.
"""

from __future__ import annotations

import json
import os
import platform
import warnings

import lightgbm as lgb
import numpy as np
import xgboost as xgb


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "support_results.json")


def tiny_data():
    X = np.asarray(
        [
            [0.0, 0.1],
            [0.2, 0.0],
            [0.3, 0.4],
            [1.0, 0.9],
            [0.8, 1.0],
            [0.7, 0.6],
        ],
        dtype=np.float32,
    )
    y = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.float32)
    groups = np.asarray([3, 3], dtype=np.uint32)
    return X, y, groups


def inspect_lightgbm():
    X, y, groups = tiny_data()
    cases = {
        "lambdarank_reference": {
            "objective": "lambdarank",
            "lambdarank_truncation_level": 30,
            "sigmoid": 1.0,
            "lambdarank_norm": True,
        },
        "lambdarank_requested_controls": {
            "objective": "lambdarank",
            "lambdarank_truncation_level": 5,
            "sigmoid": 0.5,
            "lambdarank_norm": False,
        },
        "rank_xendcg": {"objective": "rank_xendcg"},
    }
    records = {}
    for name, params in cases.items():
        caught = []
        with warnings.catch_warnings(record=True) as warning_rows:
            warnings.simplefilter("always")
            model = lgb.LGBMRanker(
                metric="ndcg",
                eval_at=[5],
                n_estimators=2,
                num_leaves=2,
                min_child_samples=1,
                verbosity=-1,
                **params,
            )
            model.fit(X, y, group=groups)
            caught = [str(row.message) for row in warning_rows]
        effective = model.booster_.params
        missing = [key for key in params if key not in effective]
        mismatched = {
            key: {"requested": value, "effective": effective.get(key)}
            for key, value in params.items()
            if key in effective and effective[key] != value
        }
        records[name] = {
            "supported": not missing and not mismatched,
            "requested": params,
            "effective": {
                key: effective.get(key)
                for key in (
                    "objective",
                    "lambdarank_truncation_level",
                    "sigmoid",
                    "lambdarank_norm",
                )
                if key in effective
            },
            "missing": missing,
            "mismatched": mismatched,
            "warnings": caught,
        }
    return records


def xgb_objective_config(model):
    config = json.loads(model.get_booster().save_config())
    return config["learner"]["objective"]


def inspect_xgboost():
    X, y, groups = tiny_data()
    cases = {
        "implicit_defaults": {},
        "mean_five_normalized": {
            "lambdarank_pair_method": "mean",
            "lambdarank_num_pair_per_sample": 5,
            "lambdarank_normalization": True,
        },
        "topk_five_unnormalized": {
            "lambdarank_pair_method": "topk",
            "lambdarank_num_pair_per_sample": 5,
            "lambdarank_normalization": False,
        },
    }
    records = {}
    for name, params in cases.items():
        with warnings.catch_warnings(record=True) as warning_rows:
            warnings.simplefilter("always")
            model = xgb.XGBRanker(
                objective="rank:ndcg",
                eval_metric="ndcg@5-",
                n_estimators=2,
                max_depth=1,
                tree_method="hist",
                n_jobs=-1,
                verbosity=0,
                **params,
            )
            model.fit(X, y, group=groups, verbose=False)
        objective = xgb_objective_config(model)
        records[name] = {
            "supported": True,
            "requested": params,
            "effective_objective_config": objective,
            "warnings": [str(row.message) for row in warning_rows],
        }
    return records


def main():
    results = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
        },
        "method": "successful synthetic native ranker fits plus effective model configuration inspection",
        "lightgbm": inspect_lightgbm(),
        "xgboost": inspect_xgboost(),
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
