"""Shared frozen-score transforms and validation utilities for Section 6h."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
FROZEN_PATH = os.path.join(THIS_DIR, ".frozen_valid_predictions.npz")
FROZEN_RESULTS_PATH = os.path.join(THIS_DIR, "frozen_predictions.json")
YIXI7_COMMON_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI7_ranking_objective_tuning", "common.py"
)
YIXI7_PHASE_A_PATH = os.path.join(
    REPO_ROOT,
    "experiments",
    "iterYIXI7_ranking_objective_tuning",
    "phase_a_results.json",
)
YIXI5_BLEND_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "blend.py"
)

sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402


MODELS = ("fm", "lgb", "xgb")
YIXI5_WEIGHTS = {"fm": 0.24, "lgb": 0.40, "xgb": 0.36}
YIXI7_WEIGHTS = {"fm": 0.24, "lgb": 0.42, "xgb": 0.34}
YIXI5_VALID = 0.6823752522468567
YIXI5_TEST = 0.6694862246513367
YIXI7_VALID = 0.6841553449630737
YIXI7_TEST = 0.6743725538253784
FM_VALID = 0.6398779153823853
PRELIMINARY_DELTA = 0.0003
PROMOTION_DELTA = 0.001
SEEDS = (0, 1, 2, 3, 4)
LOGIT_EPS = 1e-3

# Fixed before validation is inspected.  Every transform is strictly
# monotonic in a model's within-user rank (apart from preserving existing
# exact-score ties through average ranks).
TRANSFORMS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("percentile", {"family": "percentile"}),
    ("power_0.5", {"family": "power", "exponent": 0.5}),
    ("power_1.5", {"family": "power", "exponent": 1.5}),
    ("power_2", {"family": "power", "exponent": 2.0}),
    ("power_3", {"family": "power", "exponent": 3.0}),
    ("clipped_logit", {"family": "clipped_logit", "eps": LOGIT_EPS}),
    ("reciprocal_rank", {"family": "reciprocal_rank"}),
    ("ndcg_rank", {"family": "ndcg_rank"}),
)
TRANSFORM_MAP = dict(TRANSFORMS)


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


def read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def environment() -> dict[str, str]:
    import lightgbm as lgb
    import xgboost as xgb

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": lgb.__version__,
        "xgboost": xgb.__version__,
    }


def array_sha256(values) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_frozen() -> dict[str, np.ndarray]:
    if not os.path.exists(FROZEN_PATH) or not os.path.exists(FROZEN_RESULTS_PATH):
        raise FileNotFoundError("run freeze_predictions.py first")
    payload = {key: value for key, value in np.load(FROZEN_PATH).items()}
    metadata = read_json(FROZEN_RESULTS_PATH)
    for key, expected in metadata["array_sha256"].items():
        actual = array_sha256(payload[key])
        if actual != expected:
            raise AssertionError(f"frozen array hash drift for {key}")
    return payload


def rank_state(scores, users) -> dict[str, np.ndarray]:
    """Label-free average ranks computed independently inside each user."""
    values = pd.Series(np.asarray(scores, dtype=np.float64), copy=False)
    groups = pd.Series(np.asarray(users), copy=False)
    grouped = values.groupby(groups, sort=False)
    percentile = grouped.rank(method="average", pct=True, ascending=True)
    descending_rank = grouped.rank(method="average", ascending=False)
    group_size = groups.groupby(groups, sort=False).transform("size")
    state = {
        "percentile": percentile.to_numpy(dtype=np.float64),
        "descending_rank": descending_rank.to_numpy(dtype=np.float64),
        "group_size": group_size.to_numpy(dtype=np.float64),
    }
    if not all(np.all(np.isfinite(array)) for array in state.values()):
        raise AssertionError("non-finite rank state")
    return state


def _rescale_rank_curve(values, bottom, group_size):
    denominator = 1.0 - bottom
    output = np.ones_like(values, dtype=np.float64)
    mask = group_size > 1
    output[mask] = (values[mask] - bottom[mask]) / denominator[mask]
    return np.clip(output, 0.0, 1.0)


def apply_transform(state, name: str) -> np.ndarray:
    if name not in TRANSFORM_MAP:
        raise ValueError(f"unknown transform {name}")
    spec = TRANSFORM_MAP[name]
    p = state["percentile"]
    rank = state["descending_rank"]
    group_size = state["group_size"]
    family = spec["family"]
    if family == "percentile":
        output = p.copy()
    elif family == "power":
        output = np.power(p, spec["exponent"])
    elif family == "clipped_logit":
        eps = float(spec["eps"])
        raw = np.log((p + eps) / (1.0 - p + eps))
        lo = np.log(eps / (1.0 + eps))
        hi = np.log((1.0 + eps) / eps)
        output = (raw - lo) / (hi - lo)
    elif family == "reciprocal_rank":
        raw = 1.0 / rank
        bottom = 1.0 / group_size
        output = _rescale_rank_curve(raw, bottom, group_size)
    elif family == "ndcg_rank":
        raw = 1.0 / np.log2(rank + 1.0)
        bottom = 1.0 / np.log2(group_size + 1.0)
        output = _rescale_rank_curve(raw, bottom, group_size)
    else:
        raise ValueError(family)
    if not np.all(np.isfinite(output)):
        raise AssertionError(f"non-finite output for {name}")
    return output


def build_rank_states(raw_components, users):
    return {name: rank_state(raw_components[name], users) for name in MODELS}


def transformed_components(rank_states, transform_by_model):
    return {
        model: apply_transform(rank_states[model], transform_by_model[model])
        for model in MODELS
    }


def common_transform_map(name: str) -> dict[str, str]:
    return {model: name for model in MODELS}


def combined_scores(components, weights):
    return sum(float(weights[name]) * components[name] for name in MODELS)


def score_components(components, weights, users, labels):
    scores = combined_scores(components, weights)
    return scores, evaluate(users, labels, scores)


def score_weight_grid(components, users, labels, transform_by_model):
    """Established 0.10 simplex plus local 0.02 three-model refinement."""
    records = []
    seen = set()

    def assess(fm_weight, lgb_weight, xgb_weight, stage):
        key = (
            round(float(fm_weight), 8),
            round(float(lgb_weight), 8),
            round(float(xgb_weight), 8),
        )
        if key in seen:
            return
        seen.add(key)
        weights = {"fm": key[0], "lgb": key[1], "xgb": key[2]}
        _, metrics = score_components(components, weights, users, labels)
        records.append(
            {
                "stage": stage,
                "transform_by_model": dict(transform_by_model),
                "weights": weights,
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
            if all(row["weights"][name] > 0 for name in MODELS)
        ),
        key=lambda row: row["valid"]["primary"],
    )
    center_fm = int(round(coarse_three["weights"]["fm"] * 50))
    center_xgb = int(round(coarse_three["weights"]["xgb"] * 50))
    for fm_units in range(max(1, center_fm - 5), min(49, center_fm + 5) + 1):
        for xgb_units in range(
            max(1, center_xgb - 5), min(49 - fm_units, center_xgb + 5) + 1
        ):
            lgb_units = 50 - fm_units - xgb_units
            if lgb_units >= 1:
                assess(
                    fm_units / 50,
                    lgb_units / 50,
                    xgb_units / 50,
                    "refine_0.02",
                )
    best_three = max(
        (
            row
            for row in records
            if all(row["weights"][name] > 0 for name in MODELS)
        ),
        key=lambda row: row["valid"]["primary"],
    )
    best_overall = max(records, key=lambda row: row["valid"]["primary"])
    return best_three, best_overall, records


def unique_stats(scores, users):
    values = np.asarray(scores)
    user_values = np.asarray(users)
    frame = pd.DataFrame({"user": user_values, "score": values})
    grouped = frame.groupby("user", sort=False)["score"]
    fractions = grouped.nunique(dropna=False) / grouped.size()
    return {
        "rows": int(len(values)),
        "unique_scores_overall": int(len(np.unique(values))),
        "mean_per_user_unique_fraction": float(fractions.mean()),
    }


def top_nearby(records, limit=12):
    rows = [
        row
        for row in records
        if all(row["weights"][name] > 0 for name in MODELS)
    ]
    return sorted(rows, key=lambda row: row["valid"]["primary"], reverse=True)[:limit]
