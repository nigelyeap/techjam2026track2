"""Exact post-6h standalone and ensemble fidelity gate for Section 6i."""

from __future__ import annotations

import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "harness_results.json")


def main():
    frames, y, users, metadata = common.features.load_frames()
    yixi8 = common.load_module(common.YIXI8_COMMON_PATH, "yixi8_common_for_yixi9")
    frozen = yixi8.load_frozen()
    if not np.array_equal(frozen["users"], np.asarray(users["valid"])):
        raise AssertionError("YIXI8/YIXI9 validation users differ")
    if not np.array_equal(frozen["labels"], y["valid"]):
        raise AssertionError("YIXI8/YIXI9 validation labels differ")

    print("=== strongest post-6h LightGBM ===", flush=True)
    lgb_model, lgb_scores, lgb_metrics, _ = common.fit_lgb(
        frames, y, users, common.LGB_REFERENCE_COLUMNS, 0
    )
    if abs(lgb_metrics["primary"] - common.LGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("LightGBM reference drift")
    print(f"LGB PASSED={lgb_metrics['primary']:.8f}", flush=True)

    print("=== strongest post-6h standalone XGBoost ===", flush=True)
    xgb_model, xgb_scores, xgb_metrics, _ = common.fit_xgb(
        frames, y, users, common.XGB_REFERENCE_COLUMNS, 0
    )
    if abs(xgb_metrics["primary"] - common.XGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("XGBoost reference drift")
    print(f"XGB PASSED={xgb_metrics['primary']:.8f}", flush=True)

    current_raw = {
        "fm": frozen["fm"],
        "lgb": frozen["yixi7_lgb"],
        "xgb": frozen["xgb"],
    }
    states = yixi8.build_rank_states(current_raw, frozen["users"])
    components = yixi8.transformed_components(
        states, yixi8.common_transform_map("percentile")
    )
    _, ensemble_metrics = yixi8.score_components(
        components,
        common.CURRENT_WEIGHTS,
        frozen["users"],
        frozen["labels"],
    )
    if abs(ensemble_metrics["primary"] - common.ENSEMBLE_REFERENCE_VALID) > 1e-8:
        raise AssertionError("ensemble reference drift")
    print(f"ensemble PASSED={ensemble_metrics['primary']:.8f}", flush=True)

    results = {
        "experiment": "iterYIXI9_post6h_harness",
        "environment": common.environment(),
        "feature_metadata": metadata,
        "references": {
            "lightgbm": common.metric_dict(lgb_metrics),
            "xgboost": common.metric_dict(xgb_metrics),
            "ensemble": common.metric_dict(ensemble_metrics),
        },
        "tolerance": 1e-8,
        "passed": True,
        "test_metrics_computed": False,
    }
    common.write_json(RESULTS_PATH, results)
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
