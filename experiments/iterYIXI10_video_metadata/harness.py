"""Exact YIXI9 component/final-ensemble fidelity gate for Section 6j."""

from __future__ import annotations

import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "harness_results.json")


def main():
    frames, y, users, metadata = common.features.load_frames()
    print("=== current promoted LightGBM ===", flush=True)
    lgb_model, lgb_scores, lgb_metrics, _ = common.fit_lgb(
        frames, y, users, common.LGB_REFERENCE_COLUMNS, 0
    )
    if abs(lgb_metrics["primary"] - common.LGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("YIXI9 LightGBM reference drift")
    print(f"LGB PASSED={lgb_metrics['primary']:.8f}", flush=True)

    print("=== current final-ensemble XGBoost ===", flush=True)
    xgb_model, xgb_scores, xgb_metrics, _ = common.fit_xgb(
        frames, y, users, common.XGB_REFERENCE_COLUMNS, 0
    )
    if abs(xgb_metrics["primary"] - common.XGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("YIXI9 XGBoost reference drift")
    print(f"XGB PASSED={xgb_metrics['primary']:.8f}", flush=True)

    yixi8 = common.load_module(common.YIXI8_COMMON_PATH, "yixi8_common_for_yixi10")
    blend = common.load_module(common.YIXI5_BLEND_PATH, "yixi5_blend_for_yixi10_harness")
    frozen = yixi8.load_frozen()
    if not np.array_equal(frozen["users"], np.asarray(users["valid"])):
        raise AssertionError("frozen/native users differ")
    if not np.array_equal(frozen["labels"], y["valid"]):
        raise AssertionError("frozen/native labels differ")
    fm_metrics = common.evaluate(users["valid"], y["valid"], frozen["fm"])
    if abs(fm_metrics["primary"] - common.FM_REFERENCE_VALID) > 1e-8:
        raise AssertionError("FM reference drift")
    components = blend.normalize_components(
        frozen["fm"], lgb_scores, xgb_scores, users["valid"], "within_user_percentile"
    )
    ensemble_scores = blend.combined_scores(components, common.CURRENT_WEIGHTS)
    ensemble_metrics = common.evaluate(users["valid"], y["valid"], ensemble_scores)
    if abs(ensemble_metrics["primary"] - common.ENSEMBLE_REFERENCE_VALID) > 1e-8:
        raise AssertionError("YIXI9 ensemble reference drift")
    print(f"ensemble PASSED={ensemble_metrics['primary']:.8f}", flush=True)

    common.write_json(
        RESULTS_PATH,
        {
            "experiment": "iterYIXI10_yixi9_reference_harness",
            "environment": common.environment(),
            "feature_metadata": metadata,
            "references": {
                "lightgbm": common.metric_dict(lgb_metrics),
                "xgboost": common.metric_dict(xgb_metrics),
                "fm": common.metric_dict(fm_metrics),
                "ensemble": common.metric_dict(ensemble_metrics),
            },
            "tolerance": 1e-8,
            "passed": True,
            "test_metrics_computed": False,
        },
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
