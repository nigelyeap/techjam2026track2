"""Validation-only YIXI5 component and final-blend fidelity gate for 6f."""

from __future__ import annotations

import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "harness_results.json")
YIXI5_BLEND_PATH = os.path.join(
    common.REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "blend.py"
)


def main() -> None:
    print("=== loading causally verified cross-transfer frame ===", flush=True)
    frames, y, users, metadata = common.load_frames(common.DATA_DIR)
    if not metadata["causality"]["passed"]:
        raise AssertionError("causality gate failed")

    print("\n=== reproducing tuned YIXI5 XGBoost ===", flush=True)
    xgb_model, xgb_scores, xgb_metrics, _ = common.fit_xgb(
        frames, y, users, common.XGB_A0_COLUMNS, 0
    )
    if abs(xgb_metrics["primary"] - common.XGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError(
            f"YIXI5 XGBoost drift: {xgb_metrics['primary']} vs {common.XGB_REFERENCE_VALID}"
        )
    print(f"XGBoost PASSED: {xgb_metrics['primary']:.8f}", flush=True)

    print("\n=== reproducing iter63 LightGBM ===", flush=True)
    lgb_model, lgb_scores, lgb_metrics, _ = common.fit_lgb(
        frames, y, users, common.LGB_B0_COLUMNS, 0
    )
    if abs(lgb_metrics["primary"] - common.LGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError(
            f"iter63 LightGBM drift: {lgb_metrics['primary']} vs {common.LGB_REFERENCE_VALID}"
        )
    print(f"LightGBM PASSED: {lgb_metrics['primary']:.8f}", flush=True)

    print("\n=== reproducing unchanged FM ensemble ===", flush=True)
    yixi5_blend = common.load_module(YIXI5_BLEND_PATH, "yixi5_blend_for_yixi6_harness")
    fm_context, fm_scores, fm_metrics = yixi5_blend.fit_current_fm_validation(y, users)
    if abs(fm_metrics["primary"] - common.FM_REFERENCE_VALID) > 1e-8:
        raise AssertionError(
            f"FM drift: {fm_metrics['primary']} vs {common.FM_REFERENCE_VALID}"
        )
    print(f"FM PASSED: {fm_metrics['primary']:.8f}", flush=True)

    components = yixi5_blend.normalize_components(
        fm_scores, lgb_scores, xgb_scores, users["valid"], "within_user_percentile"
    )
    weights = {"fm": 0.24, "lgb": 0.40, "xgb": 0.36}
    blend_scores = yixi5_blend.combined_scores(components, weights)
    blend_metrics = common.evaluate(users["valid"], y["valid"], blend_scores)
    if abs(blend_metrics["primary"] - common.BLEND_REFERENCE_VALID) > 1e-8:
        raise AssertionError(
            f"YIXI5 blend drift: {blend_metrics['primary']} vs {common.BLEND_REFERENCE_VALID}"
        )
    print(f"YIXI5 final blend PASSED: {blend_metrics['primary']:.8f}", flush=True)

    results = {
        "experiment": "iterYIXI6_harness_fidelity",
        "environment": common.environment(),
        "feature_metadata": metadata,
        "components": {
            "fm_valid": common.metric_dict(fm_metrics),
            "lgb_valid": common.metric_dict(lgb_metrics),
            "xgb_valid": common.metric_dict(xgb_metrics),
        },
        "final_blend": {
            "normalization": "within_user_percentile",
            "weights": weights,
            "valid": common.metric_dict(blend_metrics),
        },
        "test_accessed": False,
        "passed": True,
    }
    common.write_json(RESULTS_PATH, results)
    print(f"wrote {RESULTS_PATH}", flush=True)

    # Keep references alive through scoring, then release them explicitly.
    del xgb_model, lgb_model, fm_context, xgb_scores, lgb_scores, fm_scores


if __name__ == "__main__":
    main()
