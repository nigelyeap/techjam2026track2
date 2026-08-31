"""Fit/freeze exact YIXI10 validation components and reference blend."""

from __future__ import annotations

import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "harness_results.json")


def main():
    frames, y, users, feature_metadata = common.features.load_frames(common.DATA_DIR)
    lgb_model, lgb_scores, lgb_metrics = common.fit_lgb(frames, y, users, 0)
    xgb_model, xgb_scores, xgb_metrics = common.fit_xgb(frames, y, users, 0)
    if abs(lgb_metrics["primary"] - common.LGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("YIXI10 LightGBM drift")
    if abs(xgb_metrics["primary"] - common.XGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("YIXI10 XGBoost drift")
    yixi8 = common.load_module(common.YIXI8_COMMON_PATH, "yixi8_common_for_yixi11")
    prior = yixi8.load_frozen()
    if not np.array_equal(prior["users"], np.asarray(users["valid"])):
        raise AssertionError("FM/native users differ")
    if not np.array_equal(prior["labels"], y["valid"]):
        raise AssertionError("FM/native labels differ")
    fm_scores = prior["fm"]
    fm_metrics = common.evaluate(users["valid"], y["valid"], fm_scores)
    if abs(fm_metrics["primary"] - common.FM_REFERENCE_VALID) > 1e-8:
        raise AssertionError("FM drift")

    row_regimes, regime_metadata = common.regimes.build(
        common.regimes.OFFICIAL, expected_users=users["valid"]
    )
    raw = {"fm": fm_scores, "lgb": lgb_scores, "xgb": xgb_scores}
    components = common.percentile_components(raw, users["valid"])
    ensemble_scores, ensemble_metrics = common.score_components(
        components, common.GLOBAL_WEIGHTS, users["valid"], y["valid"]
    )
    if abs(ensemble_metrics["primary"] - common.ENSEMBLE_REFERENCE_VALID) > 1e-8:
        raise AssertionError("YIXI10 ensemble drift")

    arrays = {
        "users": np.asarray(users["valid"]),
        "labels": np.asarray(y["valid"]),
        "regimes": row_regimes,
        "fm": np.asarray(fm_scores, dtype=np.float64),
        "lgb": np.asarray(lgb_scores, dtype=np.float64),
        "xgb": np.asarray(xgb_scores, dtype=np.float64),
    }
    np.savez_compressed(common.PREDICTIONS_PATH, **arrays)
    results = {
        "experiment": "iterYIXI11_yixi10_prediction_harness",
        "environment": common.environment(),
        "feature_metadata": feature_metadata,
        "regime_metadata": regime_metadata,
        "references": {
            "fm": common.metric_dict(fm_metrics),
            "lgb": common.metric_dict(lgb_metrics),
            "xgb": common.metric_dict(xgb_metrics),
            "ensemble": common.metric_dict(ensemble_metrics),
            "weights": common.GLOBAL_WEIGHTS,
        },
        "array_sha256": {
            name: common.array_sha256(array) for name, array in arrays.items()
        },
        "test_metrics_computed": False,
        "passed": True,
    }
    common.write_json(RESULTS_PATH, results)
    print(
        f"LGB={lgb_metrics['primary']:.8f} XGB={xgb_metrics['primary']:.8f} "
        f"ensemble={ensemble_metrics['primary']:.8f} threshold={regime_metadata['threshold']}",
        flush=True,
    )
    print(f"wrote {common.PREDICTIONS_PATH}", flush=True)
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
