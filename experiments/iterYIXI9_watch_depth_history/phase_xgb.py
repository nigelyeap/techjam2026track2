"""Independent XGBoost tests for the four causal watch-depth features."""

from __future__ import annotations

import gc
import os

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "phase_xgb_results.json")


def main():
    frames, y, users, metadata = common.features.load_frames()
    print("=== XGBoost reference fidelity ===", flush=True)
    ref_model, ref_scores, ref_metrics, ref_gain = common.fit_xgb(
        frames, y, users, common.XGB_REFERENCE_COLUMNS, 0
    )
    if abs(ref_metrics["primary"] - common.XGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("XGBoost phase reference drift")
    reference = common.record(
        "X0_reference", None, common.XGB_REFERENCE_COLUMNS, ref_model, ref_metrics, ref_gain, "xgb"
    )
    reference["ties"] = common.unique_stats(ref_scores, users["valid"])
    candidates = []
    seed0_artifacts = {}
    for position, feature in enumerate(common.features.FEATURES, start=1):
        columns = common.XGB_REFERENCE_COLUMNS + [feature]
        model, scores, metrics, gain = common.fit_xgb(frames, y, users, columns, 0)
        row = common.record(
            f"X{position}_{feature}", feature, columns, model, metrics, gain, "xgb"
        )
        row["delta_vs_reference"] = float(
            metrics["primary"] - ref_metrics["primary"]
        )
        row["metric_delta_vs_reference"] = {
            name: float(metrics[name] - ref_metrics[name])
            for name in ("GAUC", "nDCG@5", "primary")
        }
        row["ties"] = common.unique_stats(scores, users["valid"])
        row["clears_preliminary"] = bool(
            row["delta_vs_reference"] >= common.PRELIMINARY_DELTA
        )
        candidates.append(row)
        seed0_artifacts[feature] = (model, scores, metrics)
        print(
            f"  {feature}: valid={metrics['primary']:.8f} "
            f"delta={row['delta_vs_reference']:+.8f}",
            flush=True,
        )

    confirmations = {}
    for row in candidates:
        if row["delta_vs_reference"] >= common.PROMOTION_DELTA:
            print(f"=== confirming {row['feature']} ===", flush=True)
            _, _, metrics = seed0_artifacts[row["feature"]]
            confirmations[row["feature"]] = common.confirm_feature(
                "xgb",
                frames,
                y,
                users,
                common.XGB_REFERENCE_COLUMNS,
                row["columns"],
                common.metric_dict(ref_metrics),
                common.metric_dict(metrics),
            )
        else:
            confirmations[row["feature"]] = {
                "performed": False,
                "confirmed": False,
                "reason": "seed-0 delta below +0.001 confirmation threshold",
            }
    confirmed_features = [
        feature for feature, result in confirmations.items() if result["confirmed"]
    ]
    selected = max(
        [reference] + candidates, key=lambda row: row["valid"]["primary"]
    )
    results = {
        "experiment": "iterYIXI9_independent_xgboost_watch_features",
        "environment": common.environment(),
        "fixed_config": common.selected_xgb_config(),
        "feature_metadata": metadata,
        "selection_policy": {
            "selector": "official validation primary only",
            "features_tested_independently": True,
            "preliminary_delta": common.PRELIMINARY_DELTA,
            "confirmation_delta": common.PROMOTION_DELTA,
            "test_access": "none",
        },
        "reference": reference,
        "candidates": candidates,
        "confirmations": confirmations,
        "confirmed_features": confirmed_features,
        "selected_on_validation": selected,
    }
    common.write_json(RESULTS_PATH, results)
    print(f"confirmed_features={confirmed_features}", flush=True)
    print(f"wrote {RESULTS_PATH}", flush=True)
    del ref_model, ref_scores
    for model, scores, _ in seed0_artifacts.values():
        del model, scores
    gc.collect()


if __name__ == "__main__":
    main()
