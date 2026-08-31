"""Fit each base component once and freeze validation predictions for 6h.

This runner never predicts test.  Every later transform/weight search reads
only the immutable arrays emitted here and performs no model training.
"""

from __future__ import annotations

import gc
import os

import numpy as np

import common


def main():
    yixi7 = common.load_module(common.YIXI7_COMMON_PATH, "yixi7_common_for_yixi8")
    yixi5_blend = common.load_module(
        common.YIXI5_BLEND_PATH, "yixi5_blend_for_yixi8_freeze"
    )
    phase_a = common.read_json(common.YIXI7_PHASE_A_PATH)
    selected_lgb_rank = phase_a["selected_on_validation"]["rank_config"]
    frames, y, users, feature_metadata = yixi7.load_frames()

    print("=== fitting validation components exactly once ===", flush=True)
    y5_lgb_model, y5_lgb_scores, y5_lgb_metrics = yixi7.fit_lgb_current(
        frames, y, users, 0
    )
    xgb_model, xgb_scores, xgb_metrics = yixi7.fit_xgb_current(
        frames, y, users, 0
    )
    y7_lgb_model, y7_lgb_scores, y7_lgb_metrics, _ = yixi7.fit_lgb(
        frames, y, users, selected_lgb_rank, 0
    )
    fm_context, fm_scores, fm_metrics = yixi5_blend.fit_current_fm_validation(y, users)

    expected_components = {
        "fm": common.FM_VALID,
        "yixi5_lgb": 0.6716787219047546,
        "xgb": 0.6675541996955872,
        "yixi7_lgb": 0.6768913269042969,
    }
    actual_components = {
        "fm": fm_metrics["primary"],
        "yixi5_lgb": y5_lgb_metrics["primary"],
        "xgb": xgb_metrics["primary"],
        "yixi7_lgb": y7_lgb_metrics["primary"],
    }
    for name, expected in expected_components.items():
        if abs(actual_components[name] - expected) > 1e-8:
            raise AssertionError(
                f"{name} fidelity drift: {actual_components[name]} vs {expected}"
            )

    frozen = {
        "users": np.asarray(users["valid"]),
        "labels": np.asarray(y["valid"], dtype=np.float32),
        "fm": np.asarray(fm_scores, dtype=np.float64),
        "yixi5_lgb": np.asarray(y5_lgb_scores, dtype=np.float64),
        "xgb": np.asarray(xgb_scores, dtype=np.float64),
        "yixi7_lgb": np.asarray(y7_lgb_scores, dtype=np.float64),
    }
    lengths = {len(array) for array in frozen.values()}
    if lengths != {124909}:
        raise AssertionError(f"unexpected frozen lengths: {lengths}")

    y5_raw = {"fm": frozen["fm"], "lgb": frozen["yixi5_lgb"], "xgb": frozen["xgb"]}
    y7_raw = {"fm": frozen["fm"], "lgb": frozen["yixi7_lgb"], "xgb": frozen["xgb"]}
    y5_states = common.build_rank_states(y5_raw, frozen["users"])
    y7_states = common.build_rank_states(y7_raw, frozen["users"])
    percentile_map = common.common_transform_map("percentile")
    _, y5_metrics = common.score_components(
        common.transformed_components(y5_states, percentile_map),
        common.YIXI5_WEIGHTS,
        frozen["users"],
        frozen["labels"],
    )
    _, y7_metrics = common.score_components(
        common.transformed_components(y7_states, percentile_map),
        common.YIXI7_WEIGHTS,
        frozen["users"],
        frozen["labels"],
    )
    if abs(y5_metrics["primary"] - common.YIXI5_VALID) > 1e-8:
        raise AssertionError("YIXI5 frozen ensemble fidelity failed")
    if abs(y7_metrics["primary"] - common.YIXI7_VALID) > 1e-8:
        raise AssertionError("YIXI7 frozen ensemble fidelity failed")

    np.savez_compressed(common.FROZEN_PATH, **frozen)
    results = {
        "experiment": "iterYIXI8_frozen_validation_predictions",
        "environment": common.environment(),
        "selection_role": "none; immutable base-prediction harness",
        "test_predictions_computed": False,
        "components_valid": {
            "fm": common.metric_dict(fm_metrics),
            "yixi5_lgb": common.metric_dict(y5_lgb_metrics),
            "xgb": common.metric_dict(xgb_metrics),
            "yixi7_lgb": common.metric_dict(y7_lgb_metrics),
        },
        "references": {
            "requested_yixi5": {
                "weights": common.YIXI5_WEIGHTS,
                "valid": common.metric_dict(y5_metrics),
            },
            "current_yixi7": {
                "weights": common.YIXI7_WEIGHTS,
                "valid": common.metric_dict(y7_metrics),
            },
        },
        "yixi7_lgb_rank_config": selected_lgb_rank,
        "feature_metadata": feature_metadata,
        "array_sha256": {
            name: common.array_sha256(array) for name, array in frozen.items()
        },
        "frozen_path": os.path.basename(common.FROZEN_PATH),
        "fidelity_tolerance": 1e-8,
        "fidelity_passed": True,
    }
    common.write_json(common.FROZEN_RESULTS_PATH, results)
    print(
        f"YIXI5 PASSED={y5_metrics['primary']:.8f} "
        f"YIXI7 PASSED={y7_metrics['primary']:.8f}",
        flush=True,
    )
    print(f"wrote immutable validation cache {common.FROZEN_PATH}", flush=True)
    del (
        y5_lgb_model,
        xgb_model,
        y7_lgb_model,
        fm_context,
        frames,
        y,
        users,
    )
    gc.collect()


if __name__ == "__main__":
    main()
