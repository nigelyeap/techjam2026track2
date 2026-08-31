"""Phase B: gated weight calibration on frozen validation predictions.

Only Phase A transforms with >=+0.0003 at the fixed reference weights may be
optimized.  The percentile reference grid is always rerun as a fidelity and
nearby-plateau check; it cannot introduce a new candidate.
"""

from __future__ import annotations

import os

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_A_PATH = os.path.join(THIS_DIR, "phase_a_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "phase_b_results.json")


def main():
    phase_a = common.read_json(PHASE_A_PATH)
    frozen = common.load_frozen()
    users, labels = frozen["users"], frozen["labels"]
    y5_raw = {
        "fm": frozen["fm"],
        "lgb": frozen["yixi5_lgb"],
        "xgb": frozen["xgb"],
    }
    y7_raw = {
        "fm": frozen["fm"],
        "lgb": frozen["yixi7_lgb"],
        "xgb": frozen["xgb"],
    }
    y5_states = common.build_rank_states(y5_raw, users)
    y7_states = common.build_rank_states(y7_raw, users)
    percentile_map = common.common_transform_map("percentile")

    print("=== percentile reference weight-grid fidelity ===", flush=True)
    reference_components = common.transformed_components(y5_states, percentile_map)
    reference_best, reference_overall, reference_records = common.score_weight_grid(
        reference_components, users, labels, percentile_map
    )
    if abs(reference_best["valid"]["primary"] - common.YIXI5_VALID) > 1e-8:
        raise AssertionError("reference weight-grid score drift")
    if reference_best["weights"] != common.YIXI5_WEIGHTS:
        raise AssertionError(
            f"reference weight-grid weights drift: {reference_best['weights']}"
        )
    print(
        f"reference PASSED weights={reference_best['weights']} "
        f"valid={reference_best['valid']['primary']:.8f}",
        flush=True,
    )

    calibrated = {}
    for transform_name in phase_a["promising_transforms"]:
        transform_map = common.common_transform_map(transform_name)
        components = common.transformed_components(y5_states, transform_map)
        best, overall, records = common.score_weight_grid(
            components, users, labels, transform_map
        )
        best["delta_vs_yixi5_reference"] = float(
            best["valid"]["primary"] - common.YIXI5_VALID
        )
        calibrated[transform_name] = {
            "best_three_model": best,
            "best_including_ablations": overall,
            "nearby_plateau": common.top_nearby(records),
            "records": records,
        }
        print(
            f"{transform_name}: weights={best['weights']} "
            f"valid={best['valid']['primary']:.8f} "
            f"delta={best['delta_vs_yixi5_reference']:+.8f}",
            flush=True,
        )

    eligible_rows = [
        (name, payload["best_three_model"])
        for name, payload in calibrated.items()
        if payload["best_three_model"]["delta_vs_yixi5_reference"]
        >= common.PRELIMINARY_DELTA
    ]
    if eligible_rows:
        selected_name, selected = max(
            eligible_rows, key=lambda item: item[1]["valid"]["primary"]
        )
        selected_map = common.common_transform_map(selected_name)
        selected_is_new = True
    else:
        selected_name, selected = "percentile", reference_best
        selected_map = percentile_map
        selected_is_new = False

    # A deliberately tiny model-specific refinement is permitted only after
    # a common non-reference transform clears the preliminary threshold.
    model_specific = {
        "performed": False,
        "gate": "common calibrated transform delta >= +0.0003",
        "reason": (
            None
            if selected_is_new
            else "no common non-reference transform cleared the gate"
        ),
        "predeclared_candidates": (
            "hold the selected common transform on two components and restore percentile on exactly one component"
        ),
    }

    # Current-best transfer is also gated to a genuine selected transform.
    y7_reference_components = common.transformed_components(y7_states, percentile_map)
    _, y7_reference_metrics = common.score_components(
        y7_reference_components, common.YIXI7_WEIGHTS, users, labels
    )
    if abs(y7_reference_metrics["primary"] - common.YIXI7_VALID) > 1e-8:
        raise AssertionError("YIXI7 current-best transfer reference drift")
    current_best_transfer = {
        "performed": False,
        "gate": "selected non-reference common/model-specific transform",
        "reason": (
            None if selected_is_new else "percentile itself remained selected"
        ),
        "reference": {
            "weights": common.YIXI7_WEIGHTS,
            "valid": common.metric_dict(y7_reference_metrics),
        },
    }

    results = {
        "experiment": "iterYIXI8_phase_b_gated_weight_calibration",
        "selection_policy": {
            "selector": "official validation primary only",
            "preliminary_delta": common.PRELIMINARY_DELTA,
            "weight_grid": "coarse 0.10 simplex plus local 0.02 refinement",
            "test_access": "none",
        },
        "phase_a_promising_transforms": phase_a["promising_transforms"],
        "reference_weight_grid": {
            "best_three_model": reference_best,
            "best_including_ablations": reference_overall,
            "nearby_plateau": common.top_nearby(reference_records),
            "records": reference_records,
        },
        "calibrated_common_transforms": calibrated,
        "model_specific_refinement": model_specific,
        "current_best_transfer": current_best_transfer,
        "selected_on_validation": {
            "name": selected_name,
            "transform_by_model": selected_map,
            "weights": selected["weights"],
            "valid": selected["valid"],
            "delta_vs_yixi5_reference": float(
                selected["valid"]["primary"] - common.YIXI5_VALID
            ),
            "delta_vs_yixi7_current_best": float(
                selected["valid"]["primary"] - common.YIXI7_VALID
            ),
            "is_new_candidate": selected_is_new,
        },
        "five_seed_confirmation": {
            "performed": False,
            "reason": "no new candidate cleared +0.001 after the preliminary gate",
        },
        "verdict": "REJECT",
    }
    common.write_json(RESULTS_PATH, results)
    print(
        f"selected={selected_name} valid={selected['valid']['primary']:.8f} "
        "verdict=REJECT",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
