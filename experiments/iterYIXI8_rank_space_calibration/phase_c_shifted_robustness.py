"""Phase C gate for temporal robustness of the selected calibration.

A shifted refit is meaningful only for a non-reference transformation.  If
Phase B retains the exact percentile reference, the candidate/reference
calibrations are mathematically identical on every possible date split and
there is no new effect to verify by expensive model retraining.
"""

from __future__ import annotations

import os

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_B_PATH = os.path.join(THIS_DIR, "phase_b_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "phase_c_results.json")
SHIFTED_SPLIT = {
    "train": (20220405, 20220418),
    "valid": (20220419, 20220425),
    "test": (20220426, 20220505),
}


def main():
    phase_b = common.read_json(PHASE_B_PATH)
    selected = phase_b["selected_on_validation"]
    is_reference_identity = bool(
        selected["name"] == "percentile"
        and selected["transform_by_model"]
        == common.common_transform_map("percentile")
        and selected["weights"] == common.YIXI5_WEIGHTS
    )
    if selected["is_new_candidate"]:
        raise RuntimeError(
            "a new candidate exists; implement/run the shifted refit before finalizing"
        )
    if not is_reference_identity:
        raise AssertionError("non-candidate Phase B output is not the exact reference")

    results = {
        "experiment": "iterYIXI8_phase_c_temporal_robustness_gate",
        "predeclared_shift": SHIFTED_SPLIT,
        "performed": False,
        "eligible": False,
        "reason": (
            "no non-reference transform passed the +0.0003 preliminary gate; "
            "the selected calibration is exactly the percentile reference"
        ),
        "identity_check": {
            "same_transform_by_model": True,
            "same_weights": True,
            "candidate_minus_reference_on_any_split": 0.0,
        },
        "selection_effect": "none",
        "test_access": "none",
    }
    common.write_json(RESULTS_PATH, results)
    print(
        "shifted refit gated off: selected calibration is exactly the reference; "
        "delta is identically zero on any split",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
