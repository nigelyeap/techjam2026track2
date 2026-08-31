"""Mandatory shifted-split eligibility gate for the adaptive blend."""

from __future__ import annotations

import os

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_B_PATH = os.path.join(THIS_DIR, "phase_b_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "phase_c_results.json")


def main():
    phase_b = common.read_json(PHASE_B_PATH)
    if phase_b["eligible_for_confirmation"]:
        raise RuntimeError(
            "adaptive candidate is promotion-eligible; shifted refit must be implemented/run"
        )
    _, shifted_metadata = common.regimes.build(common.regimes.SHIFTED)
    delta = phase_b["selected_on_validation"]["delta_vs_fixed_reference"]
    results = {
        "experiment": "iterYIXI11_phase_c_shifted_robustness_gate",
        "predeclared_shift": common.regimes.SHIFTED,
        "shifted_regime_metadata": shifted_metadata,
        "performed": False,
        "eligible": False,
        "reason": (
            f"official adaptive delta {delta:+.8f} is below the +0.0003 "
            "preliminary gate and +0.001 confirmation threshold"
        ),
        "selection_effect": "no promotable adaptive system",
        "test_access": "none",
    }
    common.write_json(RESULTS_PATH, results)
    print(
        f"shifted refit gated off: official delta={delta:+.8f}; "
        f"shifted train threshold={shifted_metadata['threshold']}", flush=True
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
