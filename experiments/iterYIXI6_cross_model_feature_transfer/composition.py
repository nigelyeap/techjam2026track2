"""6f gated standalone composition and final representation selection.

The B/C union is fitted only when both changes independently clear +0.0003.
Phase A's mutually exclusive A1/A2 choice is already made within Phase A.
"""

from __future__ import annotations

import gc
import json
import os

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_A_PATH = os.path.join(THIS_DIR, "phase_a_results.json")
PHASE_B_PATH = os.path.join(THIS_DIR, "phase_b_results.json")
PHASE_C_PATH = os.path.join(THIS_DIR, "phase_c_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "representation_results.json")


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    for path in (PHASE_A_PATH, PHASE_B_PATH, PHASE_C_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"run all independent phases first: missing {path}")
    phase_a = read_json(PHASE_A_PATH)
    phase_b = read_json(PHASE_B_PATH)
    phase_c = read_json(PHASE_C_PATH)

    selected_xgb = phase_a["selected_representation"]
    lgb_options = [phase_b["reference"]]
    if phase_b["preliminary_pass"]:
        lgb_options.append(phase_b["selected_representation"])
    if phase_c["preliminary_pass"]:
        lgb_options.append(phase_c["selected_representation"])

    results = {
        "experiment": "iterYIXI6_gated_composition",
        "selection_policy": {
            "selector": "standalone official validation primary only",
            "composition_gate": "both B and C independently >= +0.0003",
            "test_access": "none",
        },
        "phase_summary": {
            "A": {
                "winner": phase_a["winner"]["name"],
                "delta": phase_a["winner"]["delta_vs_A0"],
                "preliminary_pass": phase_a["preliminary_pass"],
            },
            "B": {
                "winner": phase_b["winner"]["name"],
                "delta": phase_b["winner"]["delta_vs_B0"],
                "preliminary_pass": phase_b["preliminary_pass"],
            },
            "C": {
                "winner": phase_c["candidate"]["name"],
                "delta": phase_c["candidate"]["delta_vs_C0"],
                "preliminary_pass": phase_c["preliminary_pass"],
            },
        },
        "selected_xgb": selected_xgb,
    }

    composition = None
    if phase_b["preliminary_pass"] and phase_c["preliminary_pass"]:
        print("=== B and C passed independently; testing gated union ===", flush=True)
        frames, y, users, _ = common.load_frames(common.DATA_DIR)
        columns = list(phase_b["selected_representation"]["columns"])
        if "decay_rate_x_log_activity" not in columns:
            columns.append("decay_rate_x_log_activity")
        model, scores, metrics, gain = common.fit_lgb(frames, y, users, columns, 0)
        composition = common.fit_record(
            "B_C_gated_union", "lgb", columns, 0, model, metrics, gain
        )
        composition["delta_vs_B0"] = float(
            metrics["primary"] - common.LGB_REFERENCE_VALID
        )
        lgb_options.append(composition)
        del model, scores
        gc.collect()
        results["composition_attempted"] = True
        results["composition"] = composition
    else:
        results["composition_attempted"] = False
        results["composition"] = None
        results["composition_skip_reason"] = (
            "Phase C failed its independent +0.0003 gate"
            if not phase_c["preliminary_pass"]
            else "Phase B failed its independent +0.0003 gate"
        )
        print(f"composition skipped: {results['composition_skip_reason']}", flush=True)

    selected_lgb = max(lgb_options, key=lambda row: row["valid"]["primary"])
    results["eligible_lgb_options"] = [
        {"name": row["name"], "valid": row["valid"], "columns": row["columns"]}
        for row in lgb_options
    ]
    results["selected_lgb"] = selected_lgb
    results["test_accessed"] = False
    common.write_json(RESULTS_PATH, results)
    print(
        f"selected XGB={selected_xgb['name']} "
        f"valid={selected_xgb['valid']['primary']:.8f}",
        flush=True,
    )
    print(
        f"selected LGB={selected_lgb['name']} "
        f"valid={selected_lgb['valid']['primary']:.8f}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
