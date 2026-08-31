"""6f Phase A: transfer iter63's exact tab-rate feature to tuned XGBoost."""

from __future__ import annotations

import gc
import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "phase_a_results.json")


def main() -> None:
    frames, y, users, metadata = common.load_frames(common.DATA_DIR)
    results = {
        "experiment": "iterYIXI6_phase_a_xgb_tab_rate",
        "question": "Does iter63 decay_tab_rate_3 transfer to tuned YIXI5 XGBoost?",
        "environment": common.environment(),
        "fixed_xgb_config": common.selected_xgb_config(),
        "feature_metadata": metadata,
        "selection_policy": {
            "selector": "official standalone validation primary only",
            "preliminary_delta": common.PRELIMINARY_DELTA,
            "promotion_delta": common.PROMOTION_DELTA,
            "test_access": "none in this phase",
        },
        "candidates": [],
    }

    print("=== A0: current tuned XGBoost with decay_tab_3 ===", flush=True)
    ref_model, ref_scores, ref_metrics, ref_gain = common.fit_xgb(
        frames, y, users, common.XGB_A0_COLUMNS, 0
    )
    if abs(ref_metrics["primary"] - common.XGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("Phase A XGBoost harness drift")
    reference = common.fit_record(
        "A0_current_tab_count", "xgb", common.XGB_A0_COLUMNS, 0,
        ref_model, ref_metrics, ref_gain
    )
    results["reference"] = reference
    print(f"  valid={ref_metrics['primary']:.8f}", flush=True)

    specs = (
        ("A1_replace_with_tab_rate", common.XGB_A1_COLUMNS),
        ("A2_count_plus_tab_rate", common.XGB_A2_COLUMNS),
    )
    candidate_models = {}
    candidate_scores = {}
    for name, columns in specs:
        print(f"\n=== {name} ===", flush=True)
        model, scores, metrics, gain = common.fit_xgb(frames, y, users, columns, 0)
        record = common.fit_record(name, "xgb", columns, 0, model, metrics, gain)
        record["delta_vs_A0"] = float(metrics["primary"] - ref_metrics["primary"])
        results["candidates"].append(record)
        candidate_models[name] = model
        candidate_scores[name] = scores
        print(
            f"  valid={metrics['primary']:.8f} delta={record['delta_vs_A0']:+.8f}",
            flush=True,
        )

    winner = max(results["candidates"], key=lambda row: row["valid"]["primary"])
    preliminary = winner["delta_vs_A0"] >= common.PRELIMINARY_DELTA
    selected = winner if preliminary else reference
    results["winner"] = winner
    results["preliminary_pass"] = preliminary
    results["selected_representation"] = selected

    if winner["delta_vs_A0"] >= common.PROMOTION_DELTA:
        print("\n=== paired five-seed Phase A confirmation ===", flush=True)
        confirmation = common.confirm_candidate(
            "xgb",
            frames,
            y,
            users,
            common.XGB_A0_COLUMNS,
            winner["columns"],
            reference["valid"],
            winner["valid"],
        )
    else:
        confirmation = {
            "performed": False,
            "reason": "seed-0 gain below 0.001 promotion threshold",
            "confirmed": False,
            "rows": [],
        }
    results["five_seed_confirmation"] = confirmation

    selected_scores = (
        candidate_scores[selected["name"]]
        if selected["name"] in candidate_scores
        else ref_scores
    )
    rng = np.random.default_rng(0)
    results["diagnostics"] = {
        "selected_ties": common.tie_stats(selected_scores, users["valid"]),
        "constant_valid": common.metric_dict(
            common.evaluate(users["valid"], y["valid"], np.zeros(len(y["valid"])))
        ),
        "random_valid": common.metric_dict(
            common.evaluate(
                users["valid"], y["valid"], rng.uniform(size=len(y["valid"]))
            )
        ),
        "exact_change": {
            "A1": "replace decay_tab_3 with exact iter63 decay_tab_rate_3",
            "A2": "retain decay_tab_3 and add exact iter63 decay_tab_rate_3",
        },
    }
    results["standalone_verdict"] = (
        "PROMOTE" if confirmation.get("confirmed") else "PRELIMINARY_ONLY" if preliminary else "REJECT"
    )
    results["test_accessed"] = False
    common.write_json(RESULTS_PATH, results)
    print(
        f"\nPhase A selected={selected['name']} valid={selected['valid']['primary']:.8f} "
        f"verdict={results['standalone_verdict']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)

    del ref_model, ref_scores, candidate_models, candidate_scores
    gc.collect()


if __name__ == "__main__":
    main()
