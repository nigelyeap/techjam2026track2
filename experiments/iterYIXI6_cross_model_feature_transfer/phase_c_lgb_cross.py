"""6f Phase C: retest YIXI4's exact strongest cross on current LightGBM."""

from __future__ import annotations

import gc
import os

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "phase_c_results.json")


def main() -> None:
    frames, y, users, metadata = common.load_frames(common.DATA_DIR)
    results = {
        "experiment": "iterYIXI6_phase_c_lgb_cross",
        "question": "Does the exact 2.5-day rate x log-activity cross help current LightGBM?",
        "environment": common.environment(),
        "feature_metadata": metadata,
        "fixed_lgb_config": {
            "linear_tree": True,
            "num_leaves": 2,
            "learning_rate": 0.10,
            "n_estimators": 500,
            "decay_tab_feature": "decay_tab_rate_3",
        },
        "selection_policy": {
            "selector": "official standalone validation primary only",
            "preliminary_delta": common.PRELIMINARY_DELTA,
            "promotion_delta": common.PROMOTION_DELTA,
            "test_access": "none in this phase",
        },
    }

    print("=== C0: current iter63 LightGBM ===", flush=True)
    ref_model, ref_scores, ref_metrics, ref_gain = common.fit_lgb(
        frames, y, users, common.LGB_B0_COLUMNS, 0
    )
    if abs(ref_metrics["primary"] - common.LGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("Phase C LightGBM harness drift")
    reference = common.fit_record(
        "C0_current_lgb", "lgb", common.LGB_B0_COLUMNS, 0,
        ref_model, ref_metrics, ref_gain
    )
    results["reference"] = reference
    print(f"  valid={ref_metrics['primary']:.8f}", flush=True)

    print("\n=== C1: add exact decay_rate_2.5 * log1p(decay_act_2.5) ===", flush=True)
    model, scores, metrics, gain = common.fit_lgb(
        frames, y, users, common.LGB_C1_COLUMNS, 0
    )
    candidate = common.fit_record(
        "C1_add_exact_rate_log_activity_cross",
        "lgb",
        common.LGB_C1_COLUMNS,
        0,
        model,
        metrics,
        gain,
    )
    candidate["delta_vs_C0"] = float(metrics["primary"] - ref_metrics["primary"])
    results["candidate"] = candidate
    preliminary = candidate["delta_vs_C0"] >= common.PRELIMINARY_DELTA
    results["preliminary_pass"] = preliminary
    results["selected_representation"] = candidate if preliminary else reference
    print(
        f"  valid={metrics['primary']:.8f} delta={candidate['delta_vs_C0']:+.8f}",
        flush=True,
    )

    if candidate["delta_vs_C0"] >= common.PROMOTION_DELTA:
        print("\n=== paired five-seed Phase C confirmation ===", flush=True)
        confirmation = common.confirm_candidate(
            "lgb",
            frames,
            y,
            users,
            common.LGB_B0_COLUMNS,
            common.LGB_C1_COLUMNS,
            reference["valid"],
            candidate["valid"],
        )
    else:
        confirmation = {
            "performed": False,
            "reason": "seed-0 gain below 0.001 promotion threshold",
            "confirmed": False,
            "rows": [],
        }
    results["five_seed_confirmation"] = confirmation
    results["diagnostics"] = {
        "candidate_ties": common.tie_stats(scores, users["valid"]),
        "cross_definition": "decay_rate_2.5 * log1p(decay_act_2.5)",
        "cross_redesigned": False,
        "cross_gain_fraction": gain.get("decay_rate_x_log_activity", 0.0),
    }
    results["standalone_verdict"] = (
        "PROMOTE" if confirmation.get("confirmed") else "PRELIMINARY_ONLY" if preliminary else "REJECT"
    )
    results["test_accessed"] = False
    common.write_json(RESULTS_PATH, results)
    print(
        f"\nPhase C selected={results['selected_representation']['name']} "
        f"valid={results['selected_representation']['valid']['primary']:.8f} "
        f"verdict={results['standalone_verdict']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)
    del ref_model, ref_scores, model, scores
    gc.collect()


if __name__ == "__main__":
    main()
