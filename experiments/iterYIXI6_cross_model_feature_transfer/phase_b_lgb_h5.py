"""6f Phase B: transfer YIXI4's exact 5-day user pair to current LightGBM."""

from __future__ import annotations

import gc
import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "phase_b_results.json")


def main() -> None:
    frames, y, users, metadata = common.load_frames(common.DATA_DIR)
    results = {
        "experiment": "iterYIXI6_phase_b_lgb_h5",
        "question": "Does YIXI4's exact 5-day user pair transfer to current iter63 LightGBM?",
        "environment": common.environment(),
        "fixed_lgb_config": {
            "objective": "lambdarank",
            "linear_tree": True,
            "num_leaves": 2,
            "learning_rate": 0.10,
            "n_estimators": 500,
            "min_child_samples": 200,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 30,
        },
        "feature_metadata": metadata,
        "selection_policy": {
            "selector": "official standalone validation primary only",
            "preliminary_delta": common.PRELIMINARY_DELTA,
            "promotion_delta": common.PROMOTION_DELTA,
            "test_access": "none in this phase",
        },
        "candidates": [],
    }

    print("=== B0: current iter63 LightGBM with 2.5-day pair ===", flush=True)
    ref_model, ref_scores, ref_metrics, ref_gain = common.fit_lgb(
        frames, y, users, common.LGB_B0_COLUMNS, 0
    )
    if abs(ref_metrics["primary"] - common.LGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("Phase B LightGBM harness drift")
    reference = common.fit_record(
        "B0_current_h2.5", "lgb", common.LGB_B0_COLUMNS, 0,
        ref_model, ref_metrics, ref_gain
    )
    results["reference"] = reference
    print(f"  valid={ref_metrics['primary']:.8f}", flush=True)

    specs = (
        ("B1_replace_with_h5", common.LGB_B1_COLUMNS),
        ("B2_h2.5_plus_h5", common.LGB_B2_COLUMNS),
    )
    candidate_models = {}
    candidate_scores = {}
    for name, columns in specs:
        print(f"\n=== {name} ===", flush=True)
        model, scores, metrics, gain = common.fit_lgb(frames, y, users, columns, 0)
        record = common.fit_record(name, "lgb", columns, 0, model, metrics, gain)
        record["delta_vs_B0"] = float(metrics["primary"] - ref_metrics["primary"])
        results["candidates"].append(record)
        candidate_models[name] = model
        candidate_scores[name] = scores
        print(
            f"  valid={metrics['primary']:.8f} delta={record['delta_vs_B0']:+.8f}",
            flush=True,
        )

    winner = max(results["candidates"], key=lambda row: row["valid"]["primary"])
    preliminary = winner["delta_vs_B0"] >= common.PRELIMINARY_DELTA
    selected = winner if preliminary else reference
    results["winner"] = winner
    results["preliminary_pass"] = preliminary
    results["selected_representation"] = selected

    if winner["delta_vs_B0"] >= common.PROMOTION_DELTA:
        print("\n=== paired five-seed Phase B confirmation ===", flush=True)
        confirmation = common.confirm_candidate(
            "lgb",
            frames,
            y,
            users,
            common.LGB_B0_COLUMNS,
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
    results["diagnostics"] = {
        "selected_ties": common.tie_stats(selected_scores, users["valid"]),
        "exact_change": {
            "B1": "replace exact 2.5-day rate/activity with exact 5-day rate/activity",
            "B2": "retain exact 2.5-day pair and append exact 5-day pair",
        },
    }
    results["standalone_verdict"] = (
        "PROMOTE" if confirmation.get("confirmed") else "PRELIMINARY_ONLY" if preliminary else "REJECT"
    )
    results["test_accessed"] = False
    common.write_json(RESULTS_PATH, results)
    print(
        f"\nPhase B selected={selected['name']} valid={selected['valid']['primary']:.8f} "
        f"verdict={results['standalone_verdict']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)
    del ref_model, ref_scores, candidate_models, candidate_scores
    gc.collect()


if __name__ == "__main__":
    main()
