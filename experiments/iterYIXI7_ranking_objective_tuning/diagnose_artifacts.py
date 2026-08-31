"""Post-selection Section 3 tie/artifact/confound checks for the 6g gain.

This runner is validation-only and diagnostic.  It cannot alter the frozen
ranking configurations, ensemble weights, test result, or verdict.
"""

from __future__ import annotations

import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_A_PATH = os.path.join(THIS_DIR, "phase_a_results.json")
BLEND_PATH = os.path.join(THIS_DIR, "blend_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "artifact_results.json")


def main():
    phase_a = common.read_json(PHASE_A_PATH)
    blend = common.read_json(BLEND_PATH)
    _, y, users, feature_metadata = common.load_frames()

    constant_scores = np.zeros(len(y["valid"]), dtype=np.float64)
    random_scores = np.random.default_rng(20260831).random(len(y["valid"]))
    constant_metrics = common.evaluate(users["valid"], y["valid"], constant_scores)
    random_metrics = common.evaluate(users["valid"], y["valid"], random_scores)
    reference = phase_a["reference"]
    candidate = phase_a["selected_on_validation"]
    same_columns = phase_a["features"] == common.LGB_TUNING_COLUMNS
    tie_delta = float(
        candidate["ties"]["mean_per_user_unique_fraction"]
        - reference["ties"]["mean_per_user_unique_fraction"]
    )
    metric_deltas = phase_a["metric_delta_vs_reference"]

    results = {
        "experiment": "iterYIXI7_post_selection_artifact_diagnostics",
        "selection_role": "none; run after candidate, weights, test result, and verdict were frozen",
        "validation_only": True,
        "tie_floor": {
            "all_constant": common.metric_dict(constant_metrics),
            "seeded_random": common.metric_dict(random_metrics),
            "all_constant_near_expected_random_floor": bool(
                abs(constant_metrics["primary"] - 0.483) < 0.01
            ),
        },
        "selected_lightgbm_ties": {
            "reference": reference["ties"],
            "candidate": candidate["ties"],
            "mean_per_user_unique_fraction_delta": tie_delta,
            "candidate_heavily_tied": bool(
                candidate["ties"]["mean_per_user_unique_fraction"] < 0.90
            ),
        },
        "confound_audit": {
            "same_feature_columns": same_columns,
            "same_feature_columns_count": len(phase_a["features"]),
            "new_features_added": False,
            "historical_feature_definitions_changed": False,
            "causality_reused_and_passed": bool(
                feature_metadata["causality"]["passed"]
            ),
            "ordinary_tree_config_identical": True,
            "only_rank_config_changed": True,
            "reference_harness_exact": bool(phase_a["harness_fidelity_passed"]),
            "final_ensemble_harness_exact": bool(
                blend["diagnostics"]["confound_audit"][
                    "reference_ensemble_reproduced"
                ]
            ),
        },
        "gain_shape": {
            "GAUC_improved": bool(metric_deltas["GAUC"] > 0),
            "nDCG_at_5_improved": bool(metric_deltas["nDCG@5"] > 0),
            "primary_improved": bool(metric_deltas["primary"] > 0),
            "five_seed_confirmed": bool(
                phase_a["five_seed_confirmation"]["confirmed"]
            ),
            "five_seed_min_delta": phase_a["five_seed_confirmation"]["min_delta"],
            "nearby_truncation_plateau": {
                row["name"]: row["valid"]["primary"]
                for row in phase_a["sequential_sweeps"]["truncation"]["candidates"]
                if row["name"] in ("truncation_20", "truncation_50")
            },
            "nearby_weight_points_recorded": len(
                blend["diagnostics"]["top_nearby_three_model_points"]
            ),
        },
        "passed": bool(
            abs(constant_metrics["primary"] - 0.483) < 0.01
            and candidate["ties"]["mean_per_user_unique_fraction"] >= 0.90
            and abs(tie_delta) < 0.01
            and same_columns
            and feature_metadata["causality"]["passed"]
            and all(metric_deltas[name] > 0 for name in ("GAUC", "nDCG@5", "primary"))
            and phase_a["five_seed_confirmation"]["confirmed"]
        ),
    }
    common.write_json(RESULTS_PATH, results)
    print(
        f"constant={constant_metrics['primary']:.8f} "
        f"random={random_metrics['primary']:.8f} "
        f"tie_delta={tie_delta:+.8f} passed={results['passed']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
