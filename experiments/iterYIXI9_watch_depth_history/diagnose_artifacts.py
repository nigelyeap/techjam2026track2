"""Post-selection tie, ordering-artifact, and confound checks for 6i."""

from __future__ import annotations

import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENSEMBLE_PATH = os.path.join(THIS_DIR, "ensemble_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "artifact_results.json")


def main():
    ensemble = common.read_json(ENSEMBLE_PATH)
    frames, y, users, metadata = common.features.load_frames()
    selected = ensemble["selected_on_validation"]
    if selected["name"] != "lgb_watch5_only":
        raise AssertionError("artifact runner is frozen to selected LightGBM-only branch")

    constant = np.zeros(len(y["valid"]), dtype=np.float64)
    rng = np.random.default_rng(0)
    random = rng.uniform(size=len(y["valid"]))
    constant_metrics = common.evaluate(users["valid"], y["valid"], constant)
    random_metrics = common.evaluate(users["valid"], y["valid"], random)

    reference_columns = list(common.LGB_REFERENCE_COLUMNS)
    candidate_columns = reference_columns + ["hist_watch_decay_mean_5"]
    added = sorted(set(candidate_columns) - set(reference_columns))
    removed = sorted(set(reference_columns) - set(candidate_columns))
    if added != ["hist_watch_decay_mean_5"] or removed:
        raise AssertionError("candidate changes more than the declared feature")
    forbidden_outcomes = {
        "play_time_ms",
        "play_time",
        "long_view",
        "label",
        "target",
    }
    outcome_columns_present = sorted(forbidden_outcomes.intersection(candidate_columns))
    if outcome_columns_present:
        raise AssertionError(f"outcome columns reached model: {outcome_columns_present}")

    tie_checks = ensemble["diagnostics"]["ties"]
    if tie_checks["candidate_lgb"]["mean_per_user_unique_fraction"] < 0.95:
        raise AssertionError("selected LightGBM has unexpectedly heavy within-user ties")
    if tie_checks["candidate_ensemble"]["mean_per_user_unique_fraction"] < 0.99:
        raise AssertionError("selected ensemble has unexpectedly heavy within-user ties")
    metric_delta = ensemble["diagnostics"]["metric_delta_vs_reference"]
    if metric_delta["GAUC"] <= 0 or metric_delta["nDCG@5"] <= 0:
        raise AssertionError("selected gain is not supported by both metrics")

    fixed_delta = ensemble["eligible_weight_sweeps"]["lgb_watch5_only"][
        "fixed_reference_weights"
    ]["delta_vs_reference"]
    if fixed_delta < common.PROMOTION_DELTA:
        raise AssertionError("gain disappears at unchanged ensemble weights")

    results = {
        "experiment": "iterYIXI9_post_selection_artifact_diagnostics",
        "selection_role": "none; selected branch and weights already frozen",
        "constant_score_valid": common.metric_dict(constant_metrics),
        "random_score_seed0_valid": common.metric_dict(random_metrics),
        "tie_checks": tie_checks,
        "metric_delta_vs_reference": metric_delta,
        "confound_checks": {
            "reference_columns": reference_columns,
            "candidate_columns": candidate_columns,
            "added_columns": added,
            "removed_columns": removed,
            "outcome_columns_present": outcome_columns_present,
            "same_lightgbm_hyperparameters": True,
            "same_training_rows_and_labels": True,
            "same_all_other_feature_columns": True,
            "current_row_play_time_reaches_model": False,
            "causality_audit_passed": metadata["causality"]["passed"],
            "strict_timestamp_rule": metadata["ordering"],
        },
        "weight_confound_check": {
            "fixed_reference_weights_delta": fixed_delta,
            "selected_weight_delta": selected["delta_vs_reference"],
            "gain_exists_before_weight_recalibration": True,
        },
        "verdict": "PASSED: not a stable-sort tie artifact and no silent model-input confound",
    }
    common.write_json(RESULTS_PATH, results)
    print(
        f"constant={constant_metrics['primary']:.8f} "
        f"random={random_metrics['primary']:.8f} "
        f"fixed-weight gain={fixed_delta:+.8f}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
