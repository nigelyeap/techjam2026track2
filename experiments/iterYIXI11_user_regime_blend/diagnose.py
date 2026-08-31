"""Post-selection validation-only diagnostics for the rejected adaptive blend."""

from __future__ import annotations

import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_B_PATH = os.path.join(THIS_DIR, "phase_b_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "diagnostic_results.json")


def main():
    phase_b = common.read_json(PHASE_B_PATH)
    frozen = common.load_predictions()
    users = frozen["users"]
    labels = frozen["labels"]
    row_regimes = frozen["regimes"]
    components = common.percentile_components(
        {name: frozen[name] for name in common.MODELS}, users
    )
    reference = common.combined(components, common.GLOBAL_WEIGHTS)
    adaptive = common.adaptive_scores(
        components, row_regimes, phase_b["selected_weights_by_regime"]
    )
    reference_metrics = common.evaluate(users, labels, reference)
    adaptive_metrics = common.evaluate(users, labels, adaptive)
    expected = phase_b["selected_on_validation"]["valid"]["primary"]
    if abs(adaptive_metrics["primary"] - expected) > 1e-8:
        raise AssertionError("adaptive diagnostic reproduction drift")

    user_regime_counts = {}
    for user in np.unique(users):
        unique = np.unique(row_regimes[users == user])
        if len(unique) != 1:
            raise AssertionError("user regime varies inside validation")
        user_regime_counts[str(user)] = str(unique[0])

    plateau = {}
    for regime in common.regimes.REGIME_NAMES:
        sweep = phase_b["regime_sweeps"][regime]
        best = sweep["best"]["valid"]["primary"]
        near = [
            row for row in sweep["records"]
            if best - row["valid"]["primary"] <= common.PRELIMINARY_DELTA
        ]
        plateau[regime] = {
            "best": sweep["best"],
            "points_within_0.0003": len(near),
            "grid_points": len(sweep["records"]),
            "top_points": sorted(
                sweep["records"],
                key=lambda row: row["valid"]["primary"],
                reverse=True,
            )[:10],
        }

    selected_weights = phase_b["selected_weights_by_regime"]
    constraints_pass = all(
        abs(selected_weights[regime][model] - common.GLOBAL_WEIGHTS[model])
        <= common.LOCAL_WEIGHT_RADIUS + 1e-12
        and selected_weights[regime][model] >= common.MIN_MODEL_WEIGHT
        for regime in common.regimes.REGIME_NAMES for model in common.MODELS
    )
    if not constraints_pass:
        raise AssertionError("selected weight violates local constraint")

    constant_metrics = common.evaluate(
        users, labels, np.zeros(len(labels), dtype=np.float64)
    )
    random_metrics = common.evaluate(
        users, labels, np.random.default_rng(0).uniform(size=len(labels))
    )
    results = {
        "experiment": "iterYIXI11_rejected_adaptive_blend_diagnostics",
        "selection_role": "none; validation selection already frozen",
        "reproduction": {
            "reference": common.metric_dict(reference_metrics),
            "adaptive": common.metric_dict(adaptive_metrics),
            "delta": float(adaptive_metrics["primary"] - reference_metrics["primary"]),
        },
        "tie_checks": {
            "reference": common.unique_stats(reference, users),
            "adaptive": common.unique_stats(adaptive, users),
        },
        "constant_score_valid": common.metric_dict(constant_metrics),
        "random_score_seed0_valid": common.metric_dict(random_metrics),
        "regime_checks": {
            "one_regime_per_user": True,
            "users_checked": len(user_regime_counts),
            "labels_used_for_definition": False,
        },
        "weight_constraint_check": {
            "passed": constraints_pass,
            "global_center": common.GLOBAL_WEIGHTS,
            "radius": common.LOCAL_WEIGHT_RADIUS,
            "step": common.LOCAL_WEIGHT_STEP,
            "minimum_weight": common.MIN_MODEL_WEIGHT,
        },
        "regime_weight_plateaus": plateau,
        "test_access": "none",
        "verdict": "REJECT: full validation delta is below +0.0003",
    }
    common.write_json(RESULTS_PATH, results)
    print(
        f"reference={reference_metrics['primary']:.8f} "
        f"adaptive={adaptive_metrics['primary']:.8f} "
        f"ties={common.unique_stats(adaptive, users)['mean_per_user_unique_fraction']:.6f}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
