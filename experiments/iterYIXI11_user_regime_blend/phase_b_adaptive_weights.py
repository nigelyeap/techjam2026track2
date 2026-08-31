"""Constrained validation-only local weight sweep per eligible user regime."""

from __future__ import annotations

import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_A_PATH = os.path.join(THIS_DIR, "phase_a_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "phase_b_results.json")


def main():
    phase_a = common.read_json(PHASE_A_PATH)
    if not phase_a["adaptive_weight_sweep_eligible"]:
        raise RuntimeError("Phase A model-order gate did not open")
    frozen = common.load_predictions()
    users = frozen["users"]
    labels = frozen["labels"]
    row_regimes = frozen["regimes"]
    raw = {name: frozen[name] for name in common.MODELS}
    components = common.percentile_components(raw, users)
    reference_scores, reference_metrics = common.score_components(
        components, common.GLOBAL_WEIGHTS, users, labels
    )
    if abs(reference_metrics["primary"] - common.ENSEMBLE_REFERENCE_VALID) > 1e-8:
        raise AssertionError("global fixed reference drift")

    grid = common.local_weight_grid()
    print(f"local grid points per regime={len(grid)}", flush=True)
    sweeps = {}
    selected_weights = {}
    for regime in common.regimes.REGIME_NAMES:
        mask = row_regimes == regime
        regime_components = {
            name: values[mask] for name, values in components.items()
        }
        records = []
        for weights in grid:
            _, metrics = common.score_components(
                regime_components, weights, users[mask], labels[mask]
            )
            records.append(
                {"weights": weights, "valid": common.metric_dict(metrics)}
            )
        best = max(records, key=lambda row: row["valid"]["primary"])
        _, global_regime_metrics = common.score_components(
            regime_components, common.GLOBAL_WEIGHTS, users[mask], labels[mask]
        )
        best["delta_vs_global_weights"] = float(
            best["valid"]["primary"] - global_regime_metrics["primary"]
        )
        selected_weights[regime] = best["weights"]
        nearby = sorted(
            records, key=lambda row: row["valid"]["primary"], reverse=True
        )[:15]
        sweeps[regime] = {
            "global_weights_valid": common.metric_dict(global_regime_metrics),
            "best": best,
            "nearby": nearby,
            "records": records,
        }
        print(
            f"{regime}: weights={best['weights']} "
            f"valid={best['valid']['primary']:.8f} "
            f"regime_delta={best['delta_vs_global_weights']:+.8f}", flush=True
        )

    adaptive = common.adaptive_scores(components, row_regimes, selected_weights)
    adaptive_metrics = common.evaluate(users, labels, adaptive)
    delta = float(adaptive_metrics["primary"] - reference_metrics["primary"])
    results = {
        "experiment": "iterYIXI11_phase_b_constrained_adaptive_weights",
        "selection_policy": {
            "selector": "official validation primary independently within each disjoint user regime",
            "grid": {
                "global_center": common.GLOBAL_WEIGHTS,
                "step": common.LOCAL_WEIGHT_STEP,
                "per_coordinate_radius": common.LOCAL_WEIGHT_RADIUS,
                "minimum_model_weight": common.MIN_MODEL_WEIGHT,
                "points_per_regime": len(grid),
            },
            "regime_definition": "frozen Phase A training-only history policy",
            "test_access": "none",
        },
        "reference": {
            "weights": common.GLOBAL_WEIGHTS,
            "valid": common.metric_dict(reference_metrics),
            "ties": common.unique_stats(reference_scores, users),
        },
        "regime_sweeps": sweeps,
        "selected_weights_by_regime": selected_weights,
        "selected_on_validation": {
            "valid": common.metric_dict(adaptive_metrics),
            "delta_vs_fixed_reference": delta,
            "ties": common.unique_stats(adaptive, users),
        },
        "clears_preliminary": bool(delta >= common.PRELIMINARY_DELTA),
        "eligible_for_confirmation": bool(delta >= common.PROMOTION_DELTA),
    }
    common.write_json(RESULTS_PATH, results)
    print(
        f"adaptive valid={adaptive_metrics['primary']:.8f} delta={delta:+.8f} "
        f"confirmation_eligible={results['eligible_for_confirmation']}", flush=True
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
