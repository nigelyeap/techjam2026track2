"""Validation-only tie diagnostics for all predeclared rank transforms."""

from __future__ import annotations

import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_A_PATH = os.path.join(THIS_DIR, "phase_a_results.json")
PHASE_B_PATH = os.path.join(THIS_DIR, "phase_b_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "diagnostic_results.json")


def main():
    phase_a = common.read_json(PHASE_A_PATH)
    phase_b = common.read_json(PHASE_B_PATH)
    frozen = common.load_frozen()
    users, labels = frozen["users"], frozen["labels"]
    constant = np.zeros(len(labels), dtype=np.float64)
    random = np.random.default_rng(20260831).random(len(labels))
    constant_metrics = common.evaluate(users, labels, constant)
    random_metrics = common.evaluate(users, labels, random)

    selected = phase_b["selected_on_validation"]
    raw = {
        "fm": frozen["fm"],
        "lgb": frozen["yixi5_lgb"],
        "xgb": frozen["xgb"],
    }
    states = common.build_rank_states(raw, users)
    components = common.transformed_components(
        states, selected["transform_by_model"]
    )
    scores, metrics = common.score_components(
        components, selected["weights"], users, labels
    )
    selected_stats = common.unique_stats(scores, users)
    component_stats = {
        model: common.unique_stats(components[model], users)
        for model in common.MODELS
    }
    phase_a_fractions = {
        row["name"]: row["blend_unique_scores"][
            "mean_per_user_unique_fraction"
        ]
        for row in phase_a["records"]
    }
    results = {
        "experiment": "iterYIXI8_validation_tie_diagnostics",
        "selection_role": "none; post-selection diagnostic",
        "validation_only": True,
        "constant": common.metric_dict(constant_metrics),
        "seeded_random": common.metric_dict(random_metrics),
        "selected": {
            "name": selected["name"],
            "valid": common.metric_dict(metrics),
            "blend_unique_scores": selected_stats,
            "component_unique_scores": component_stats,
        },
        "all_transform_blend_unique_fractions": phase_a_fractions,
        "checks": {
            "constant_near_random_floor": bool(
                abs(constant_metrics["primary"] - 0.483) < 0.01
            ),
            "selected_not_heavily_tied": bool(
                selected_stats["mean_per_user_unique_fraction"] > 0.95
            ),
            "all_transforms_not_heavily_tied": bool(
                min(phase_a_fractions.values()) > 0.95
            ),
            "selected_metric_reproduced": bool(
                abs(metrics["primary"] - selected["valid"]["primary"]) < 1e-8
            ),
        },
    }
    results["passed"] = bool(all(results["checks"].values()))
    common.write_json(RESULTS_PATH, results)
    print(
        f"constant={constant_metrics['primary']:.8f} "
        f"random={random_metrics['primary']:.8f} "
        f"selected_unique_fraction={selected_stats['mean_per_user_unique_fraction']:.8f} "
        f"passed={results['passed']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
