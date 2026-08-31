"""Phase A: predeclared common monotonic transforms on frozen predictions."""

from __future__ import annotations

import os

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "phase_a_results.json")


def main():
    frozen = common.load_frozen()
    users, labels = frozen["users"], frozen["labels"]
    raw = {"fm": frozen["fm"], "lgb": frozen["yixi5_lgb"], "xgb": frozen["xgb"]}
    states = common.build_rank_states(raw, users)
    results = {
        "experiment": "iterYIXI8_phase_a_common_rank_transforms",
        "selection_policy": {
            "base_predictions": "immutable validation arrays from freeze_predictions.py",
            "base_model_retraining": False,
            "selector": "official validation primary only",
            "weights": common.YIXI5_WEIGHTS,
            "preliminary_delta": common.PRELIMINARY_DELTA,
            "test_access": "none",
            "model_specific_transforms": False,
        },
        "predeclared_transforms": [
            {"name": name, **spec} for name, spec in common.TRANSFORMS
        ],
        "records": [],
    }

    for name, spec in common.TRANSFORMS:
        transform_map = common.common_transform_map(name)
        components = common.transformed_components(states, transform_map)
        scores, metrics = common.score_components(
            components, common.YIXI5_WEIGHTS, users, labels
        )
        delta = float(metrics["primary"] - common.YIXI5_VALID)
        row = {
            "name": name,
            "spec": spec,
            "transform_by_model": transform_map,
            "weights": common.YIXI5_WEIGHTS,
            "valid": common.metric_dict(metrics),
            "delta_vs_percentile_reference": delta,
            "clears_preliminary_threshold": bool(delta >= common.PRELIMINARY_DELTA),
            "blend_unique_scores": common.unique_stats(scores, users),
            "component_unique_scores": {
                model: common.unique_stats(components[model], users)
                for model in common.MODELS
            },
        }
        results["records"].append(row)
        print(
            f"{name:20s} primary={metrics['primary']:.8f} "
            f"GAUC={metrics['GAUC']:.8f} nDCG@5={metrics['nDCG@5']:.8f} "
            f"delta={delta:+.8f}",
            flush=True,
        )

    reference = next(row for row in results["records"] if row["name"] == "percentile")
    if abs(reference["valid"]["primary"] - common.YIXI5_VALID) > 1e-8:
        raise AssertionError("frozen percentile reference drift")
    promising = [
        row["name"]
        for row in results["records"]
        if row["clears_preliminary_threshold"]
    ]
    best = max(results["records"], key=lambda row: row["valid"]["primary"])
    results["reference"] = reference
    results["harness_fidelity_passed"] = True
    results["promising_transforms"] = promising
    results["best_fixed_weight"] = best
    results["best_fixed_weight"]["selected_on_validation"] = True
    common.write_json(RESULTS_PATH, results)
    print(f"promising={promising}", flush=True)
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
