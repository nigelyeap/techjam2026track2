"""Validation-only composition and final evaluation for YIXI Section 6i.

Only independently confirmed watch-depth branches are eligible.  The runner
reproduces the exact post-6h ensemble, sweeps within-user-percentile weights on
validation, freezes the selected branch/weights and verdict, and only then
predicts or evaluates test.
"""

from __future__ import annotations

import gc
import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LGB_RESULTS_PATH = os.path.join(THIS_DIR, "phase_lgb_results.json")
XGB_RESULTS_PATH = os.path.join(THIS_DIR, "phase_xgb_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "ensemble_results.json")
YIXI5_BLEND_PATH = os.path.join(
    common.REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "blend.py"
)


def normalized(blend, fm, lgb, xgb, users):
    return blend.normalize_components(
        fm, lgb, xgb, users, "within_user_percentile"
    )


def score(blend, components, weights, users, labels):
    scores = blend.combined_scores(components, weights)
    return scores, common.evaluate(users, labels, scores)


def metric_delta(candidate, reference):
    return {
        name: float(candidate[name] - reference[name])
        for name in ("GAUC", "nDCG@5", "primary")
    }


def top_nearby(records, limit=12):
    return sorted(
        (
            row
            for row in records
            if all(row["weights"][name] > 0 for name in ("fm", "lgb", "xgb"))
        ),
        key=lambda row: row["valid"]["primary"],
        reverse=True,
    )[:limit]


def candidate_specs(lgb_confirmed, xgb_confirmed):
    lgb_feature = "hist_watch_decay_mean_5"
    xgb_features = (
        "hist_watch_decay_mean_2.5",
        "hist_watch_decay_mean_5",
    )
    if lgb_feature not in lgb_confirmed:
        raise AssertionError("expected confirmed LightGBM 5-day branch is absent")
    if not set(xgb_features).issubset(xgb_confirmed):
        raise AssertionError("expected confirmed XGBoost decay branches are absent")
    return {
        "lgb_watch5_only": {
            "lgb_feature": lgb_feature,
            "xgb_feature": None,
            "xgb_base": "current_ensemble_A0",
        },
        "xgb_watch2.5_only": {
            "lgb_feature": None,
            "xgb_feature": xgb_features[0],
            "xgb_base": "strongest_standalone_A1",
        },
        "xgb_watch5_only": {
            "lgb_feature": None,
            "xgb_feature": xgb_features[1],
            "xgb_base": "strongest_standalone_A1",
        },
        "both_lgb5_xgb2.5": {
            "lgb_feature": lgb_feature,
            "xgb_feature": xgb_features[0],
            "xgb_base": "strongest_standalone_A1",
        },
        "both_lgb5_xgb5": {
            "lgb_feature": lgb_feature,
            "xgb_feature": xgb_features[1],
            "xgb_base": "strongest_standalone_A1",
        },
    }


def main():
    lgb_phase = common.read_json(LGB_RESULTS_PATH)
    xgb_phase = common.read_json(XGB_RESULTS_PATH)
    specs = candidate_specs(
        set(lgb_phase["confirmed_features"]),
        set(xgb_phase["confirmed_features"]),
    )
    frames, y, users, metadata = common.features.load_frames()
    blend = common.load_module(YIXI5_BLEND_PATH, "yixi5_blend_for_yixi9")
    yixi8 = common.load_module(common.YIXI8_COMMON_PATH, "yixi8_common_for_yixi9")
    frozen = yixi8.load_frozen()
    if not np.array_equal(frozen["users"], np.asarray(users["valid"])):
        raise AssertionError("frozen/native validation users differ")
    if not np.array_equal(frozen["labels"], y["valid"]):
        raise AssertionError("frozen/native validation labels differ")
    fm_scores = frozen["fm"]
    fm_metrics = common.evaluate(users["valid"], y["valid"], fm_scores)
    if abs(fm_metrics["primary"] - common.FM_REFERENCE_VALID) > 1e-8:
        raise AssertionError("frozen FM validation drift")

    print("=== exact current ensemble component fidelity ===", flush=True)
    lgb_ref_model, lgb_ref_scores, lgb_ref_metrics, _ = common.fit_lgb(
        frames, y, users, common.LGB_REFERENCE_COLUMNS, 0
    )
    xgb_a0_model, xgb_a0_scores, xgb_a0_metrics, _ = common.fit_xgb(
        frames, y, users, common.CURRENT_XGB_COLUMNS, 0
    )
    if abs(lgb_ref_metrics["primary"] - common.LGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("LightGBM ensemble reference drift")
    if abs(xgb_a0_metrics["primary"] - common.CURRENT_XGB_VALID) > 1e-8:
        raise AssertionError("XGBoost ensemble A0 reference drift")
    reference_components = normalized(
        blend, fm_scores, lgb_ref_scores, xgb_a0_scores, users["valid"]
    )
    reference_scores, reference_metrics = score(
        blend,
        reference_components,
        common.CURRENT_WEIGHTS,
        users["valid"],
        y["valid"],
    )
    if abs(reference_metrics["primary"] - common.ENSEMBLE_REFERENCE_VALID) > 1e-8:
        raise AssertionError("post-6h ensemble reference drift")
    print(f"reference PASSED={reference_metrics['primary']:.8f}", flush=True)

    print("=== fitting confirmed seed-0 branch components ===", flush=True)
    lgb_watch_columns = common.LGB_REFERENCE_COLUMNS + [
        "hist_watch_decay_mean_5"
    ]
    lgb_watch_model, lgb_watch_scores, lgb_watch_metrics, lgb_watch_gain = (
        common.fit_lgb(frames, y, users, lgb_watch_columns, 0)
    )
    expected_lgb = lgb_phase["selected_on_validation"]["valid"]["primary"]
    if abs(lgb_watch_metrics["primary"] - expected_lgb) > 1e-8:
        raise AssertionError("confirmed LightGBM refit drift")

    xgb_a1_model, xgb_a1_scores, xgb_a1_metrics, _ = common.fit_xgb(
        frames, y, users, common.XGB_REFERENCE_COLUMNS, 0
    )
    if abs(xgb_a1_metrics["primary"] - common.XGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("standalone XGBoost A1 reference drift")
    xgb_models = {}
    xgb_scores = {}
    xgb_metrics = {}
    xgb_gains = {}
    expected_by_feature = {
        row["feature"]: row["valid"]["primary"]
        for row in xgb_phase["candidates"]
    }
    for feature in (
        "hist_watch_decay_mean_2.5",
        "hist_watch_decay_mean_5",
    ):
        columns = common.XGB_REFERENCE_COLUMNS + [feature]
        model, scores, metrics, gains = common.fit_xgb(
            frames, y, users, columns, 0
        )
        if abs(metrics["primary"] - expected_by_feature[feature]) > 1e-8:
            raise AssertionError(f"confirmed XGBoost refit drift for {feature}")
        xgb_models[feature] = model
        xgb_scores[feature] = scores
        xgb_metrics[feature] = metrics
        xgb_gains[feature] = gains
        print(f"  {feature}={metrics['primary']:.8f}", flush=True)

    # This is diagnostic only: it separates the known A0->A1 representation
    # transfer from the newly tested historical-watch additions.
    a1_components = normalized(
        blend, fm_scores, lgb_ref_scores, xgb_a1_scores, users["valid"]
    )
    a1_best, a1_best_overall, a1_records = blend.score_weight_grid(
        a1_components,
        users["valid"],
        y["valid"],
        "within_user_percentile",
    )

    print("=== validation-only eligible ensemble sweeps ===", flush=True)
    candidate_artifacts = {}
    sweep_results = {}
    for name, spec in specs.items():
        lgb_scores = lgb_watch_scores if spec["lgb_feature"] else lgb_ref_scores
        if spec["xgb_feature"]:
            candidate_xgb_scores = xgb_scores[spec["xgb_feature"]]
        else:
            candidate_xgb_scores = xgb_a0_scores
        components = normalized(
            blend, fm_scores, lgb_scores, candidate_xgb_scores, users["valid"]
        )
        fixed_scores, fixed_metrics = score(
            blend,
            components,
            common.CURRENT_WEIGHTS,
            users["valid"],
            y["valid"],
        )
        best_three, best_overall, records = blend.score_weight_grid(
            components,
            users["valid"],
            y["valid"],
            "within_user_percentile",
        )
        delta = float(best_three["valid"]["primary"] - reference_metrics["primary"])
        best_three["delta_vs_reference"] = delta
        sweep_results[name] = {
            "spec": spec,
            "fixed_reference_weights": {
                "weights": common.CURRENT_WEIGHTS,
                "valid": common.metric_dict(fixed_metrics),
                "delta_vs_reference": float(
                    fixed_metrics["primary"] - reference_metrics["primary"]
                ),
            },
            "best_three_model": best_three,
            "best_including_ablations": best_overall,
            "top_nearby_three_model_points": top_nearby(records),
            "records": records,
        }
        candidate_artifacts[name] = {
            "components": components,
            "scores": blend.combined_scores(components, best_three["weights"]),
            "fixed_scores": fixed_scores,
        }
        print(
            f"  {name}: weights={best_three['weights']} "
            f"valid={best_three['valid']['primary']:.8f} delta={delta:+.8f}",
            flush=True,
        )

    selected_name = max(
        sweep_results,
        key=lambda name: sweep_results[name]["best_three_model"]["valid"][
            "primary"
        ],
    )
    selected = sweep_results[selected_name]["best_three_model"]
    selected_spec = specs[selected_name]
    selected_delta = float(selected["valid"]["primary"] - reference_metrics["primary"])
    selected_weights = selected["weights"]
    selected_scores = candidate_artifacts[selected_name]["scores"]
    print(
        f"selected={selected_name} valid={selected['valid']['primary']:.8f} "
        f"delta={selected_delta:+.8f}",
        flush=True,
    )

    results = {
        "experiment": "iterYIXI9_watch_depth_final_ensemble",
        "environment": common.environment(),
        "feature_metadata": metadata,
        "selection_policy": {
            "branch_eligibility": "independent standalone confirmation only",
            "selector": "final validation primary only",
            "normalization": "within-user average-tie percentile",
            "weights": "coarse 0.10 simplex then local 0.02 refinement",
            "promotion_delta": common.PROMOTION_DELTA,
            "test_access": "none until branch, weights, confirmation, diagnostics, and verdict are frozen",
        },
        "confirmed_standalone_features": {
            "lgb": lgb_phase["confirmed_features"],
            "xgb": xgb_phase["confirmed_features"],
        },
        "components_valid": {
            "fm": common.metric_dict(fm_metrics),
            "current_lgb": common.metric_dict(lgb_ref_metrics),
            "candidate_lgb_watch5": common.metric_dict(lgb_watch_metrics),
            "current_ensemble_xgb_A0": common.metric_dict(xgb_a0_metrics),
            "standalone_xgb_A1": common.metric_dict(xgb_a1_metrics),
            "candidate_xgb_watch2.5": common.metric_dict(
                xgb_metrics["hist_watch_decay_mean_2.5"]
            ),
            "candidate_xgb_watch5": common.metric_dict(
                xgb_metrics["hist_watch_decay_mean_5"]
            ),
        },
        "reference": {
            "weights": common.CURRENT_WEIGHTS,
            "normalization": "within_user_percentile",
            "valid": common.metric_dict(reference_metrics),
        },
        "a1_without_watch_confound_diagnostic": {
            "role": "diagnostic only; not a newly eligible 6i branch",
            "best_three_model": a1_best,
            "best_including_ablations": a1_best_overall,
            "delta_vs_reference": float(
                a1_best["valid"]["primary"] - reference_metrics["primary"]
            ),
            "top_nearby_three_model_points": top_nearby(a1_records),
        },
        "eligible_weight_sweeps": sweep_results,
        "selected_on_validation": {
            "name": selected_name,
            "spec": selected_spec,
            "weights": selected_weights,
            "valid": selected["valid"],
            "delta_vs_reference": selected_delta,
        },
        "diagnostics": {
            "metric_delta_vs_reference": metric_delta(
                selected["valid"], reference_metrics
            ),
            "ties": {
                "reference_lgb": common.unique_stats(
                    lgb_ref_scores, users["valid"]
                ),
                "candidate_lgb": common.unique_stats(
                    lgb_watch_scores, users["valid"]
                ),
                "reference_xgb_A0": common.unique_stats(
                    xgb_a0_scores, users["valid"]
                ),
                "reference_xgb_A1": common.unique_stats(
                    xgb_a1_scores, users["valid"]
                ),
                "candidate_xgb": common.unique_stats(
                    xgb_scores[selected_spec["xgb_feature"]], users["valid"]
                )
                if selected_spec["xgb_feature"]
                else common.unique_stats(xgb_a0_scores, users["valid"]),
                "reference_ensemble": common.unique_stats(
                    reference_scores, users["valid"]
                ),
                "candidate_ensemble": common.unique_stats(
                    selected_scores, users["valid"]
                ),
            },
            "selected_feature_gain_fraction": {
                "lgb_watch5": lgb_watch_gain.get(
                    "hist_watch_decay_mean_5", 0.0
                ),
                "xgb_watch": xgb_gains[selected_spec["xgb_feature"]].get(
                    selected_spec["xgb_feature"], 0.0
                )
                if selected_spec["xgb_feature"]
                else None,
            },
            "confound_audit": {
                "tree_hyperparameters_fixed": True,
                "all_features_tested_independently_first": True,
                "only_confirmed_features_composed": True,
                "A1_without_watch_scored_separately": True,
                "FM_predictions_unchanged": True,
                "test_used_for_selection": False,
            },
        },
    }

    confirmation_rows = []
    if selected_delta >= common.PROMOTION_DELTA:
        print("=== paired five-seed fixed-ensemble confirmation ===", flush=True)
        confirmation_rows.append(
            {
                "seed": 0,
                "reference_valid": common.metric_dict(reference_metrics),
                "candidate_valid": selected["valid"],
                "delta": selected_delta,
            }
        )
        for seed in common.SEEDS[1:]:
            ref_lgb_model, ref_lgb_seed_scores, _, _ = common.fit_lgb(
                frames, y, users, common.LGB_REFERENCE_COLUMNS, seed
            )
            if selected_spec["lgb_feature"]:
                cand_lgb_model, cand_lgb_seed_scores, _, _ = common.fit_lgb(
                    frames, y, users, lgb_watch_columns, seed
                )
            else:
                cand_lgb_model = None
                cand_lgb_seed_scores = ref_lgb_seed_scores
            candidate_xgb_seed_scores = (
                xgb_scores[selected_spec["xgb_feature"]]
                if selected_spec["xgb_feature"]
                else xgb_a0_scores
            )
            ref_components = normalized(
                blend,
                fm_scores,
                ref_lgb_seed_scores,
                xgb_a0_scores,
                users["valid"],
            )
            cand_components = normalized(
                blend,
                fm_scores,
                cand_lgb_seed_scores,
                candidate_xgb_seed_scores,
                users["valid"],
            )
            _, ref_metrics = score(
                blend,
                ref_components,
                common.CURRENT_WEIGHTS,
                users["valid"],
                y["valid"],
            )
            _, cand_metrics = score(
                blend,
                cand_components,
                selected_weights,
                users["valid"],
                y["valid"],
            )
            delta = float(cand_metrics["primary"] - ref_metrics["primary"])
            confirmation_rows.append(
                {
                    "seed": seed,
                    "reference_valid": common.metric_dict(ref_metrics),
                    "candidate_valid": common.metric_dict(cand_metrics),
                    "delta": delta,
                }
            )
            print(
                f"  seed={seed} reference={ref_metrics['primary']:.8f} "
                f"candidate={cand_metrics['primary']:.8f} delta={delta:+.8f}",
                flush=True,
            )
            del ref_lgb_model, ref_lgb_seed_scores, ref_components, cand_components
            if cand_lgb_model is not None:
                del cand_lgb_model, cand_lgb_seed_scores
            gc.collect()

    confirmation = common.confirmation_summary(confirmation_rows)
    results["five_seed_confirmation"] = confirmation
    results["verdict"] = "PROMOTE" if confirmation["confirmed"] else "REJECT"
    results["test_evaluation"] = {
        "performed": False,
        "timing": "selection, confirmation, diagnostics, and verdict frozen",
    }
    common.write_json(RESULTS_PATH, results)

    print("=== FINAL TEST EVALUATION (choice and verdict already frozen) ===", flush=True)
    lgb_ref_test = lgb_ref_model.predict(frames["test"][common.LGB_REFERENCE_COLUMNS])
    lgb_candidate_test = (
        lgb_watch_model.predict(frames["test"][lgb_watch_columns])
        if selected_spec["lgb_feature"]
        else lgb_ref_test
    )
    xgb_a0_test = xgb_a0_model.predict(frames["test"][common.CURRENT_XGB_COLUMNS])
    selected_xgb_model = (
        xgb_models[selected_spec["xgb_feature"]]
        if selected_spec["xgb_feature"]
        else xgb_a0_model
    )
    selected_xgb_columns = (
        common.XGB_REFERENCE_COLUMNS + [selected_spec["xgb_feature"]]
        if selected_spec["xgb_feature"]
        else common.CURRENT_XGB_COLUMNS
    )
    xgb_candidate_test = selected_xgb_model.predict(
        frames["test"][selected_xgb_columns]
    )

    # FM is trained only after validation selection and the verdict have been
    # written.  Its test predictions are unchanged from the reference system.
    fm_context, fm_valid_refit, fm_valid_metrics = blend.fit_current_fm_validation(
        y, users
    )
    if abs(fm_valid_metrics["primary"] - common.FM_REFERENCE_VALID) > 1e-8:
        raise AssertionError("FM refit drift before final test")
    Xte_fm, yte_fm, ute_fm = fm_context["encoded"]["test"]
    if not np.array_equal(np.asarray(yte_fm), y["test"]):
        raise AssertionError("FM/native test labels differ")
    if not np.array_equal(np.asarray(ute_fm), np.asarray(users["test"])):
        raise AssertionError("FM/native test users differ")
    fm_test = np.mean(
        np.stack(
            [
                fm_context["module"].sigmoid(model.predict(Xte_fm))
                for model in fm_context["models"]
            ]
        ),
        axis=0,
    )
    reference_test_components = normalized(
        blend, fm_test, lgb_ref_test, xgb_a0_test, users["test"]
    )
    candidate_test_components = normalized(
        blend, fm_test, lgb_candidate_test, xgb_candidate_test, users["test"]
    )
    _, reference_test_metrics = score(
        blend,
        reference_test_components,
        common.CURRENT_WEIGHTS,
        users["test"],
        y["test"],
    )
    if (
        abs(reference_test_metrics["primary"] - common.ENSEMBLE_REFERENCE_TEST)
        > 1e-8
    ):
        raise AssertionError("post-6h test reference drift")
    _, candidate_test_metrics = score(
        blend,
        candidate_test_components,
        selected_weights,
        users["test"],
        y["test"],
    )
    results["components_test"] = {
        "fm": common.metric_dict(common.evaluate(users["test"], y["test"], fm_test)),
        "reference_lgb": common.metric_dict(
            common.evaluate(users["test"], y["test"], lgb_ref_test)
        ),
        "candidate_lgb": common.metric_dict(
            common.evaluate(users["test"], y["test"], lgb_candidate_test)
        ),
        "reference_xgb_A0": common.metric_dict(
            common.evaluate(users["test"], y["test"], xgb_a0_test)
        ),
        "candidate_xgb": common.metric_dict(
            common.evaluate(users["test"], y["test"], xgb_candidate_test)
        ),
    }
    results["reference"]["test"] = common.metric_dict(reference_test_metrics)
    results["selected_on_validation"]["test"] = common.metric_dict(
        candidate_test_metrics
    )
    results["selected_on_validation"]["delta_vs_reference_test"] = float(
        candidate_test_metrics["primary"] - reference_test_metrics["primary"]
    )
    results["test_evaluation"] = {
        "performed": True,
        "timing": "after validation selection, confirmation, diagnostics, and verdict freeze",
    }
    common.write_json(RESULTS_PATH, results)
    print(
        f"final valid={selected['valid']['primary']:.8f} "
        f"test={candidate_test_metrics['primary']:.8f} "
        f"verdict={results['verdict']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
