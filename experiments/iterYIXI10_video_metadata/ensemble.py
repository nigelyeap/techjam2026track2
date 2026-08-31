"""Final validation-only ensemble for the confirmed Section 6j branch."""

from __future__ import annotations

import gc
import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LGB_RESULTS_PATH = os.path.join(THIS_DIR, "phase_lgb_results.json")
XGB_RESULTS_PATH = os.path.join(THIS_DIR, "phase_xgb_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "ensemble_results.json")


def normalized(blend, fm, lgb, xgb, users):
    return blend.normalize_components(
        fm, lgb, xgb, users, "within_user_percentile"
    )


def score(blend, components, weights, users, labels):
    scores = blend.combined_scores(components, weights)
    return scores, common.evaluate(users, labels, scores)


def top_nearby(records, limit=15):
    return sorted(
        (
            row for row in records
            if all(row["weights"][name] > 0 for name in ("fm", "lgb", "xgb"))
        ),
        key=lambda row: row["valid"]["primary"],
        reverse=True,
    )[:limit]


def main():
    lgb_phase = common.read_json(LGB_RESULTS_PATH)
    xgb_phase = common.read_json(XGB_RESULTS_PATH)
    if not lgb_phase["confirmed_branch"]:
        raise RuntimeError("no confirmed LightGBM metadata branch")
    if xgb_phase["confirmed_branch"]:
        raise RuntimeError("unexpected confirmed XGBoost branch; update composition explicitly")
    selected_groups = lgb_phase["selected_on_validation"]["groups"]
    selected_columns = lgb_phase["selected_on_validation"]["columns"]
    if selected_groups != ["upload_type"]:
        raise AssertionError(f"unexpected selected LightGBM groups {selected_groups}")

    frames, y, users, metadata = common.features.load_frames()
    blend = common.load_module(common.YIXI5_BLEND_PATH, "yixi5_blend_for_yixi10_final")
    yixi8 = common.load_module(common.YIXI8_COMMON_PATH, "yixi8_common_for_yixi10_final")
    frozen = yixi8.load_frozen()
    if not np.array_equal(frozen["users"], np.asarray(users["valid"])):
        raise AssertionError("frozen/native users differ")
    if not np.array_equal(frozen["labels"], y["valid"]):
        raise AssertionError("frozen/native labels differ")
    fm_scores = frozen["fm"]
    fm_metrics = common.evaluate(users["valid"], y["valid"], fm_scores)
    if abs(fm_metrics["primary"] - common.FM_REFERENCE_VALID) > 1e-8:
        raise AssertionError("FM validation drift")

    print("=== exact current components and ensemble ===", flush=True)
    ref_lgb_model, ref_lgb_scores, ref_lgb_metrics, _ = common.fit_lgb(
        frames, y, users, common.LGB_REFERENCE_COLUMNS, 0
    )
    cand_lgb_model, cand_lgb_scores, cand_lgb_metrics, cand_gain = common.fit_lgb(
        frames, y, users, selected_columns, 0
    )
    xgb_model, xgb_scores, xgb_metrics, _ = common.fit_xgb(
        frames, y, users, common.XGB_REFERENCE_COLUMNS, 0
    )
    if abs(ref_lgb_metrics["primary"] - common.LGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("reference LightGBM drift")
    expected_lgb = lgb_phase["selected_on_validation"]["valid"]["primary"]
    if abs(cand_lgb_metrics["primary"] - expected_lgb) > 1e-8:
        raise AssertionError("candidate LightGBM drift")
    if abs(xgb_metrics["primary"] - common.XGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("reference XGBoost drift")
    reference_components = normalized(
        blend, fm_scores, ref_lgb_scores, xgb_scores, users["valid"]
    )
    reference_scores, reference_metrics = score(
        blend, reference_components, common.CURRENT_WEIGHTS,
        users["valid"], y["valid"]
    )
    if abs(reference_metrics["primary"] - common.ENSEMBLE_REFERENCE_VALID) > 1e-8:
        raise AssertionError("YIXI9 ensemble drift")
    print(f"reference PASSED={reference_metrics['primary']:.8f}", flush=True)

    candidate_components = normalized(
        blend, fm_scores, cand_lgb_scores, xgb_scores, users["valid"]
    )
    fixed_scores, fixed_metrics = score(
        blend, candidate_components, common.CURRENT_WEIGHTS,
        users["valid"], y["valid"]
    )
    print("=== validation-only percentile weight sweep ===", flush=True)
    best_three, best_overall, records = blend.score_weight_grid(
        candidate_components, users["valid"], y["valid"],
        "within_user_percentile"
    )
    selected_weights = best_three["weights"]
    selected_scores = blend.combined_scores(candidate_components, selected_weights)
    selected_delta = float(
        best_three["valid"]["primary"] - reference_metrics["primary"]
    )
    print(
        f"weights={selected_weights} valid={best_three['valid']['primary']:.8f} "
        f"delta={selected_delta:+.8f}", flush=True
    )

    results = {
        "experiment": "iterYIXI10_final_metadata_ensemble",
        "environment": common.environment(),
        "feature_metadata": metadata,
        "selection_policy": {
            "branch_eligibility": "standalone five-seed confirmed branch only",
            "selector": "final validation primary only",
            "normalization": "within-user average-tie percentile",
            "weights": "coarse 0.10 simplex then local 0.02 refinement",
            "promotion_delta": common.PROMOTION_DELTA,
            "test_access": "none until branch, weights, confirmation, diagnostics, and verdict are frozen",
        },
        "confirmed_branches": {"lgb": True, "xgb": False},
        "selected_lgb_groups": selected_groups,
        "components_valid": {
            "fm": common.metric_dict(fm_metrics),
            "reference_lgb": common.metric_dict(ref_lgb_metrics),
            "candidate_lgb": common.metric_dict(cand_lgb_metrics),
            "xgb": common.metric_dict(xgb_metrics),
        },
        "reference": {
            "weights": common.CURRENT_WEIGHTS,
            "normalization": "within_user_percentile",
            "valid": common.metric_dict(reference_metrics),
        },
        "fixed_reference_weights": {
            "weights": common.CURRENT_WEIGHTS,
            "valid": common.metric_dict(fixed_metrics),
            "delta_vs_reference": float(
                fixed_metrics["primary"] - reference_metrics["primary"]
            ),
        },
        "weight_sweep": {
            "best_three_model": best_three,
            "best_including_ablations": best_overall,
            "records": records,
        },
        "selected_on_validation": {
            "lgb_groups": selected_groups,
            "lgb_columns": selected_columns,
            "weights": selected_weights,
            "valid": best_three["valid"],
            "delta_vs_reference": selected_delta,
        },
        "diagnostics": {
            "metric_delta_vs_reference": {
                metric: float(best_three["valid"][metric] - reference_metrics[metric])
                for metric in ("GAUC", "nDCG@5", "primary")
            },
            "ties": {
                "reference_lgb": common.unique_stats(ref_lgb_scores, users["valid"]),
                "candidate_lgb": common.unique_stats(cand_lgb_scores, users["valid"]),
                "xgb": common.unique_stats(xgb_scores, users["valid"]),
                "reference_ensemble": common.unique_stats(reference_scores, users["valid"]),
                "candidate_fixed_ensemble": common.unique_stats(fixed_scores, users["valid"]),
                "candidate_selected_ensemble": common.unique_stats(selected_scores, users["valid"]),
            },
            "upload_type_gain_fraction": cand_gain.get("meta_upload_type", 0.0),
            "top_nearby_three_model_points": top_nearby(records),
            "confound_audit": {
                "only_added_model_column": "meta_upload_type",
                "all_other_features_fixed": True,
                "tree_hyperparameters_fixed": True,
                "xgboost_unchanged": True,
                "fm_unchanged": True,
                "statistic_file_read": False,
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
                "candidate_valid": best_three["valid"],
                "delta": selected_delta,
            }
        )
        for seed in common.SEEDS[1:]:
            seed_ref_model, seed_ref_scores, _, _ = common.fit_lgb(
                frames, y, users, common.LGB_REFERENCE_COLUMNS, seed
            )
            seed_cand_model, seed_cand_scores, _, _ = common.fit_lgb(
                frames, y, users, selected_columns, seed
            )
            seed_ref_components = normalized(
                blend, fm_scores, seed_ref_scores, xgb_scores, users["valid"]
            )
            seed_cand_components = normalized(
                blend, fm_scores, seed_cand_scores, xgb_scores, users["valid"]
            )
            _, seed_ref_metrics = score(
                blend, seed_ref_components, common.CURRENT_WEIGHTS,
                users["valid"], y["valid"]
            )
            _, seed_cand_metrics = score(
                blend, seed_cand_components, selected_weights,
                users["valid"], y["valid"]
            )
            delta = float(
                seed_cand_metrics["primary"] - seed_ref_metrics["primary"]
            )
            confirmation_rows.append(
                {
                    "seed": seed,
                    "reference_valid": common.metric_dict(seed_ref_metrics),
                    "candidate_valid": common.metric_dict(seed_cand_metrics),
                    "delta": delta,
                }
            )
            print(
                f"  seed={seed} reference={seed_ref_metrics['primary']:.8f} "
                f"candidate={seed_cand_metrics['primary']:.8f} delta={delta:+.8f}",
                flush=True,
            )
            del seed_ref_model, seed_ref_scores, seed_cand_model, seed_cand_scores
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
    ref_lgb_test = ref_lgb_model.predict(frames["test"][common.LGB_REFERENCE_COLUMNS])
    cand_lgb_test = cand_lgb_model.predict(frames["test"][selected_columns])
    xgb_test = xgb_model.predict(frames["test"][common.XGB_REFERENCE_COLUMNS])
    fm_context, _, fm_refit_metrics = blend.fit_current_fm_validation(y, users)
    if abs(fm_refit_metrics["primary"] - common.FM_REFERENCE_VALID) > 1e-8:
        raise AssertionError("FM refit drift")
    Xte_fm, yte_fm, ute_fm = fm_context["encoded"]["test"]
    if not np.array_equal(np.asarray(yte_fm), y["test"]):
        raise AssertionError("FM/native test labels differ")
    if not np.array_equal(np.asarray(ute_fm), np.asarray(users["test"])):
        raise AssertionError("FM/native test users differ")
    fm_test = np.mean(
        np.stack([
            fm_context["module"].sigmoid(model.predict(Xte_fm))
            for model in fm_context["models"]
        ]), axis=0
    )
    reference_test_components = normalized(
        blend, fm_test, ref_lgb_test, xgb_test, users["test"]
    )
    candidate_test_components = normalized(
        blend, fm_test, cand_lgb_test, xgb_test, users["test"]
    )
    _, reference_test_metrics = score(
        blend, reference_test_components, common.CURRENT_WEIGHTS,
        users["test"], y["test"]
    )
    if abs(reference_test_metrics["primary"] - common.ENSEMBLE_REFERENCE_TEST) > 1e-8:
        raise AssertionError("YIXI9 test ensemble drift")
    _, candidate_test_metrics = score(
        blend, candidate_test_components, selected_weights,
        users["test"], y["test"]
    )
    results["components_test"] = {
        "fm": common.metric_dict(common.evaluate(users["test"], y["test"], fm_test)),
        "reference_lgb": common.metric_dict(common.evaluate(users["test"], y["test"], ref_lgb_test)),
        "candidate_lgb": common.metric_dict(common.evaluate(users["test"], y["test"], cand_lgb_test)),
        "xgb": common.metric_dict(common.evaluate(users["test"], y["test"], xgb_test)),
    }
    results["reference"]["test"] = common.metric_dict(reference_test_metrics)
    results["selected_on_validation"]["test"] = common.metric_dict(candidate_test_metrics)
    results["selected_on_validation"]["delta_vs_reference_test"] = float(
        candidate_test_metrics["primary"] - reference_test_metrics["primary"]
    )
    results["test_evaluation"] = {
        "performed": True,
        "timing": "after validation selection, confirmation, diagnostics, and verdict freeze",
    }
    common.write_json(RESULTS_PATH, results)
    print(
        f"final valid={best_three['valid']['primary']:.8f} "
        f"test={candidate_test_metrics['primary']:.8f} verdict={results['verdict']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
