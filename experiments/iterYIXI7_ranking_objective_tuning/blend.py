"""Final 6g percentile ensemble reconstruction and validation-only calibration.

Only ranking-objective branches already confirmed by the standalone runners
are eligible.  The unchanged YIXI5 ensemble is reproduced first.  Test scores
are not predicted until weights, paired confirmation, diagnostics, and the
overall verdict have been frozen and written.
"""

from __future__ import annotations

import gc
import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_A_PATH = os.path.join(THIS_DIR, "phase_a_results.json")
PHASE_B_PATH = os.path.join(THIS_DIR, "phase_b_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "blend_results.json")


def score(yixi5_blend, components, weights, users, labels):
    values = yixi5_blend.combined_scores(components, weights)
    return values, common.evaluate(users, labels, values)


def normalized(yixi5_blend, fm, lgb, xgb, users):
    return yixi5_blend.normalize_components(
        fm, lgb, xgb, users, "within_user_percentile"
    )


def score_distribution(scores):
    values = np.asarray(scores, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q50": float(np.quantile(values, 0.50)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "finite": bool(np.all(np.isfinite(values))),
    }


def correlation(left, right):
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def main():
    phase_a = common.read_json(PHASE_A_PATH)
    phase_b = common.read_json(PHASE_B_PATH)
    confirmed = {
        "lgb": bool(phase_a["confirmed_branch"]),
        "xgb": bool(phase_b["confirmed_branch"]),
    }
    if not any(confirmed.values()):
        raise RuntimeError("no confirmed standalone branch is eligible for blending")

    frames, y, users, feature_metadata = common.load_frames()
    yixi5_blend = common.load_module(
        common.YIXI5_BLEND_PATH, "yixi5_blend_for_yixi7_final"
    )
    selected_lgb_config = phase_a["selected_on_validation"]["rank_config"]
    selected_xgb_config = phase_b["selected_on_validation"]["rank_config"]

    results = {
        "experiment": "iterYIXI7_final_percentile_ensemble",
        "environment": common.environment(),
        "feature_metadata": feature_metadata,
        "selection_policy": {
            "branch_eligibility": "standalone five-seed confirmation at >=0.001 on every seed",
            "normalization": "YIXI5 within-user average-tie percentile, fixed",
            "weights": "coarse 0.10 simplex then local 0.02 refinement on validation",
            "promotion_reference": common.ENSEMBLE_VALID,
            "promotion_delta": common.PROMOTION_DELTA,
            "test_access": "after fixed-weight paired confirmation, diagnostics, and verdict freeze",
        },
        "confirmed_standalone_branches": confirmed,
        "selected_rank_configs": {
            "lgb": selected_lgb_config,
            "xgb": selected_xgb_config,
        },
    }

    print("=== exact current component and ensemble fidelity ===", flush=True)
    current_lgb_model, current_lgb_scores, current_lgb_metrics = common.fit_lgb_current(
        frames, y, users, 0
    )
    current_xgb_model, current_xgb_scores, current_xgb_metrics = common.fit_xgb_current(
        frames, y, users, 0
    )
    if abs(current_lgb_metrics["primary"] - common.CURRENT_LGB_VALID) > 1e-8:
        raise AssertionError("current LightGBM validation drift")
    if abs(current_xgb_metrics["primary"] - common.CURRENT_XGB_VALID) > 1e-8:
        raise AssertionError("current XGBoost validation drift")

    candidate_lgb_model = None
    candidate_xgb_model = None
    if confirmed["lgb"]:
        (
            candidate_lgb_model,
            candidate_lgb_scores,
            candidate_lgb_metrics,
            _,
        ) = common.fit_lgb(frames, y, users, selected_lgb_config, 0)
        expected = phase_a["selected_on_validation"]["valid"]["primary"]
        if abs(candidate_lgb_metrics["primary"] - expected) > 1e-8:
            raise AssertionError("confirmed LightGBM refit drift")
    else:
        candidate_lgb_scores = current_lgb_scores
        candidate_lgb_metrics = current_lgb_metrics

    if confirmed["xgb"]:
        (
            candidate_xgb_model,
            candidate_xgb_scores,
            candidate_xgb_metrics,
            _,
        ) = common.fit_xgb(frames, y, users, selected_xgb_config, 0)
        expected = phase_b["selected_on_validation"]["valid"]["primary"]
        if abs(candidate_xgb_metrics["primary"] - expected) > 1e-8:
            raise AssertionError("confirmed XGBoost refit drift")
    else:
        candidate_xgb_scores = current_xgb_scores
        candidate_xgb_metrics = current_xgb_metrics

    print("\n=== unchanged FM five-seed ensemble ===", flush=True)
    fm_context, fm_scores, fm_metrics = yixi5_blend.fit_current_fm_validation(y, users)
    if abs(fm_metrics["primary"] - common.FM_VALID) > 1e-8:
        raise AssertionError("FM validation drift")

    current_components = normalized(
        yixi5_blend,
        fm_scores,
        current_lgb_scores,
        current_xgb_scores,
        users["valid"],
    )
    current_scores, current_metrics = score(
        yixi5_blend,
        current_components,
        common.CURRENT_WEIGHTS,
        users["valid"],
        y["valid"],
    )
    if abs(current_metrics["primary"] - common.ENSEMBLE_VALID) > 1e-8:
        raise AssertionError(
            f"YIXI5 ensemble drift: {current_metrics['primary']} "
            f"vs {common.ENSEMBLE_VALID}"
        )
    print(f"YIXI5 ensemble PASSED: {current_metrics['primary']:.8f}", flush=True)

    candidate_components = normalized(
        yixi5_blend,
        fm_scores,
        candidate_lgb_scores,
        candidate_xgb_scores,
        users["valid"],
    )
    _, fixed_metrics = score(
        yixi5_blend,
        candidate_components,
        common.CURRENT_WEIGHTS,
        users["valid"],
        y["valid"],
    )

    print("\n=== validation-only local percentile-weight calibration ===", flush=True)
    best_three, best_overall, records = yixi5_blend.score_weight_grid(
        candidate_components,
        users["valid"],
        y["valid"],
        "within_user_percentile",
    )
    selected_weights = best_three["weights"]
    selected_scores = yixi5_blend.combined_scores(
        candidate_components, selected_weights
    )
    selected_delta = float(
        best_three["valid"]["primary"] - current_metrics["primary"]
    )
    print(
        f"selected weights={selected_weights} "
        f"valid={best_three['valid']['primary']:.8f} delta={selected_delta:+.8f}",
        flush=True,
    )

    current_lgb_pct = current_components["lgb"]
    current_xgb_pct = current_components["xgb"]
    candidate_lgb_pct = candidate_components["lgb"]
    candidate_xgb_pct = candidate_components["xgb"]
    results["components_valid"] = {
        "fm": common.metric_dict(fm_metrics),
        "current_lgb": common.metric_dict(current_lgb_metrics),
        "candidate_lgb": common.metric_dict(candidate_lgb_metrics),
        "current_xgb": common.metric_dict(current_xgb_metrics),
        "candidate_xgb": common.metric_dict(candidate_xgb_metrics),
    }
    results["reference"] = {
        "weights": common.CURRENT_WEIGHTS,
        "normalization": "within_user_percentile",
        "valid": common.metric_dict(current_metrics),
    }
    results["fixed_current_weight_candidate"] = {
        "weights": common.CURRENT_WEIGHTS,
        "valid": common.metric_dict(fixed_metrics),
        "delta_vs_reference": float(
            fixed_metrics["primary"] - current_metrics["primary"]
        ),
    }
    results["weight_sweep"] = {
        "best_three_model": best_three,
        "best_including_ablations": best_overall,
        "records": records,
    }
    results["selected_on_validation"] = best_three
    results["selected_on_validation"]["delta_vs_reference"] = selected_delta
    results["diagnostics"] = {
        "metric_delta_vs_reference": {
            metric: float(best_three["valid"][metric] - current_metrics[metric])
            for metric in ("GAUC", "nDCG@5", "primary")
        },
        "tie_checks": {
            "current_lgb": common.tie_stats(current_lgb_scores, users["valid"]),
            "candidate_lgb": common.tie_stats(candidate_lgb_scores, users["valid"]),
            "current_xgb": common.tie_stats(current_xgb_scores, users["valid"]),
            "candidate_xgb": common.tie_stats(candidate_xgb_scores, users["valid"]),
            "reference_ensemble": common.tie_stats(current_scores, users["valid"]),
            "candidate_ensemble": common.tie_stats(selected_scores, users["valid"]),
        },
        "raw_score_distributions": {
            "current_lgb": score_distribution(current_lgb_scores),
            "candidate_lgb": score_distribution(candidate_lgb_scores),
            "current_xgb": score_distribution(current_xgb_scores),
            "candidate_xgb": score_distribution(candidate_xgb_scores),
        },
        "tree_percentile_correlation": {
            "current_lgb_vs_current_xgb": correlation(
                current_lgb_pct, current_xgb_pct
            ),
            "candidate_lgb_vs_candidate_xgb": correlation(
                candidate_lgb_pct, candidate_xgb_pct
            ),
            "delta": correlation(candidate_lgb_pct, candidate_xgb_pct)
            - correlation(current_lgb_pct, current_xgb_pct),
        },
        "confound_audit": {
            "features_fixed_within_each_standalone_sweep": True,
            "ordinary_tree_parameters_fixed": True,
            "only_confirmed_branches_entered_ensemble": True,
            "unchanged_fm_reproduced": True,
            "unchanged_current_lgb_reproduced": True,
            "unchanged_current_xgb_reproduced": True,
            "reference_ensemble_reproduced": True,
            "test_used_for_selection": False,
        },
        "top_nearby_three_model_points": sorted(
            (
                row
                for row in records
                if all(row["weights"][part] > 0 for part in ("fm", "lgb", "xgb"))
            ),
            key=lambda row: row["valid"]["primary"],
            reverse=True,
        )[:10],
    }

    confirmation_rows = []
    if selected_delta >= common.PROMOTION_DELTA:
        print("\n=== paired five-seed fixed-ensemble confirmation ===", flush=True)
        confirmation_rows.append(
            {
                "seed": 0,
                "reference_valid": common.metric_dict(current_metrics),
                "candidate_valid": best_three["valid"],
                "delta": selected_delta,
            }
        )
        for seed in common.SEEDS[1:]:
            ref_lgb_model, ref_lgb_scores, _ = common.fit_lgb_current(
                frames, y, users, seed
            )
            if confirmed["lgb"]:
                cand_lgb_model, cand_lgb_scores, _, _ = common.fit_lgb(
                    frames, y, users, selected_lgb_config, seed
                )
            else:
                cand_lgb_model, cand_lgb_scores = ref_lgb_model, ref_lgb_scores

            # XGBoost's fixed full-row/full-column configuration is
            # deterministic across random_state.  It is unchanged in this
            # experiment because Phase B rejected every ranking change.
            ref_components = normalized(
                yixi5_blend,
                fm_scores,
                ref_lgb_scores,
                current_xgb_scores,
                users["valid"],
            )
            cand_components = normalized(
                yixi5_blend,
                fm_scores,
                cand_lgb_scores,
                candidate_xgb_scores,
                users["valid"],
            )
            _, ref_metrics = score(
                yixi5_blend,
                ref_components,
                common.CURRENT_WEIGHTS,
                users["valid"],
                y["valid"],
            )
            _, cand_metrics = score(
                yixi5_blend,
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
            del ref_components, cand_components
            if confirmed["lgb"]:
                del cand_lgb_model, cand_lgb_scores
            del ref_lgb_model, ref_lgb_scores
            gc.collect()

    confirmation = common.confirmation_summary(confirmation_rows)
    results["five_seed_confirmation"] = confirmation
    results["verdict"] = "PROMOTE" if confirmation["confirmed"] else "REJECT"
    results["test_evaluation"] = {
        "performed": False,
        "timing": "selection, confirmation, diagnostics, and verdict frozen",
    }
    common.write_json(RESULTS_PATH, results)

    print("\n=== FINAL TEST EVALUATION (selection and verdict frozen) ===", flush=True)
    current_lgb_test = current_lgb_model.predict(
        frames["test"][common.CURRENT_LGB_COLUMNS]
    )
    current_xgb_test = current_xgb_model.predict(
        frames["test"][common.CURRENT_XGB_COLUMNS]
    )
    if confirmed["lgb"]:
        candidate_lgb_test = candidate_lgb_model.predict(
            frames["test"][common.LGB_TUNING_COLUMNS]
        )
    else:
        candidate_lgb_test = current_lgb_test
    if confirmed["xgb"]:
        candidate_xgb_test = candidate_xgb_model.predict(
            frames["test"][common.XGB_TUNING_COLUMNS]
        )
    else:
        candidate_xgb_test = current_xgb_test

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
    current_test_components = normalized(
        yixi5_blend,
        fm_test,
        current_lgb_test,
        current_xgb_test,
        users["test"],
    )
    candidate_test_components = normalized(
        yixi5_blend,
        fm_test,
        candidate_lgb_test,
        candidate_xgb_test,
        users["test"],
    )
    _, current_test_metrics = score(
        yixi5_blend,
        current_test_components,
        common.CURRENT_WEIGHTS,
        users["test"],
        y["test"],
    )
    if abs(current_test_metrics["primary"] - common.ENSEMBLE_TEST) > 1e-8:
        raise AssertionError("YIXI5 test ensemble drift")
    _, selected_test_metrics = score(
        yixi5_blend,
        candidate_test_components,
        selected_weights,
        users["test"],
        y["test"],
    )
    results["components_test"] = {
        "fm": common.metric_dict(common.evaluate(users["test"], y["test"], fm_test)),
        "current_lgb": common.metric_dict(
            common.evaluate(users["test"], y["test"], current_lgb_test)
        ),
        "candidate_lgb": common.metric_dict(
            common.evaluate(users["test"], y["test"], candidate_lgb_test)
        ),
        "current_xgb": common.metric_dict(
            common.evaluate(users["test"], y["test"], current_xgb_test)
        ),
        "candidate_xgb": common.metric_dict(
            common.evaluate(users["test"], y["test"], candidate_xgb_test)
        ),
    }
    results["reference"]["test"] = common.metric_dict(current_test_metrics)
    results["selected_on_validation"]["test"] = common.metric_dict(
        selected_test_metrics
    )
    results["selected_on_validation"]["delta_vs_reference_test"] = float(
        selected_test_metrics["primary"] - current_test_metrics["primary"]
    )
    results["test_evaluation"] = {
        "performed": True,
        "timing": "after selection, confirmation, diagnostics, and verdict freeze",
    }
    common.write_json(RESULTS_PATH, results)
    print(
        f"final valid={best_three['valid']['primary']:.8f} "
        f"test={selected_test_metrics['primary']:.8f} verdict={results['verdict']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
