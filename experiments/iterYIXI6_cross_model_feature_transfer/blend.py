"""6f final ensemble: frozen representations, percentile weight calibration.

Representations come only from standalone validation selection in
``representation_results.json``.  This runner first reproduces the exact
YIXI5 validation blend, then performs validation-only weight sweeps.  Test is
first predicted after fixed-weight paired confirmation and the verdict freeze.
"""

from __future__ import annotations

import gc
import json
import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPRESENTATION_PATH = os.path.join(THIS_DIR, "representation_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "blend_results.json")
YIXI5_BLEND_PATH = os.path.join(
    common.REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "blend.py"
)
CURRENT_WEIGHTS = {"fm": 0.24, "lgb": 0.40, "xgb": 0.36}


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalized(yixi5_blend, fm, lgb, xgb, users):
    return yixi5_blend.normalize_components(
        fm, lgb, xgb, users, "within_user_percentile"
    )


def score(yixi5_blend, components, weights, users, labels):
    values = yixi5_blend.combined_scores(components, weights)
    return values, common.evaluate(users, labels, values)


def best_ablation(records, zero_component: str):
    eligible = [row for row in records if row["weights"][zero_component] == 0]
    return max(eligible, key=lambda row: row["valid"]["primary"])


def main() -> None:
    if not os.path.exists(REPRESENTATION_PATH):
        raise FileNotFoundError("run composition.py before blend.py")
    representation = read_json(REPRESENTATION_PATH)
    selected_xgb_columns = representation["selected_xgb"]["columns"]
    selected_lgb_columns = representation["selected_lgb"]["columns"]
    frames, y, users, metadata = common.load_frames(common.DATA_DIR)
    yixi5_blend = common.load_module(YIXI5_BLEND_PATH, "yixi5_blend_for_yixi6_final")

    results = {
        "experiment": "iterYIXI6_final_percentile_ensemble",
        "environment": common.environment(),
        "feature_metadata": metadata,
        "selected_representations": {
            "xgb": representation["selected_xgb"],
            "lgb": representation["selected_lgb"],
        },
        "selection_policy": {
            "representations": "standalone validation only, frozen before this runner",
            "normalization": "YIXI5 within-user average-tie percentile, fixed",
            "weights": "coarse 0.10 simplex then local 0.02 refinement on validation",
            "promotion_delta": common.PROMOTION_DELTA,
            "test_access": "after fixed-weight five-seed confirmation and verdict freeze",
        },
    }

    print("=== fitting current and selected tree components on validation ===", flush=True)
    current_xgb_model, current_xgb_scores, current_xgb_metrics, _ = common.fit_xgb(
        frames, y, users, common.XGB_A0_COLUMNS, 0
    )
    selected_xgb_model, selected_xgb_scores, selected_xgb_metrics, _ = common.fit_xgb(
        frames, y, users, selected_xgb_columns, 0
    )
    current_lgb_model, current_lgb_scores, current_lgb_metrics, _ = common.fit_lgb(
        frames, y, users, common.LGB_B0_COLUMNS, 0
    )
    selected_lgb_model, selected_lgb_scores, selected_lgb_metrics, _ = common.fit_lgb(
        frames, y, users, selected_lgb_columns, 0
    )
    if abs(current_xgb_metrics["primary"] - common.XGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("current XGBoost drift in final blend")
    if abs(current_lgb_metrics["primary"] - common.LGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("current LightGBM drift in final blend")
    if (
        abs(
            selected_xgb_metrics["primary"]
            - representation["selected_xgb"]["valid"]["primary"]
        )
        > 1e-8
    ):
        raise AssertionError("selected XGBoost refit drift")
    if (
        abs(
            selected_lgb_metrics["primary"]
            - representation["selected_lgb"]["valid"]["primary"]
        )
        > 1e-8
    ):
        raise AssertionError("selected LightGBM refit drift")
    print(
        f"  XGB current={current_xgb_metrics['primary']:.8f} "
        f"selected={selected_xgb_metrics['primary']:.8f}",
        flush=True,
    )
    print(
        f"  LGB current={current_lgb_metrics['primary']:.8f} "
        f"selected={selected_lgb_metrics['primary']:.8f}",
        flush=True,
    )

    print("\n=== fitting unchanged FM five-seed ensemble ===", flush=True)
    fm_context, fm_scores, fm_metrics = yixi5_blend.fit_current_fm_validation(y, users)
    if abs(fm_metrics["primary"] - common.FM_REFERENCE_VALID) > 1e-8:
        raise AssertionError("FM drift in final blend")

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
        CURRENT_WEIGHTS,
        users["valid"],
        y["valid"],
    )
    if abs(current_metrics["primary"] - common.BLEND_REFERENCE_VALID) > 1e-8:
        raise AssertionError(
            f"YIXI5 final blend drift: {current_metrics['primary']} "
            f"vs {common.BLEND_REFERENCE_VALID}"
        )
    print(f"YIXI5 final blend PASSED: {current_metrics['primary']:.8f}", flush=True)

    current_xgb_only_components = normalized(
        yixi5_blend,
        fm_scores,
        current_lgb_scores,
        selected_xgb_scores,
        users["valid"],
    )
    current_lgb_only_components = normalized(
        yixi5_blend,
        fm_scores,
        selected_lgb_scores,
        current_xgb_scores,
        users["valid"],
    )
    selected_components = normalized(
        yixi5_blend,
        fm_scores,
        selected_lgb_scores,
        selected_xgb_scores,
        users["valid"],
    )

    # Fixed-current-weight scores isolate representation effects before any
    # recalibration.  These are diagnostics and do not select representations.
    fixed_weight_diagnostics = {}
    for name, components in (
        ("current", current_components),
        ("xgb_transfer_only", current_xgb_only_components),
        ("lgb_transfer_only", current_lgb_only_components),
        ("both_transfers", selected_components),
    ):
        _, metrics = score(
            yixi5_blend,
            components,
            CURRENT_WEIGHTS,
            users["valid"],
            y["valid"],
        )
        fixed_weight_diagnostics[name] = common.metric_dict(metrics)

    print("\n=== validation-only percentile weight sweeps ===", flush=True)
    sweep_inputs = {
        "xgb_transfer_only": current_xgb_only_components,
        "lgb_transfer_only": current_lgb_only_components,
        "both_transfers": selected_components,
    }
    sweep_results = {}
    for name, components in sweep_inputs.items():
        best_three, best_overall, records = yixi5_blend.score_weight_grid(
            components, users["valid"], y["valid"], "within_user_percentile"
        )
        best_three["delta_vs_yixi5"] = float(
            best_three["valid"]["primary"] - current_metrics["primary"]
        )
        sweep_results[name] = {
            "best_three_model": best_three,
            "best_including_ablations": best_overall,
            "records": records,
        }
        print(
            f"  {name}: weights={best_three['weights']} "
            f"valid={best_three['valid']['primary']:.8f} "
            f"delta={best_three['delta_vs_yixi5']:+.8f}",
            flush=True,
        )

    # Representation selection was already frozen; only the both-transfer
    # calibration is eligible as the final 6f candidate.
    selected = sweep_results["both_transfers"]["best_three_model"]
    selected_weights = selected["weights"]
    selected_scores = yixi5_blend.combined_scores(
        selected_components, selected_weights
    )
    selected_delta = float(selected["valid"]["primary"] - current_metrics["primary"])
    results["components_valid"] = {
        "fm": common.metric_dict(fm_metrics),
        "current_xgb": common.metric_dict(current_xgb_metrics),
        "selected_xgb": common.metric_dict(selected_xgb_metrics),
        "current_lgb": common.metric_dict(current_lgb_metrics),
        "selected_lgb": common.metric_dict(selected_lgb_metrics),
    }
    results["reference"] = {
        "weights": CURRENT_WEIGHTS,
        "normalization": "within_user_percentile",
        "valid": common.metric_dict(current_metrics),
    }
    results["fixed_current_weight_diagnostics"] = fixed_weight_diagnostics
    results["weight_sweeps"] = sweep_results
    results["selected_on_validation"] = selected
    results["selected_on_validation"]["delta_vs_yixi5"] = selected_delta

    records = sweep_results["both_transfers"]["records"]
    results["diagnostics"] = {
        "selected_ties": common.tie_stats(selected_scores, users["valid"]),
        "metric_delta_vs_yixi5": {
            key: float(selected["valid"][key] - current_metrics[key])
            for key in ("GAUC", "nDCG@5", "primary")
        },
        "best_no_xgb": best_ablation(records, "xgb"),
        "best_no_lgb": best_ablation(records, "lgb"),
        "best_no_fm": best_ablation(records, "fm"),
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
    common.write_json(RESULTS_PATH, results)

    confirmation_rows = []
    if selected_delta >= common.PROMOTION_DELTA:
        print("\n=== paired five-seed fixed-ensemble confirmation ===", flush=True)
        confirmation_rows.append(
            {
                "seed": 0,
                "reference_valid": common.metric_dict(current_metrics),
                "candidate_valid": selected["valid"],
                "delta": selected_delta,
            }
        )
        for seed in common.SEEDS[1:]:
            ref_xgb_model, ref_xgb_scores, _, _ = common.fit_xgb(
                frames, y, users, common.XGB_A0_COLUMNS, seed
            )
            cand_xgb_model, cand_xgb_scores, _, _ = common.fit_xgb(
                frames, y, users, selected_xgb_columns, seed
            )
            ref_lgb_model, ref_lgb_scores, _, _ = common.fit_lgb(
                frames, y, users, common.LGB_B0_COLUMNS, seed
            )
            cand_lgb_model, cand_lgb_scores, _, _ = common.fit_lgb(
                frames, y, users, selected_lgb_columns, seed
            )
            seed_reference_components = normalized(
                yixi5_blend,
                fm_scores,
                ref_lgb_scores,
                ref_xgb_scores,
                users["valid"],
            )
            seed_candidate_components = normalized(
                yixi5_blend,
                fm_scores,
                cand_lgb_scores,
                cand_xgb_scores,
                users["valid"],
            )
            _, seed_reference_metrics = score(
                yixi5_blend,
                seed_reference_components,
                CURRENT_WEIGHTS,
                users["valid"],
                y["valid"],
            )
            _, seed_candidate_metrics = score(
                yixi5_blend,
                seed_candidate_components,
                selected_weights,
                users["valid"],
                y["valid"],
            )
            delta = float(
                seed_candidate_metrics["primary"] - seed_reference_metrics["primary"]
            )
            confirmation_rows.append(
                {
                    "seed": seed,
                    "reference_valid": common.metric_dict(seed_reference_metrics),
                    "candidate_valid": common.metric_dict(seed_candidate_metrics),
                    "delta": delta,
                }
            )
            print(
                f"  seed={seed} reference={seed_reference_metrics['primary']:.8f} "
                f"candidate={seed_candidate_metrics['primary']:.8f} delta={delta:+.8f}",
                flush=True,
            )
            del (
                ref_xgb_model,
                ref_xgb_scores,
                cand_xgb_model,
                cand_xgb_scores,
                ref_lgb_model,
                ref_lgb_scores,
                cand_lgb_model,
                cand_lgb_scores,
                seed_reference_components,
                seed_candidate_components,
            )
            gc.collect()

    deltas = np.asarray([row["delta"] for row in confirmation_rows], dtype=np.float64)
    confirmed = bool(
        len(deltas) == len(common.SEEDS)
        and np.mean(deltas) >= common.PROMOTION_DELTA
        and np.all(deltas >= common.PROMOTION_DELTA)
    )
    results["five_seed_confirmation"] = {
        "performed": bool(confirmation_rows),
        "rows": confirmation_rows,
        "mean_delta": float(np.mean(deltas)) if len(deltas) else None,
        "std_delta": float(np.std(deltas)) if len(deltas) else None,
        "min_delta": float(np.min(deltas)) if len(deltas) else None,
        "confirmed": confirmed,
    }
    results["verdict"] = "PROMOTE" if confirmed else "REJECT"
    common.write_json(RESULTS_PATH, results)

    # First test-score access in all 6f candidate runners.  Representations,
    # weights, confirmation, diagnostics, and verdict are already frozen.
    print("\n=== FINAL TEST EVALUATION (selection frozen) ===", flush=True)
    current_xgb_test_scores = current_xgb_model.predict(
        frames["test"][common.XGB_A0_COLUMNS]
    )
    selected_xgb_test_scores = selected_xgb_model.predict(
        frames["test"][selected_xgb_columns]
    )
    current_lgb_test_scores = current_lgb_model.predict(
        frames["test"][common.LGB_B0_COLUMNS]
    )
    selected_lgb_test_scores = selected_lgb_model.predict(
        frames["test"][selected_lgb_columns]
    )
    Xte_fm, yte_fm, ute_fm = fm_context["encoded"]["test"]
    if not np.array_equal(np.asarray(yte_fm), y["test"]):
        raise AssertionError("FM/native test labels differ")
    if not np.array_equal(np.asarray(ute_fm), np.asarray(users["test"])):
        raise AssertionError("FM/native test users differ")
    fm_test_scores = np.mean(
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
        fm_test_scores,
        current_lgb_test_scores,
        current_xgb_test_scores,
        users["test"],
    )
    selected_test_components = normalized(
        yixi5_blend,
        fm_test_scores,
        selected_lgb_test_scores,
        selected_xgb_test_scores,
        users["test"],
    )
    _, current_test_metrics = score(
        yixi5_blend,
        current_test_components,
        CURRENT_WEIGHTS,
        users["test"],
        y["test"],
    )
    if abs(current_test_metrics["primary"] - common.BLEND_REFERENCE_TEST) > 1e-8:
        raise AssertionError("YIXI5 test reference drift")
    _, selected_test_metrics = score(
        yixi5_blend,
        selected_test_components,
        selected_weights,
        users["test"],
        y["test"],
    )
    results["components_test"] = {
        "fm": common.metric_dict(common.evaluate(users["test"], y["test"], fm_test_scores)),
        "current_xgb": common.metric_dict(
            common.evaluate(users["test"], y["test"], current_xgb_test_scores)
        ),
        "selected_xgb": common.metric_dict(
            common.evaluate(users["test"], y["test"], selected_xgb_test_scores)
        ),
        "current_lgb": common.metric_dict(
            common.evaluate(users["test"], y["test"], current_lgb_test_scores)
        ),
        "selected_lgb": common.metric_dict(
            common.evaluate(users["test"], y["test"], selected_lgb_test_scores)
        ),
    }
    results["reference"]["test"] = common.metric_dict(current_test_metrics)
    results["selected_on_validation"]["test"] = common.metric_dict(selected_test_metrics)
    results["selected_on_validation"]["delta_vs_yixi5_test"] = float(
        selected_test_metrics["primary"] - current_test_metrics["primary"]
    )
    results["test_evaluation"] = {
        "performed": True,
        "timing": "after representation/weight selection, confirmation, diagnostics, and verdict",
    }
    common.write_json(RESULTS_PATH, results)
    print(
        f"selected ensemble: valid={selected['valid']['primary']:.8f} "
        f"test={selected_test_metrics['primary']:.8f} verdict={results['verdict']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
