"""Validation-only native-LightGBM feature-depth experiment (YIXI 6b).

Run from the repository root:

    python3 experiments/iterYIXI2_feature_depth/run_experiment.py

Test scores are not computed until every feature-family choice, follow-up
combination, diagnostic, and optional five-seed validation check is complete.
"""

from __future__ import annotations

import gc
import json
import os
import platform
from typing import Any

import lightgbm as lgb
import numpy as np

from features import (
    TAB_HALFLIVES,
    USER_HALFLIVES,
    fmt_h,
    iter44,
    prepare_feature_frames,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
RESULTS_PATH = os.path.join(THIS_DIR, "results.json")

import sys

sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402


BASELINE_REFERENCE_VALID = 0.6613549
BASELINE_REFERENCE_TEST = 0.6479420
FOLLOWUP_DELTA = 0.0003
PROMOTION_DELTA = 0.001
SEEDS = (0, 1, 2, 3, 4)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_results(results: dict[str, Any]) -> None:
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, indent=2, sort_keys=True)
        f.write("\n")


def metric_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in metrics.items()}


def stable_user_order(user_ids):
    user_ids = np.asarray(user_ids)
    order = np.argsort(user_ids, kind="stable")
    groups = np.unique(user_ids[order], return_counts=True)[1]
    return order, groups


def make_model(seed: int) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[5],
        num_leaves=2,
        learning_rate=0.05,
        n_estimators=500,
        min_child_samples=200,
        reg_lambda=1.0,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1,
    )


def fit_columns(
    frames,
    y,
    users,
    train_order,
    train_groups,
    valid_order,
    valid_groups,
    columns: list[str],
    seed: int,
):
    Xtr = frames["train"][columns].iloc[train_order].reset_index(drop=True)
    ytr = y["train"][train_order]
    Xva = frames["valid"][columns].iloc[valid_order].reset_index(drop=True)
    yva = y["valid"][valid_order]
    model = make_model(seed)
    model.fit(
        Xtr,
        ytr,
        group=train_groups,
        eval_set=[(Xva, yva)],
        eval_group=[valid_groups],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    valid_scores = model.predict(frames["valid"][columns])
    valid_metrics = evaluate(users["valid"], y["valid"], valid_scores)
    gain = model.booster_.feature_importance(importance_type="gain")
    gain_total = float(gain.sum())
    gain_fraction = {
        column: (float(value) / gain_total if gain_total > 0 else 0.0)
        for column, value in zip(columns, gain)
    }
    del Xtr, Xva
    gc.collect()
    return model, valid_scores, valid_metrics, gain_fraction


def tie_stats(scores, user_ids) -> dict[str, Any]:
    scores = np.asarray(scores)
    user_ids = np.asarray(user_ids)
    fractions = []
    for uid in np.unique(user_ids):
        group_scores = scores[user_ids == uid]
        fractions.append(len(np.unique(group_scores)) / len(group_scores))
    return {
        "unique_scores_overall": int(len(np.unique(scores))),
        "rows": int(len(scores)),
        "mean_per_user_unique_fraction": float(np.mean(fractions)),
    }


def config_columns(base_columns: list[str], added=(), removed=()) -> list[str]:
    removed = set(removed)
    columns = [column for column in base_columns if column not in removed]
    for column in added:
        if column not in columns:
            columns.append(column)
    return columns


def merge_modifications(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    added = []
    removed = []
    for record in records:
        for column in record["config"]["removed"]:
            if column not in removed:
                removed.append(column)
        for column in record["config"]["added"]:
            if column not in added:
                added.append(column)
    return added, removed


def main() -> None:
    print("=== preparing Section 6b feature matrix ===", flush=True)
    frames, y, users, metadata = prepare_feature_frames(DATA_DIR)
    if not metadata["causality"]["passed"]:
        raise AssertionError("causality gate did not pass")
    print(
        "causality gate PASSED: "
        f"max_abs_error={metadata['causality']['global_max_abs_error']:.3e}, "
        f"same_date_rows_excluded={metadata['causality']['same_date_matching_rows_excluded_across_checks']}",
        flush=True,
    )

    base_columns = list(metadata["base_columns"])
    assert base_columns == list(iter44.CAT_COLS + iter44.NUM_COLS)
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])

    results: dict[str, Any] = {
        "experiment": "iterYIXI2_feature_depth",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
        "selection_policy": {
            "selector": "official validation primary only",
            "followup_delta": FOLLOWUP_DELTA,
            "promotion_delta": PROMOTION_DELTA,
            "test_access": "after all feature selection and verification are frozen",
        },
        "fixed_model": {
            "objective": "lambdarank",
            "num_leaves": 2,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 200,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 30,
        },
        "feature_metadata": metadata,
        "candidates": [],
        "stages": {},
    }
    write_results(results)

    records_by_signature: dict[tuple[str, ...], dict[str, Any]] = {}
    baseline_record = None
    best_experimental_record = None
    best_experimental_model = None
    best_experimental_scores = None

    def run_candidate(name: str, family: str, added=(), removed=()):
        nonlocal baseline_record
        nonlocal best_experimental_record, best_experimental_model, best_experimental_scores

        columns = config_columns(base_columns, added, removed)
        signature = tuple(columns)
        if signature in records_by_signature:
            return records_by_signature[signature]
        print(f"  fitting {name} ({len(columns)} columns)", flush=True)
        model, scores, metrics, gain_fraction = fit_columns(
            frames,
            y,
            users,
            train_order,
            train_groups,
            valid_order,
            valid_groups,
            columns,
            seed=0,
        )
        record = {
            "name": name,
            "family": family,
            "config": {
                "added": list(added),
                "removed": list(removed),
                "columns": columns,
            },
            "seed": 0,
            "best_iteration": int(model.best_iteration_),
            "valid": metric_dict(metrics),
            "feature_gain_fraction": gain_fraction,
            "added_feature_gain_fraction": {
                column: gain_fraction[column] for column in added
            },
        }
        if baseline_record is not None:
            record["delta_vs_baseline_valid"] = float(
                metrics["primary"] - baseline_record["valid"]["primary"]
            )
        results["candidates"].append(record)
        records_by_signature[signature] = record
        print(
            f"    valid={metrics['primary']:.5f} "
            f"GAUC={metrics['GAUC']:.5f} nDCG@5={metrics['nDCG@5']:.5f} "
            f"best_iter={model.best_iteration_}",
            flush=True,
        )

        if family == "baseline":
            baseline_record = record
            del model, scores
        elif (
            best_experimental_record is None
            or metrics["primary"] > best_experimental_record["valid"]["primary"] + 1e-12
        ):
            if best_experimental_model is not None:
                del best_experimental_model
            best_experimental_record = record
            best_experimental_model = model
            best_experimental_scores = scores
        else:
            del model, scores
        gc.collect()
        write_results(results)
        return record

    print("\n=== Stage 0: harness fidelity inside the new runner ===", flush=True)
    baseline = run_candidate("iter44_baseline", "baseline")
    baseline_primary = baseline["valid"]["primary"]
    if abs(baseline_primary - BASELINE_REFERENCE_VALID) > 5e-5:
        raise AssertionError(
            f"new runner drifted: {baseline_primary} vs {BASELINE_REFERENCE_VALID}"
        )
    results["stages"]["baseline"] = [baseline["name"]]
    print(f"  baseline reproduction PASSED ({baseline_primary:.8f})", flush=True)

    print("\n=== Stage 1: user decay half-life replacements and parallel additions ===", flush=True)
    user_records = []
    user_parallel_records = []
    for halflife in USER_HALFLIVES:
        suffix = fmt_h(halflife)
        pair = (f"decay_rate_{suffix}", f"decay_act_{suffix}")
        user_records.append(
            run_candidate(
                f"user_h{suffix}_replace",
                "user_halflife",
                added=pair,
                removed=("decay_rate_2.5", "decay_act_2.5"),
            )
        )
        parallel = run_candidate(
            f"user_h{suffix}_parallel",
            "user_halflife",
            added=pair,
        )
        user_records.append(parallel)
        user_parallel_records.append(parallel)
    qualifying_user_parallel = [
        record
        for record in user_parallel_records
        if record["delta_vs_baseline_valid"] >= FOLLOWUP_DELTA
    ]
    if len(qualifying_user_parallel) >= 2:
        added, removed = merge_modifications(qualifying_user_parallel)
        user_records.append(
            run_candidate(
                "user_parallel_qualifying_union",
                "user_halflife",
                added=added,
                removed=removed,
            )
        )
    user_best = max(user_records, key=lambda record: record["valid"]["primary"])
    results["stages"]["user_halflife"] = [record["name"] for record in user_records]
    results["stages"]["user_halflife_best"] = user_best["name"]

    print("\n=== Stage 2: user-tab decay half-life replacements and parallel additions ===", flush=True)
    tab_records = []
    tab_parallel_records = []
    for halflife in TAB_HALFLIVES:
        suffix = fmt_h(halflife)
        feature = (f"decay_tab_{suffix}",)
        tab_records.append(
            run_candidate(
                f"tab_h{suffix}_replace",
                "tab_halflife",
                added=feature,
                removed=("decay_tab_3",),
            )
        )
        parallel = run_candidate(
            f"tab_h{suffix}_parallel",
            "tab_halflife",
            added=feature,
        )
        tab_records.append(parallel)
        tab_parallel_records.append(parallel)
    qualifying_tab_parallel = [
        record
        for record in tab_parallel_records
        if record["delta_vs_baseline_valid"] >= FOLLOWUP_DELTA
    ]
    if len(qualifying_tab_parallel) >= 2:
        added, removed = merge_modifications(qualifying_tab_parallel)
        tab_records.append(
            run_candidate(
                "tab_parallel_qualifying_union",
                "tab_halflife",
                added=added,
                removed=removed,
            )
        )
    tab_best = max(tab_records, key=lambda record: record["valid"]["primary"])
    results["stages"]["tab_halflife"] = [record["name"] for record in tab_records]
    results["stages"]["tab_halflife_best"] = tab_best["name"]

    print("\n=== Stage 3: author/video popularity decay ===", flush=True)
    author_specs = [
        ("author_rate", ("author_decay_rate_2.5",)),
        ("author_activity", ("author_decay_act_2.5",)),
        ("author_rate_activity", ("author_decay_rate_2.5", "author_decay_act_2.5")),
    ]
    video_specs = [
        ("video_rate", ("video_decay_rate_2.5",)),
        ("video_activity", ("video_decay_act_2.5",)),
        ("video_rate_activity", ("video_decay_rate_2.5", "video_decay_act_2.5")),
    ]
    author_records = [
        run_candidate(name, "author_popularity", added=added) for name, added in author_specs
    ]
    video_records = [
        run_candidate(name, "video_popularity", added=added) for name, added in video_specs
    ]
    author_best = max(author_records, key=lambda record: record["valid"]["primary"])
    video_best = max(video_records, key=lambda record: record["valid"]["primary"])
    popularity_records = author_records + video_records
    if (
        author_best["delta_vs_baseline_valid"] >= FOLLOWUP_DELTA
        and video_best["delta_vs_baseline_valid"] >= FOLLOWUP_DELTA
    ):
        added, removed = merge_modifications([author_best, video_best])
        popularity_records.append(
            run_candidate(
                "author_video_qualifying_union",
                "popularity",
                added=added,
                removed=removed,
            )
        )
    popularity_best = max(popularity_records, key=lambda record: record["valid"]["primary"])
    results["stages"]["popularity"] = [record["name"] for record in popularity_records]
    results["stages"]["popularity_best"] = popularity_best["name"]

    print("\n=== Stage 4: predeclared pairwise crosses ===", flush=True)
    cross_features = (
        "decay_rate_per_duration",
        "decay_rate_x_log_activity",
        "activity_tier_x_tab",
    )
    cross_records = [
        run_candidate(f"cross_{feature}", "cross", added=(feature,))
        for feature in cross_features
    ]
    qualifying_crosses = [
        record
        for record in cross_records
        if record["delta_vs_baseline_valid"] >= FOLLOWUP_DELTA
    ]
    if len(qualifying_crosses) >= 2:
        added, removed = merge_modifications(qualifying_crosses)
        cross_records.append(
            run_candidate(
                "cross_qualifying_union",
                "cross",
                added=added,
                removed=removed,
            )
        )
    cross_best = max(cross_records, key=lambda record: record["valid"]["primary"])
    results["stages"]["cross"] = [record["name"] for record in cross_records]
    results["stages"]["cross_best"] = cross_best["name"]

    print("\n=== Stage 5: validation-qualified family combination ===", flush=True)
    family_bests = [user_best, tab_best, popularity_best, cross_best]
    qualifying_family_bests = [
        record
        for record in family_bests
        if record["delta_vs_baseline_valid"] >= FOLLOWUP_DELTA
    ]
    combination_records = []
    if len(qualifying_family_bests) >= 2:
        added, removed = merge_modifications(qualifying_family_bests)
        combination_records.append(
            run_candidate(
                "qualified_family_union",
                "final_combination",
                added=added,
                removed=removed,
            )
        )
    results["stages"]["qualified_family_bests"] = [
        record["name"] for record in qualifying_family_bests
    ]
    results["stages"]["final_combinations"] = [
        record["name"] for record in combination_records
    ]

    assert best_experimental_record is not None
    assert best_experimental_model is not None and best_experimental_scores is not None
    selected = best_experimental_record
    selected_delta = float(selected["valid"]["primary"] - baseline_primary)
    results["selected_on_validation"] = selected
    results["selected_delta_vs_baseline_valid"] = selected_delta
    print(
        f"\nselected on validation: {selected['name']} "
        f"valid={selected['valid']['primary']:.5f} delta={selected_delta:+.5f}",
        flush=True,
    )

    rng = np.random.default_rng(0)
    constant_metrics = evaluate(users["valid"], y["valid"], np.zeros(len(y["valid"])))
    random_metrics = evaluate(
        users["valid"], y["valid"], rng.uniform(size=len(y["valid"]))
    )
    results["tie_diagnostic"] = {
        "constant_valid": metric_dict(constant_metrics),
        "random_valid": metric_dict(random_metrics),
        "selected": tie_stats(best_experimental_scores, users["valid"]),
    }
    print(
        "tie diagnostic: "
        f"constant={constant_metrics['primary']:.5f} random={random_metrics['primary']:.5f} "
        f"selected_unique_fraction={results['tie_diagnostic']['selected']['mean_per_user_unique_fraction']:.4f}",
        flush=True,
    )

    # Paired five-seed confirmation is triggered only by a >=0.001 seed-0
    # gain. Both baseline and candidate are refit at each seed, so the delta
    # cannot be attributed to seed-specific LightGBM behavior.
    seed_confirmation = []
    promotion_confirmed = False
    if selected_delta >= PROMOTION_DELTA:
        print("\n=== paired five-seed validation confirmation ===", flush=True)
        seed_confirmation.append(
            {
                "seed": 0,
                "baseline_valid": baseline["valid"],
                "candidate_valid": selected["valid"],
                "delta": selected_delta,
            }
        )
        selected_columns = selected["config"]["columns"]
        for seed in SEEDS[1:]:
            baseline_model, baseline_scores, baseline_metrics, _ = fit_columns(
                frames,
                y,
                users,
                train_order,
                train_groups,
                valid_order,
                valid_groups,
                base_columns,
                seed,
            )
            candidate_model, candidate_scores, candidate_metrics, _ = fit_columns(
                frames,
                y,
                users,
                train_order,
                train_groups,
                valid_order,
                valid_groups,
                selected_columns,
                seed,
            )
            delta = float(candidate_metrics["primary"] - baseline_metrics["primary"])
            seed_confirmation.append(
                {
                    "seed": seed,
                    "baseline_valid": metric_dict(baseline_metrics),
                    "candidate_valid": metric_dict(candidate_metrics),
                    "delta": delta,
                }
            )
            print(
                f"  seed={seed} baseline={baseline_metrics['primary']:.5f} "
                f"candidate={candidate_metrics['primary']:.5f} delta={delta:+.5f}",
                flush=True,
            )
            del baseline_model, baseline_scores, candidate_model, candidate_scores
            gc.collect()
        deltas = np.asarray([record["delta"] for record in seed_confirmation])
        promotion_confirmed = bool(np.all(deltas >= PROMOTION_DELTA))
        results["seed_confirmation_summary"] = {
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "min_delta": float(np.min(deltas)),
            "all_seeds_at_least_0.001": promotion_confirmed,
        }
    else:
        results["seed_confirmation_summary"] = {
            "not_run": True,
            "reason": "seed-0 validation gain was below +0.001 promotion threshold",
        }
    results["seed_confirmation"] = seed_confirmation
    results["promotion_confirmed"] = promotion_confirmed
    write_results(results)

    # Every feature and follow-up decision is frozen above. This is the first
    # and only test-score computation/evaluation in the 6b experiment.
    print("\n=== FINAL TEST EVALUATION (selection frozen) ===", flush=True)
    selected_columns = selected["config"]["columns"]
    test_scores = best_experimental_model.predict(frames["test"][selected_columns])
    test_metrics = evaluate(users["test"], y["test"], test_scores)
    results["selected_on_validation"]["test"] = metric_dict(test_metrics)
    results["selected_delta_vs_baseline_test"] = float(
        test_metrics["primary"] - BASELINE_REFERENCE_TEST
    )
    results["verdict"] = "PROMOTE" if promotion_confirmed else "REJECT"
    results["test_evaluation"] = {
        "performed": True,
        "timing": "after feature selection, diagnostics, and seed verification were frozen",
    }
    write_results(results)
    print(
        f"selected candidate: valid={selected['valid']['primary']:.5f} "
        f"test={test_metrics['primary']:.5f} verdict={results['verdict']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()

