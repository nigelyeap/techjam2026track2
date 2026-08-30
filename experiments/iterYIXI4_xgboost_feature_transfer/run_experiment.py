"""Section 6d: transfer the unchanged 6b features to the fixed 6a XGBoost.

Feature construction is imported directly from iterYIXI2_feature_depth so
this experiment changes only the learner. Model construction is imported
directly from iterYIXI1_xgboost_native and fixed to its validation winner:
rank:ndcg, max_depth=1.

Run from the repository root:

    python3 experiments/iterYIXI4_xgboost_feature_transfer/run_experiment.py
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
from typing import Any

import numpy as np
import xgboost as xgb


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
YIXI1_RUNNER_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI1_xgboost_native", "run_experiment.py"
)
YIXI2_FEATURES_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI2_feature_depth", "features.py"
)
RESULTS_PATH = os.path.join(THIS_DIR, "results.json")

sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402


REFERENCE_VALID = 0.6586387157440186
REFERENCE_TEST = 0.6451212167739868
PRELIMINARY_DELTA = 0.0003
PROMOTION_DELTA = 0.001
SEEDS = (0, 1, 2, 3, 4)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


yixi1 = load_module(YIXI1_RUNNER_PATH, "yixi1_runner_for_yixi4")
yixi2_features = load_module(YIXI2_FEATURES_PATH, "yixi2_features_for_yixi4")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def metric_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: jsonable(value) for key, value in metrics.items()}


def write_results(results: dict[str, Any]) -> None:
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(jsonable(results), f, indent=2, sort_keys=True)
        f.write("\n")


def stable_user_order(user_ids):
    user_ids = np.asarray(user_ids)
    order = np.argsort(user_ids, kind="stable")
    groups = np.unique(user_ids[order], return_counts=True)[1]
    return order, groups


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
    """Fit the exact promoted 6a standalone XGBoost configuration."""
    Xtr = frames["train"][columns].iloc[train_order].reset_index(drop=True)
    ytr = y["train"][train_order]
    Xva = frames["valid"][columns].iloc[valid_order].reset_index(drop=True)
    yva = y["valid"][valid_order]
    model = yixi1.make_xgb_ranker("rank:ndcg", 1, seed)
    model.fit(
        Xtr,
        ytr,
        group=train_groups,
        eval_set=[(Xva, yva)],
        eval_group=[valid_groups],
        verbose=False,
    )
    valid_scores = model.predict(frames["valid"][columns])
    valid_metrics = evaluate(users["valid"], y["valid"], valid_scores)

    raw_gain = model.get_booster().get_score(importance_type="gain")
    gain_total = float(sum(raw_gain.values()))
    gain_fraction = {
        column: (float(raw_gain.get(column, 0.0)) / gain_total if gain_total > 0 else 0.0)
        for column in columns
    }
    del Xtr, Xva
    gc.collect()
    return model, valid_scores, valid_metrics, gain_fraction


def config_columns(base_columns: list[str], added=(), removed=()) -> list[str]:
    removed_set = set(removed)
    columns = [column for column in base_columns if column not in removed_set]
    for column in added:
        if column not in columns:
            columns.append(column)
    return columns


def merge_modifications(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    added: list[str] = []
    removed: list[str] = []
    for record in records:
        for column in record["config"]["removed"]:
            if column not in removed:
                removed.append(column)
        for column in record["config"]["added"]:
            if column not in added:
                added.append(column)
    return added, removed


def tie_stats(scores, user_ids) -> dict[str, Any]:
    scores = np.asarray(scores)
    user_ids = np.asarray(user_ids)
    fractions = []
    for user_id in np.unique(user_ids):
        user_scores = scores[user_ids == user_id]
        fractions.append(len(np.unique(user_scores)) / len(user_scores))
    return {
        "unique_scores_overall": int(len(np.unique(scores))),
        "rows": int(len(scores)),
        "mean_per_user_unique_fraction": float(np.mean(fractions)),
    }


def main() -> None:
    print("=== loading unchanged 6b features and causal checks ===", flush=True)
    frames, y, users, feature_metadata = yixi2_features.prepare_feature_frames(DATA_DIR)
    if not feature_metadata["causality"]["passed"]:
        raise AssertionError("imported 6b causality checks failed")
    print(
        "6b causal gate PASSED: "
        f"max_abs_error={feature_metadata['causality']['global_max_abs_error']:.3e}, "
        f"same_date_rows_excluded="
        f"{feature_metadata['causality']['same_date_matching_rows_excluded_across_checks']}",
        flush=True,
    )

    base_columns = list(feature_metadata["base_columns"])
    expected_base = list(yixi2_features.iter44.CAT_COLS + yixi2_features.iter44.NUM_COLS)
    assert base_columns == expected_base
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])

    results: dict[str, Any] = {
        "experiment": "iterYIXI4_xgboost_feature_transfer",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
        },
        "provenance": {
            "model_builder": os.path.relpath(YIXI1_RUNNER_PATH, REPO_ROOT),
            "model_builder_sha256": file_sha256(YIXI1_RUNNER_PATH),
            "feature_builder": os.path.relpath(YIXI2_FEATURES_PATH, REPO_ROOT),
            "feature_builder_sha256": file_sha256(YIXI2_FEATURES_PATH),
            "feature_definitions_modified": False,
        },
        "selection_policy": {
            "selector": "official validation primary only",
            "preliminary_delta": PRELIMINARY_DELTA,
            "promotion_delta": PROMOTION_DELTA,
            "test_access": "after family selection, confirmation, composition, and diagnostics",
        },
        "fixed_model": {
            "objective": "rank:ndcg",
            "max_depth": 1,
            "eval_metric": "ndcg@5-",
            "n_estimators": yixi1.N_ESTIMATORS,
            "learning_rate": yixi1.LEARNING_RATE,
            "early_stopping_rounds": yixi1.EARLY_STOPPING_ROUNDS,
            "min_child_weight": 1.0,
            "reg_lambda": yixi1.REG_LAMBDA,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "max_bin": 256,
            "enable_categorical": True,
            "max_cat_to_onehot": 4,
        },
        "feature_metadata": feature_metadata,
        "candidates": [],
        "families": {},
        "confirmations": {},
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
            "best_iteration": int(model.best_iteration),
            "best_internal_ndcg_at_5_minus": float(model.best_score),
            "valid": metric_dict(metrics),
            "feature_gain_fraction": gain_fraction,
            "added_feature_gain_fraction": {
                column: gain_fraction.get(column, 0.0) for column in added
            },
        }
        if baseline_record is not None:
            record["delta_vs_reference_valid"] = float(
                metrics["primary"] - baseline_record["valid"]["primary"]
            )
        results["candidates"].append(record)
        records_by_signature[signature] = record
        print(
            f"    valid={metrics['primary']:.5f} "
            f"GAUC={metrics['GAUC']:.5f} nDCG@5={metrics['nDCG@5']:.5f} "
            f"best_iter={model.best_iteration}",
            flush=True,
        )

        if family == "reference":
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

    print("\n=== Stage 0: exact 6a XGBoost reference ===", flush=True)
    reference = run_candidate("yixi1_xgb_reference", "reference")
    reference_primary = reference["valid"]["primary"]
    if abs(reference_primary - REFERENCE_VALID) > 1e-8:
        raise AssertionError(
            f"6a harness drift: reproduced {reference_primary}, expected {REFERENCE_VALID}"
        )
    print(f"  6a validation reproduction PASSED ({reference_primary:.8f})", flush=True)

    print("\n=== Family 1: additional decay timescales ===", flush=True)
    timescale_records = []
    user_parallel_records = []
    user_records = []
    for halflife in yixi2_features.USER_HALFLIVES:
        suffix = yixi2_features.fmt_h(halflife)
        pair = (f"decay_rate_{suffix}", f"decay_act_{suffix}")
        user_records.append(
            run_candidate(
                f"user_h{suffix}_replace",
                "additional_timescales",
                added=pair,
                removed=("decay_rate_2.5", "decay_act_2.5"),
            )
        )
        parallel = run_candidate(
            f"user_h{suffix}_parallel", "additional_timescales", added=pair
        )
        user_records.append(parallel)
        user_parallel_records.append(parallel)
    timescale_records.extend(user_records)

    qualifying_user_parallel = [
        record
        for record in user_parallel_records
        if record["delta_vs_reference_valid"] >= PRELIMINARY_DELTA
    ]
    if len(qualifying_user_parallel) >= 2:
        added, removed = merge_modifications(qualifying_user_parallel)
        timescale_records.append(
            run_candidate(
                "user_parallel_preliminary_union",
                "additional_timescales",
                added=added,
                removed=removed,
            )
        )

    tab_parallel_records = []
    tab_records = []
    for halflife in yixi2_features.TAB_HALFLIVES:
        suffix = yixi2_features.fmt_h(halflife)
        feature = (f"decay_tab_{suffix}",)
        tab_records.append(
            run_candidate(
                f"tab_h{suffix}_replace",
                "additional_timescales",
                added=feature,
                removed=("decay_tab_3",),
            )
        )
        parallel = run_candidate(
            f"tab_h{suffix}_parallel", "additional_timescales", added=feature
        )
        tab_records.append(parallel)
        tab_parallel_records.append(parallel)
    timescale_records.extend(tab_records)

    qualifying_tab_parallel = [
        record
        for record in tab_parallel_records
        if record["delta_vs_reference_valid"] >= PRELIMINARY_DELTA
    ]
    if len(qualifying_tab_parallel) >= 2:
        added, removed = merge_modifications(qualifying_tab_parallel)
        timescale_records.append(
            run_candidate(
                "tab_parallel_preliminary_union",
                "additional_timescales",
                added=added,
                removed=removed,
            )
        )

    best_user = max(user_records, key=lambda record: record["valid"]["primary"])
    best_tab = max(tab_records, key=lambda record: record["valid"]["primary"])
    if (
        best_user["delta_vs_reference_valid"] >= PRELIMINARY_DELTA
        and best_tab["delta_vs_reference_valid"] >= PRELIMINARY_DELTA
    ):
        added, removed = merge_modifications([best_user, best_tab])
        timescale_records.append(
            run_candidate(
                "user_tab_timescale_preliminary_union",
                "additional_timescales",
                added=added,
                removed=removed,
            )
        )
    timescale_best = max(timescale_records, key=lambda record: record["valid"]["primary"])
    results["families"]["additional_timescales"] = {
        "candidates": [record["name"] for record in timescale_records],
        "winner": timescale_best["name"],
        "preliminary_pass": (
            timescale_best["delta_vs_reference_valid"] >= PRELIMINARY_DELTA
        ),
    }

    print("\n=== Family 2: author/video popularity decay ===", flush=True)
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
        run_candidate(name, "popularity_decay", added=added) for name, added in author_specs
    ]
    video_records = [
        run_candidate(name, "popularity_decay", added=added) for name, added in video_specs
    ]
    popularity_records = author_records + video_records
    best_author = max(author_records, key=lambda record: record["valid"]["primary"])
    best_video = max(video_records, key=lambda record: record["valid"]["primary"])
    if (
        best_author["delta_vs_reference_valid"] >= PRELIMINARY_DELTA
        and best_video["delta_vs_reference_valid"] >= PRELIMINARY_DELTA
    ):
        added, removed = merge_modifications([best_author, best_video])
        popularity_records.append(
            run_candidate(
                "author_video_preliminary_union",
                "popularity_decay",
                added=added,
                removed=removed,
            )
        )
    popularity_best = max(popularity_records, key=lambda record: record["valid"]["primary"])
    results["families"]["popularity_decay"] = {
        "candidates": [record["name"] for record in popularity_records],
        "winner": popularity_best["name"],
        "preliminary_pass": (
            popularity_best["delta_vs_reference_valid"] >= PRELIMINARY_DELTA
        ),
    }

    print("\n=== Family 3: unchanged 6b cross-features ===", flush=True)
    cross_names = (
        "decay_rate_per_duration",
        "decay_rate_x_log_activity",
        "activity_tier_x_tab",
    )
    cross_records = [
        run_candidate(f"cross_{column}", "cross_features", added=(column,))
        for column in cross_names
    ]
    qualifying_crosses = [
        record
        for record in cross_records
        if record["delta_vs_reference_valid"] >= PRELIMINARY_DELTA
    ]
    if len(qualifying_crosses) >= 2:
        added, removed = merge_modifications(qualifying_crosses)
        cross_records.append(
            run_candidate(
                "cross_preliminary_union",
                "cross_features",
                added=added,
                removed=removed,
            )
        )
    cross_best = max(cross_records, key=lambda record: record["valid"]["primary"])
    results["families"]["cross_features"] = {
        "candidates": [record["name"] for record in cross_records],
        "winner": cross_best["name"],
        "preliminary_pass": cross_best["delta_vs_reference_valid"] >= PRELIMINARY_DELTA,
    }

    family_winners = {
        "additional_timescales": timescale_best,
        "popularity_decay": popularity_best,
        "cross_features": cross_best,
    }
    preliminary_winners = {
        family: record
        for family, record in family_winners.items()
        if record["delta_vs_reference_valid"] >= PRELIMINARY_DELTA
    }
    results["preliminary_family_winners"] = {
        family: record["name"] for family, record in preliminary_winners.items()
    }
    write_results(results)

    # Build paired baseline metrics once. With the exact 6a full-sampling
    # configuration these should be identical across seeds, but the paired
    # check is explicit and reusable for each preliminary-positive family.
    baseline_by_seed = {0: reference["valid"]}
    if preliminary_winners:
        print("\n=== paired baseline seeds for confirmation ===", flush=True)
        for seed in SEEDS[1:]:
            model, scores, metrics, _ = fit_columns(
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
            baseline_by_seed[seed] = metric_dict(metrics)
            print(f"  seed={seed} reference={metrics['primary']:.5f}", flush=True)
            del model, scores
            gc.collect()

    def confirm_record(record: dict[str, Any], label: str) -> dict[str, Any]:
        print(f"\n=== five-seed confirmation: {label} ===", flush=True)
        rows = [
            {
                "seed": 0,
                "reference_valid": reference["valid"],
                "candidate_valid": record["valid"],
                "delta": record["delta_vs_reference_valid"],
            }
        ]
        for seed in SEEDS[1:]:
            model, scores, metrics, _ = fit_columns(
                frames,
                y,
                users,
                train_order,
                train_groups,
                valid_order,
                valid_groups,
                record["config"]["columns"],
                seed,
            )
            delta = float(metrics["primary"] - baseline_by_seed[seed]["primary"])
            rows.append(
                {
                    "seed": seed,
                    "reference_valid": baseline_by_seed[seed],
                    "candidate_valid": metric_dict(metrics),
                    "delta": delta,
                }
            )
            print(
                f"  seed={seed} candidate={metrics['primary']:.5f} delta={delta:+.5f}",
                flush=True,
            )
            del model, scores
            gc.collect()
        deltas = np.asarray([row["delta"] for row in rows], dtype=np.float64)
        summary = {
            "candidate": record["name"],
            "rows": rows,
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "min_delta": float(np.min(deltas)),
            "all_seeds_at_least_0.001": bool(np.all(deltas >= PROMOTION_DELTA)),
            "confirmed": bool(
                np.mean(deltas) >= PROMOTION_DELTA
                and np.all(deltas >= PROMOTION_DELTA)
            ),
        }
        return summary

    confirmed_families: dict[str, dict[str, Any]] = {}
    for family, record in preliminary_winners.items():
        confirmation = confirm_record(record, family)
        results["confirmations"][family] = confirmation
        if confirmation["confirmed"]:
            confirmed_families[family] = record
        write_results(results)

    # Cross-family composition is allowed only after two independent families
    # have each cleared the full five-seed >=0.001 criterion.
    composition_record = None
    composition_confirmation = None
    if len(confirmed_families) >= 2:
        print("\n=== optional composition of independently confirmed families ===", flush=True)
        added, removed = merge_modifications(list(confirmed_families.values()))
        composition_record = run_candidate(
            "confirmed_family_composition",
            "confirmed_composition",
            added=added,
            removed=removed,
        )
        if composition_record["delta_vs_reference_valid"] >= PRELIMINARY_DELTA:
            composition_confirmation = confirm_record(
                composition_record, "confirmed_family_composition"
            )
            results["confirmations"]["confirmed_family_composition"] = (
                composition_confirmation
            )
    results["confirmed_families"] = list(confirmed_families)
    results["composition_attempted"] = composition_record is not None

    confirmed_records = list(confirmed_families.values())
    if composition_record is not None and composition_confirmation is not None:
        if composition_confirmation["confirmed"]:
            confirmed_records.append(composition_record)

    assert best_experimental_record is not None
    assert best_experimental_model is not None and best_experimental_scores is not None
    if confirmed_records:
        selected = max(confirmed_records, key=lambda record: record["valid"]["primary"])
    else:
        selected = best_experimental_record

    # The retained model/scores correspond to the best seed-0 experimental
    # candidate. If confirmation selects a different robust candidate, refit
    # its deterministic seed-0 model before diagnostics and final test.
    if selected["name"] == best_experimental_record["name"]:
        selected_model = best_experimental_model
        selected_scores = best_experimental_scores
    else:
        selected_model, selected_scores, selected_metrics, _ = fit_columns(
            frames,
            y,
            users,
            train_order,
            train_groups,
            valid_order,
            valid_groups,
            selected["config"]["columns"],
            seed=0,
        )
        if abs(selected_metrics["primary"] - selected["valid"]["primary"]) > 1e-10:
            raise AssertionError("deterministic selected-model refit drifted")

    results["selected_on_validation"] = selected
    results["selected_reason"] = (
        "best confirmed candidate" if confirmed_records else "best seed-0 candidate; no family confirmed"
    )
    rng = np.random.default_rng(0)
    constant_metrics = evaluate(users["valid"], y["valid"], np.zeros(len(y["valid"])))
    random_metrics = evaluate(
        users["valid"], y["valid"], rng.uniform(size=len(y["valid"]))
    )
    results["tie_diagnostic"] = {
        "constant_valid": metric_dict(constant_metrics),
        "random_valid": metric_dict(random_metrics),
        "selected": tie_stats(selected_scores, users["valid"]),
    }
    print(
        f"\nselected={selected['name']} valid={selected['valid']['primary']:.5f} "
        f"delta={selected['delta_vs_reference_valid']:+.5f}",
        flush=True,
    )
    print(
        "tie diagnostic: "
        f"constant={constant_metrics['primary']:.5f} random={random_metrics['primary']:.5f} "
        f"unique_fraction="
        f"{results['tie_diagnostic']['selected']['mean_per_user_unique_fraction']:.4f}",
        flush=True,
    )
    write_results(results)

    # First and only test-score computation in 6d. All selection and
    # confirmation decisions above are frozen.
    print("\n=== FINAL TEST EVALUATION (selection frozen) ===", flush=True)
    selected_columns = selected["config"]["columns"]
    test_scores = selected_model.predict(frames["test"][selected_columns])
    test_metrics = evaluate(users["test"], y["test"], test_scores)
    results["selected_on_validation"]["test"] = metric_dict(test_metrics)
    results["selected_delta_vs_reference_test"] = float(
        test_metrics["primary"] - REFERENCE_TEST
    )
    results["verdict"] = "PROMOTE" if confirmed_families else "REJECT"
    results["test_evaluation"] = {
        "performed": True,
        "timing": "after family selection, five-seed confirmations, optional composition, and diagnostics",
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

