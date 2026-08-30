"""Section 6c, part 1: lightweight attention feature on fixed h5 XGBoost.

This runner owns only the standalone feature-utility comparison. It freezes
the 6a XGBoost hyperparameters and YIXI4's confirmed 5-day decay replacement,
selects attention scalars using validation primary only, and evaluates one
frozen attention candidate on test. Project-level blending is isolated in
``blend.py``.

Run from the repository root:

    python3 experiments/iterYIXI3_attention_pool_feature/run_experiment.py
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
FEATURES_PATH = os.path.join(THIS_DIR, "features.py")
RESULTS_PATH = os.path.join(THIS_DIR, "results.json")

sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402


H5_REFERENCE_VALID = 0.6664905548095703
H5_REFERENCE_TEST = 0.6519709825515747
CURRENT_PROJECT_VALID = 0.6700649857521057
CURRENT_PROJECT_TEST = 0.6564562320709229
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


yixi1 = load_module(YIXI1_RUNNER_PATH, "yixi1_runner_for_yixi3")
feature_builder = load_module(FEATURES_PATH, "yixi3_feature_builder")


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
        column: (float(raw_gain.get(column, 0.0)) / gain_total if gain_total else 0.0)
        for column in columns
    }
    del Xtr, Xva
    gc.collect()
    return model, valid_scores, valid_metrics, gain_fraction


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
    print("=== loading h5 native frames and causal attention features ===", flush=True)
    frames, y, users, feature_metadata = feature_builder.prepare_feature_frames(DATA_DIR)
    if not feature_metadata["causality"]["passed"]:
        raise AssertionError("feature causality gate failed")
    h5_columns = list(feature_metadata["h5_columns"])
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])

    results: dict[str, Any] = {
        "experiment": "iterYIXI3_attention_pool_feature_standalone",
        "scope": "attention feature utility on fixed YIXI4 h5 XGBoost component",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
        },
        "provenance": {
            "xgb_builder": os.path.relpath(YIXI1_RUNNER_PATH, REPO_ROOT),
            "xgb_builder_sha256": file_sha256(YIXI1_RUNNER_PATH),
            "feature_builder": os.path.relpath(FEATURES_PATH, REPO_ROOT),
            "feature_builder_sha256": file_sha256(FEATURES_PATH),
            "reused_yixi2_sha256": feature_metadata["identity"]["yixi2_sha256"],
            "reused_iter32_sha256": feature_metadata["identity"]["iter32_sha256"],
        },
        "project_context": {
            "yixi1_promoted_blend_valid": CURRENT_PROJECT_VALID,
            "yixi1_promoted_blend_test": CURRENT_PROJECT_TEST,
            "yixi2_verdict": "REJECT on native LightGBM after five-seed verification",
            "yixi4_h5_xgb_valid": H5_REFERENCE_VALID,
            "yixi4_h5_xgb_test": H5_REFERENCE_TEST,
            "promotion_note": (
                "standalone attention utility is compared with h5; project promotion "
                "must be established separately by blend.py against YIXI1"
            ),
        },
        "selection_policy": {
            "selector": "official validation primary only",
            "preliminary_delta": PRELIMINARY_DELTA,
            "promotion_delta": PROMOTION_DELTA,
            "test_access": "one selected attention candidate after all validation decisions",
        },
        "fixed_model": {
            "objective": "rank:ndcg",
            "max_depth": 1,
            "learning_rate": yixi1.LEARNING_RATE,
            "n_estimators": yixi1.N_ESTIMATORS,
            "early_stopping_rounds": yixi1.EARLY_STOPPING_ROUNDS,
            "feature_reference": "YIXI4 h5 replacement",
            "columns": h5_columns,
        },
        "feature_metadata": feature_metadata,
        "candidates": [],
    }
    write_results(results)

    retained_models: dict[str, Any] = {}
    retained_scores: dict[str, np.ndarray] = {}

    def run_candidate(name: str, family: str, columns: list[str]):
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
            "seed": 0,
            "columns": columns,
            "added": [column for column in columns if column not in h5_columns],
            "best_iteration": int(model.best_iteration),
            "best_internal_ndcg_at_5_minus": float(model.best_score),
            "valid": metric_dict(metrics),
            "feature_gain_fraction": gain_fraction,
        }
        if family != "reference":
            record["delta_vs_h5_valid"] = float(metrics["primary"] - H5_REFERENCE_VALID)
            record["added_feature_gain_fraction"] = {
                column: gain_fraction.get(column, 0.0) for column in record["added"]
            }
        results["candidates"].append(record)
        retained_models[name] = model
        retained_scores[name] = scores
        print(
            f"    valid={metrics['primary']:.8f} GAUC={metrics['GAUC']:.8f} "
            f"nDCG@5={metrics['nDCG@5']:.8f} best_iter={model.best_iteration}",
            flush=True,
        )
        write_results(results)
        return record

    print("\n=== Stage 0: exact YIXI4 h5 XGBoost harness ===", flush=True)
    reference = run_candidate("h5_xgb_reference", "reference", h5_columns)
    if abs(reference["valid"]["primary"] - H5_REFERENCE_VALID) > 1e-8:
        raise AssertionError(
            f"YIXI4 h5 harness drift: {reference['valid']['primary']} vs {H5_REFERENCE_VALID}"
        )
    print(f"  h5 validation reproduction PASSED ({H5_REFERENCE_VALID:.8f})", flush=True)

    print("\n=== Stage 1: independent attention-window candidates ===", flush=True)
    attention_records = []
    for window in feature_builder.ATTENTION_WINDOWS:
        column = f"attn_rate_{window}"
        attention_records.append(
            run_candidate(f"attention_k{window}", "attention", h5_columns + [column])
        )

    individual_preliminary = [
        record
        for record in attention_records
        if record["delta_vs_h5_valid"] >= PRELIMINARY_DELTA
    ]
    if len(individual_preliminary) == len(attention_records):
        attention_records.append(
            run_candidate(
                "attention_k20_k40_union",
                "attention",
                h5_columns + list(feature_metadata["attention_columns"]),
            )
        )

    winner = max(attention_records, key=lambda record: record["valid"]["primary"])
    results["attention_selection"] = {
        "winner": winner["name"],
        "winner_valid": winner["valid"],
        "delta_vs_h5_valid": winner["delta_vs_h5_valid"],
        "preliminary_pass": winner["delta_vs_h5_valid"] >= PRELIMINARY_DELTA,
    }

    # A non-attention long-window mean is diagnostic only and cannot replace
    # the preselected attention winner.
    control_columns = [
        column.replace("attn_rate_", "uniform_rate_") for column in winner["added"]
    ]
    control = run_candidate(
        f"uniform_control_for_{winner['name']}",
        "diagnostic_control",
        h5_columns + control_columns,
    )
    results["uniform_control"] = {
        "candidate": control["name"],
        "valid": control["valid"],
        "delta_vs_h5_valid": control["delta_vs_h5_valid"],
        "not_selection_eligible": True,
    }

    baseline_by_seed = {0: reference["valid"]}
    confirmation = None
    if winner["delta_vs_h5_valid"] >= PRELIMINARY_DELTA:
        print("\n=== paired five-seed attention confirmation ===", flush=True)
        rows = [
            {
                "seed": 0,
                "reference_valid": reference["valid"],
                "candidate_valid": winner["valid"],
                "delta": winner["delta_vs_h5_valid"],
            }
        ]
        for seed in SEEDS[1:]:
            base_model, base_scores, base_metrics, _ = fit_columns(
                frames,
                y,
                users,
                train_order,
                train_groups,
                valid_order,
                valid_groups,
                h5_columns,
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
                winner["columns"],
                seed,
            )
            baseline_by_seed[seed] = metric_dict(base_metrics)
            delta = float(candidate_metrics["primary"] - base_metrics["primary"])
            rows.append(
                {
                    "seed": seed,
                    "reference_valid": metric_dict(base_metrics),
                    "candidate_valid": metric_dict(candidate_metrics),
                    "delta": delta,
                }
            )
            print(
                f"  seed={seed} reference={base_metrics['primary']:.8f} "
                f"candidate={candidate_metrics['primary']:.8f} delta={delta:+.8f}",
                flush=True,
            )
            del base_model, base_scores, candidate_model, candidate_scores
            gc.collect()
        deltas = np.asarray([row["delta"] for row in rows], dtype=np.float64)
        confirmation = {
            "candidate": winner["name"],
            "rows": rows,
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "min_delta": float(np.min(deltas)),
            "confirmed": bool(
                np.mean(deltas) >= PROMOTION_DELTA
                and np.all(deltas >= PROMOTION_DELTA)
            ),
        }
    results["five_seed_confirmation"] = confirmation
    attention_confirmed = bool(confirmation and confirmation["confirmed"])
    results["blend_eligible"] = attention_confirmed
    results["standalone_feature_verdict"] = (
        "PROMOTE_ATTENTION_FOR_BLEND_TEST" if attention_confirmed else "REJECT_ATTENTION"
    )

    selected_model = retained_models[winner["name"]]
    selected_scores = retained_scores[winner["name"]]
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
    results["selected_on_validation"] = winner
    write_results(results)

    print("\n=== FINAL ATTENTION-CANDIDATE TEST (selection frozen) ===", flush=True)
    selected_test_scores = selected_model.predict(frames["test"][winner["columns"]])
    selected_test_metrics = evaluate(users["test"], y["test"], selected_test_scores)
    results["selected_on_validation"]["test"] = metric_dict(selected_test_metrics)
    results["selected_delta_vs_h5_test"] = float(
        selected_test_metrics["primary"] - H5_REFERENCE_TEST
    )
    results["test_evaluation"] = {
        "performed": True,
        "timing": "after attention selection, control, confirmation, and diagnostics",
        "reference_test_source": "published YIXI4 h5 result; not recomputed here",
    }
    write_results(results)
    print(
        f"selected={winner['name']} valid={winner['valid']['primary']:.8f} "
        f"test={selected_test_metrics['primary']:.8f} "
        f"verdict={results['standalone_feature_verdict']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
