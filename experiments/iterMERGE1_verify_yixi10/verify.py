"""Independent from-scratch reproduction of yixi's iterYIXI10 claimed result.

Claimed (experiments/iterYIXI10_video_metadata/RESULT.md, seed 0, the number
reported in ensemble_results.json's "selected_on_validation" / components_test):
  10% FM / 52% LightGBM / 38% XGBoost, within-user-percentile blend
  valid primary = 0.69943440   test primary = 0.68432260

This script rebuilds her exact final feature sets (chained iterYIXI9 base
features + the iterYIXI10 leakage-safe `meta_upload_type` addition), retrains
all three components from scratch at seed 0 (FM ensemble uses her fixed
five-seed protocol, unchanged since iter38/iterYIXI5), applies her stated
blend weights, and evaluates valid + test primary with the project's
authoritative evaluate.py.

Feature construction reuses yixi's own features.py chain via importlib
(iterYIXI10 -> iterYIXI9 -> iterYIXI6 -> iter63/iterYIXI2), because that code
recomputes features directly from the raw CSVs -- it does not load any
frozen *prediction* artifact. Model training (LightGBM, XGBoost, FM) is done
independently in this file, not by calling her phase_lgb.py/phase_xgb.py/
ensemble.py. Her own feature-cache pickle files (regenerable, not committed
results) may be created as a side effect in her experiment directories the
first time this runs; run_and_clean.sh removes any such new files afterward
so her directories are left byte-for-byte as found.
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402

YIXI10_DIR = os.path.join(REPO_ROOT, "experiments", "iterYIXI10_video_metadata")
YIXI5_RESULTS_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "results.json"
)
MAKE_SUBMISSION_PATH = os.path.join(REPO_ROOT, "make_submission.py")

FM_SEEDS = (0, 1, 2, 3, 4)   # unchanged 5-seed FM ensemble protocol (iter38 -> YIXI5 -> YIXI10)
TRAIN_SEED = 0                # the LGB/XGB seed behind her headline 0.69943440 / 0.68432260

CAT_COLS = ["user_id", "video_id", "author_id", "tab", "last1"]
LGB_REFERENCE_COLUMNS = CAT_COLS + [
    "duration_ms",
    "decay_rate_5",
    "decay_act_5",
    "lastk_rate",
    "gap",
    "decay_tab_rate_3",
    "hist_watch_decay_mean_5",
]
LGB_CANDIDATE_COLUMNS = LGB_REFERENCE_COLUMNS + ["meta_upload_type"]
XGB_COLUMNS = CAT_COLS + [
    "duration_ms",
    "decay_tab_3",
    "lastk_rate",
    "gap",
    "decay_rate_5",
    "decay_act_5",
]

LGB_CONFIG = dict(
    objective="lambdarank",
    lambdarank_truncation_level=50,
    sigmoid=2.0,
    metric="ndcg",
    eval_at=[5],
    num_leaves=2,
    learning_rate=0.10,
    n_estimators=500,
    min_child_samples=200,
    reg_lambda=1.0,
    verbosity=-1,
    n_jobs=-1,
    linear_tree=True,
)

WEIGHTS = {"fm": 0.10, "lgb": 0.52, "xgb": 0.38}

CLAIMED_VALID = 0.69943440
CLAIMED_TEST = 0.68432260
TOLERANCE = 1e-3


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_user_order(user_ids):
    values = np.asarray(user_ids)
    order = np.argsort(values, kind="stable")
    groups = np.unique(values[order], return_counts=True)[1]
    return order, groups


def xgb_config():
    payload = json.load(open(YIXI5_RESULTS_PATH, encoding="utf-8"))
    return payload["selected_on_validation"]["config"]


def fit_lgb(frames, y, users, columns, seed):
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])
    Xtr = frames["train"][columns].iloc[train_order].reset_index(drop=True)
    Xva = frames["valid"][columns].iloc[valid_order].reset_index(drop=True)
    model = lgb.LGBMRanker(**LGB_CONFIG, random_state=seed)
    model.fit(
        Xtr,
        y["train"][train_order],
        group=train_groups,
        eval_set=[(Xva, y["valid"][valid_order])],
        eval_group=[valid_groups],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def fit_xgb(frames, y, users, columns, seed):
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])
    Xtr = frames["train"][columns].iloc[train_order].reset_index(drop=True)
    Xva = frames["valid"][columns].iloc[valid_order].reset_index(drop=True)
    model = xgb.XGBRanker(**xgb_config(), random_state=seed, n_jobs=-1, verbosity=0)
    model.fit(
        Xtr,
        y["train"][train_order],
        group=train_groups,
        eval_set=[(Xva, y["valid"][valid_order])],
        eval_group=[valid_groups],
        verbose=False,
    )
    return model


def within_user_percentile(scores, user_ids):
    s = pd.Series(np.asarray(scores, dtype=np.float64), copy=False)
    u = pd.Series(np.asarray(user_ids), copy=False)
    ranked = s.groupby(u, sort=False).rank(method="average", pct=True)
    values = ranked.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise AssertionError("non-finite within-user percentile")
    return values


def main():
    results = {"experiment": "iterMERGE1_verify_yixi10"}

    print("=== [1/4] rebuilding YIXI10 feature frames from raw CSVs ===", flush=True)
    features_mod = load_module(
        os.path.join(YIXI10_DIR, "features.py"), "verify_yixi10_features"
    )
    frames, y, users, _meta = features_mod.load_frames()

    missing = [c for c in LGB_CANDIDATE_COLUMNS if c not in frames["train"].columns]
    if missing:
        raise AssertionError(f"missing candidate LGB columns: {missing}")
    print(
        f"  train/valid/test rows = {len(frames['train'])}/{len(frames['valid'])}/{len(frames['test'])}",
        flush=True,
    )

    print("=== [2/4] training LightGBM (reference + candidate) and XGBoost at seed 0 ===", flush=True)
    lgb_ref_model = fit_lgb(frames, y, users, LGB_REFERENCE_COLUMNS, TRAIN_SEED)
    lgb_cand_model = fit_lgb(frames, y, users, LGB_CANDIDATE_COLUMNS, TRAIN_SEED)
    xgb_model = fit_xgb(frames, y, users, XGB_COLUMNS, TRAIN_SEED)

    lgb_ref_valid = lgb_ref_model.predict(frames["valid"][LGB_REFERENCE_COLUMNS])
    lgb_cand_valid = lgb_cand_model.predict(frames["valid"][LGB_CANDIDATE_COLUMNS])
    xgb_valid = xgb_model.predict(frames["valid"][XGB_COLUMNS])
    lgb_ref_test = lgb_ref_model.predict(frames["test"][LGB_REFERENCE_COLUMNS])
    lgb_cand_test = lgb_cand_model.predict(frames["test"][LGB_CANDIDATE_COLUMNS])
    xgb_test = xgb_model.predict(frames["test"][XGB_COLUMNS])

    results["components_standalone"] = {
        "lgb_reference_valid": evaluate(users["valid"], y["valid"], lgb_ref_valid),
        "lgb_candidate_valid": evaluate(users["valid"], y["valid"], lgb_cand_valid),
        "xgb_valid": evaluate(users["valid"], y["valid"], xgb_valid),
        "lgb_reference_test": evaluate(users["test"], y["test"], lgb_ref_test),
        "lgb_candidate_test": evaluate(users["test"], y["test"], lgb_cand_test),
        "xgb_test": evaluate(users["test"], y["test"], xgb_test),
    }
    print(
        f"  candidate LGB valid primary = {results['components_standalone']['lgb_candidate_valid']['primary']:.8f}"
        f"  (her claim: 0.68834144)",
        flush=True,
    )
    print(
        f"  xgb valid primary           = {results['components_standalone']['xgb_valid']['primary']:.8f}"
        f"  (her claim: 0.66755420)",
        flush=True,
    )

    print("=== [3/4] training FM 5-seed ensemble via root make_submission.py (train_one_fm) ===", flush=True)
    submission = load_module(MAKE_SUBMISSION_PATH, "verify_yixi10_submission")
    splits = submission.load_ext(
        os.path.join(REPO_ROOT, "KuaiRand-Pure", "data"),
        halflives=submission.HALFLIVES,
        tab_halflives=submission.TAB_HALFLIVES,
    )
    encoded, dim = submission.encode_ext(
        splits,
        feature_set=submission.FEATURES,
        halflives=submission.HALFLIVES,
        tab_halflives=submission.TAB_HALFLIVES,
        alpha=0.5,
        n_buckets=20,
    )
    Xtr, ytr, utr = encoded["train"]
    Xva, yva, uva = encoded["valid"]
    Xte, yte, ute = encoded["test"]
    if not np.array_equal(np.asarray(yva), y["valid"]):
        raise AssertionError("FM/native valid label mismatch")
    if not np.array_equal(np.asarray(uva), np.asarray(users["valid"])):
        raise AssertionError("FM/native valid user mismatch")
    if not np.array_equal(np.asarray(yte), y["test"]):
        raise AssertionError("FM/native test label mismatch")
    if not np.array_equal(np.asarray(ute), np.asarray(users["test"])):
        raise AssertionError("FM/native test user mismatch")

    fm_valid_seed_scores, fm_test_seed_scores = [], []
    for seed in FM_SEEDS:
        print(f"  fitting FM seed={seed}", flush=True)
        model = submission.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits["train"], dim, seed)
        fm_valid_seed_scores.append(submission.sigmoid(model.predict(Xva)))
        fm_test_seed_scores.append(submission.sigmoid(model.predict(Xte)))
    fm_valid = np.mean(np.stack(fm_valid_seed_scores), axis=0)
    fm_test = np.mean(np.stack(fm_test_seed_scores), axis=0)
    results["components_standalone"]["fm_valid"] = evaluate(users["valid"], y["valid"], fm_valid)
    results["components_standalone"]["fm_test"] = evaluate(users["test"], y["test"], fm_test)
    print(
        f"  fm ensemble valid primary = {results['components_standalone']['fm_valid']['primary']:.8f}"
        f"  (her claim: 0.63987792)",
        flush=True,
    )

    print("=== [4/4] within-user-percentile blend at 10% FM / 52% LGB / 38% XGB ===", flush=True)
    valid_components = {
        "fm": within_user_percentile(fm_valid, users["valid"]),
        "lgb": within_user_percentile(lgb_cand_valid, users["valid"]),
        "xgb": within_user_percentile(xgb_valid, users["valid"]),
    }
    test_components = {
        "fm": within_user_percentile(fm_test, users["test"]),
        "lgb": within_user_percentile(lgb_cand_test, users["test"]),
        "xgb": within_user_percentile(xgb_test, users["test"]),
    }
    blended_valid = sum(WEIGHTS[k] * valid_components[k] for k in WEIGHTS)
    blended_test = sum(WEIGHTS[k] * test_components[k] for k in WEIGHTS)
    valid_metrics = evaluate(users["valid"], y["valid"], blended_valid)
    test_metrics = evaluate(users["test"], y["test"], blended_test)

    results["weights"] = WEIGHTS
    results["blend_valid"] = valid_metrics
    results["blend_test"] = test_metrics
    results["claimed"] = {"valid_primary": CLAIMED_VALID, "test_primary": CLAIMED_TEST}
    results["delta"] = {
        "valid_primary": valid_metrics["primary"] - CLAIMED_VALID,
        "test_primary": test_metrics["primary"] - CLAIMED_TEST,
    }
    results["reproduced"] = {
        "valid": bool(abs(results["delta"]["valid_primary"]) <= TOLERANCE),
        "test": bool(abs(results["delta"]["test_primary"]) <= TOLERANCE),
        "tolerance": TOLERANCE,
    }
    results["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": lgb.__version__,
        "xgboost": xgb.__version__,
    }

    out_path = os.path.join(THIS_DIR, "verify_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(jsonable(results), f, indent=2, sort_keys=True)
        f.write("\n")

    print("\n=== RESULT ===")
    print(
        f"valid primary: ours={valid_metrics['primary']:.8f}  claimed={CLAIMED_VALID:.8f}  "
        f"delta={results['delta']['valid_primary']:+.8f}"
    )
    print(
        f"test  primary: ours={test_metrics['primary']:.8f}  claimed={CLAIMED_TEST:.8f}  "
        f"delta={results['delta']['test_primary']:+.8f}"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
