"""iterMERGE5 confirmation pass.

Per coordinator instruction: the coarse/fine weight search already found a
PRELIMINARY point (fm=0.072, lgb=0.5184, xgb=0.3296, i63=0.08, valid
+0.00038314 vs. the 3-model reference) but its TEST score regressed
(-0.00066 vs reference), and the point was found via a 135-combo grid
search on valid only -- a real risk of overfitting to the search grid
rather than a genuine gain. This script checks robustness, not a new
sweep:

  1. Re-fits the fixed pipeline once (yixi frames, LGB/XGB seed 0, FM
     5-seed ensemble, iter63 rate_only at 5 seeds) -- same code paths as
     run.py, nothing new architecturally -- but this time keeps every
     per-seed iter63 raw score array (run.py only kept the sigmoid-mean
     ensemble and per-seed scalar metrics, not the raw per-seed arrays
     needed to blend seed-by-seed).
  2. At the best weight point from run.py's fine grid search, evaluates
     the 4-way blend using iter63's rate_only score from EACH of its 5
     individual seeds (not the sigmoid-mean ensemble) against the same
     fixed FM/LGB/XGB components -- does the valid gain hold up across
     all 5, or was the ensemble specifically what produced the borderline
     positive?
  3. Evaluates the 2nd and 3rd best fine-grid weight points (using the
     already-established sigmoid-mean iter63 ensemble) to check whether
     the valid gain is a plateau (real) or a narrow grid-cell spike
     (overfit to the search), and whether the test regression is specific
     to the single best point or shared by its neighbors.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402

YIXI10_DIR = os.path.join(REPO_ROOT, "experiments", "iterYIXI10_video_metadata")
ITER63_DIR = os.path.join(REPO_ROOT, "experiments", "iter63_decay_tab_rate")
MAKE_SUBMISSION_PATH = os.path.join(REPO_ROOT, "make_submission.py")
RESULTS_PATH = os.path.join(THIS_DIR, "results.json")
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")

FM_SEEDS = (0, 1, 2, 3, 4)
ITER63_SEEDS = (0, 1, 2, 3, 4)
TRAIN_SEED = 0

CAT_COLS = ["user_id", "video_id", "author_id", "tab", "last1"]
LGB_REFERENCE_COLUMNS = CAT_COLS + [
    "duration_ms", "decay_rate_5", "decay_act_5", "lastk_rate", "gap",
    "decay_tab_rate_3", "hist_watch_decay_mean_5",
]
LGB_CANDIDATE_COLUMNS = LGB_REFERENCE_COLUMNS + ["meta_upload_type"]
XGB_COLUMNS = CAT_COLS + [
    "duration_ms", "decay_tab_3", "lastk_rate", "gap", "decay_rate_5", "decay_act_5",
]
LGB_CONFIG = dict(
    objective="lambdarank", lambdarank_truncation_level=50, sigmoid=2.0,
    metric="ndcg", eval_at=[5], num_leaves=2, learning_rate=0.10,
    n_estimators=500, min_child_samples=200, reg_lambda=1.0,
    verbosity=-1, n_jobs=-1, linear_tree=True,
)

BEST_WEIGHTS = {"fm": 0.072, "lgb": 0.5184, "xgb": 0.3296, "i63": 0.08}
SECOND_WEIGHTS = {"fm": 0.052, "lgb": 0.5184, "xgb": 0.3496, "i63": 0.08}
THIRD_WEIGHTS = {"fm": 0.072, "lgb": 0.5184, "xgb": 0.3496, "i63": 0.06}
REFERENCE_VALID = 0.6994343996047974
REFERENCE_TEST = 0.6843225955963135
PRELIMINARY_DELTA = 0.0003
PROMOTION_DELTA = 0.001


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


def save_results(results):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(jsonable(results), f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  [wrote {RESULTS_PATH}]", flush=True)


def stable_user_order(user_ids):
    values = np.asarray(user_ids)
    order = np.argsort(values, kind="stable")
    groups = np.unique(values[order], return_counts=True)[1]
    return order, groups


def xgb_config():
    payload = json.load(open(os.path.join(REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "results.json"), encoding="utf-8"))
    return payload["selected_on_validation"]["config"]


def fit_lgb(frames, y, users, columns, seed):
    import lightgbm as lgb
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])
    Xtr = frames["train"][columns].iloc[train_order].reset_index(drop=True)
    Xva = frames["valid"][columns].iloc[valid_order].reset_index(drop=True)
    model = lgb.LGBMRanker(**LGB_CONFIG, random_state=seed)
    model.fit(
        Xtr, y["train"][train_order], group=train_groups,
        eval_set=[(Xva, y["valid"][valid_order])], eval_group=[valid_groups],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def fit_xgb(frames, y, users, columns, seed):
    import xgboost as xgb
    train_order, train_groups = stable_user_order(users["train"])
    valid_order, valid_groups = stable_user_order(users["valid"])
    Xtr = frames["train"][columns].iloc[train_order].reset_index(drop=True)
    Xva = frames["valid"][columns].iloc[valid_order].reset_index(drop=True)
    model = xgb.XGBRanker(**xgb_config(), random_state=seed, n_jobs=-1, verbosity=0)
    model.fit(
        Xtr, y["train"][train_order], group=train_groups,
        eval_set=[(Xva, y["valid"][valid_order])], eval_group=[valid_groups], verbose=False,
    )
    return model


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def within_user_percentile(scores, user_ids):
    s = pd.Series(np.asarray(scores, dtype=np.float64), copy=False)
    u = pd.Series(np.asarray(user_ids), copy=False)
    ranked = s.groupby(u, sort=False).rank(method="average", pct=True)
    return ranked.to_numpy(dtype=np.float64)


def blend(components, weights, users_arr, y_arr):
    b = sum(weights[k] * components[k] for k in weights)
    return evaluate(users_arr, y_arr, b)


def main():
    t0 = time.time()
    results = json.load(open(RESULTS_PATH, encoding="utf-8")) if os.path.exists(RESULTS_PATH) else {}

    print("=== loading yixi frames + fitting fixed FM/LGB/XGB components (seed 0 / 5-seed FM) ===", flush=True)
    features_mod = load_module(os.path.join(YIXI10_DIR, "features.py"), "confirm_yixi10_features")
    frames, y, users, _meta = features_mod.load_frames()

    lgb_model = fit_lgb(frames, y, users, LGB_CANDIDATE_COLUMNS, TRAIN_SEED)
    xgb_model = fit_xgb(frames, y, users, XGB_COLUMNS, TRAIN_SEED)
    lgb_valid = lgb_model.predict(frames["valid"][LGB_CANDIDATE_COLUMNS])
    lgb_test = lgb_model.predict(frames["test"][LGB_CANDIDATE_COLUMNS])
    xgb_valid = xgb_model.predict(frames["valid"][XGB_COLUMNS])
    xgb_test = xgb_model.predict(frames["test"][XGB_COLUMNS])

    submission = load_module(MAKE_SUBMISSION_PATH, "confirm_submission")
    splits = submission.load_ext(DATA_DIR, halflives=submission.HALFLIVES, tab_halflives=submission.TAB_HALFLIVES)
    encoded, dim = submission.encode_ext(
        splits, feature_set=submission.FEATURES, halflives=submission.HALFLIVES,
        tab_halflives=submission.TAB_HALFLIVES, alpha=0.5, n_buckets=20,
    )
    Xtr, ytr, utr = encoded["train"]
    Xva, yva, uva = encoded["valid"]
    Xte, yte, ute = encoded["test"]
    fm_valid_scores, fm_test_scores = [], []
    for seed in FM_SEEDS:
        print(f"  FM seed={seed}", flush=True)
        model = submission.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits["train"], dim, seed)
        fm_valid_scores.append(sigmoid(model.predict(Xva)))
        fm_test_scores.append(sigmoid(model.predict(Xte)))
    fm_valid = np.mean(np.stack(fm_valid_scores), axis=0)
    fm_test = np.mean(np.stack(fm_test_scores), axis=0)

    fixed_valid = {
        "fm": within_user_percentile(fm_valid, users["valid"]),
        "lgb": within_user_percentile(lgb_valid, users["valid"]),
        "xgb": within_user_percentile(xgb_valid, users["valid"]),
    }
    fixed_test = {
        "fm": within_user_percentile(fm_test, users["test"]),
        "lgb": within_user_percentile(lgb_test, users["test"]),
        "xgb": within_user_percentile(xgb_test, users["test"]),
    }
    sanity = blend({**fixed_valid, "i63": np.zeros_like(fixed_valid["fm"])},
                    {"fm": 0.10, "lgb": 0.52, "xgb": 0.38, "i63": 0.0}, users["valid"], y["valid"])
    print(f"  sanity check: 3-model blend valid = {sanity['primary']:.8f} (ref {REFERENCE_VALID:.8f})", flush=True)

    print("\n=== fitting iter63 rate_only GBM at 5 seeds, keeping per-seed raw scores ===", flush=True)
    iter63_train = load_module(os.path.join(ITER63_DIR, "train.py"), "confirm_iter63_train")
    i63_model0, i63_va0, i63_te0, i63_cache = iter63_train.run(DATA_DIR, "rate_only", seed=TRAIN_SEED, verbose=False)
    i63_dfs, i63_y, i63_u = i63_cache
    per_seed_valid_scores = {0: sigmoid(i63_model0.predict(i63_dfs["valid"]))}
    per_seed_test_scores = {0: sigmoid(i63_model0.predict(i63_dfs["test"]))}
    for seed in ITER63_SEEDS[1:]:
        m, _va, _te, _ = iter63_train.run(DATA_DIR, "rate_only", seed=seed, _cache=i63_cache)
        per_seed_valid_scores[seed] = sigmoid(m.predict(i63_dfs["valid"]))
        per_seed_test_scores[seed] = sigmoid(m.predict(i63_dfs["test"]))
        print(f"  seed={seed} scored", flush=True)

    # identity check re-confirmed (cheap): i63_u must match users (trusted arrays)
    for name, u_arr in (("valid", users["valid"]), ("test", users["test"])):
        trusted = np.asarray(u_arr).astype(str)
        i63_trusted = np.asarray(i63_u[name]).astype(str)
        if not np.array_equal(trusted, i63_trusted):
            raise AssertionError(f"identity re-check failed for {name}")
    print("  identity re-check OK (0 mismatches, both splits)", flush=True)

    print("\n=== per-seed iter63 blend at best weight point ===", flush=True)
    per_seed_blend = {}
    for seed in ITER63_SEEDS:
        i63_valid_pct = within_user_percentile(per_seed_valid_scores[seed], i63_u["valid"])
        i63_test_pct = within_user_percentile(per_seed_test_scores[seed], i63_u["test"])
        vm = blend({**fixed_valid, "i63": i63_valid_pct}, BEST_WEIGHTS, users["valid"], y["valid"])
        tm = blend({**fixed_test, "i63": i63_test_pct}, BEST_WEIGHTS, users["test"], y["test"])
        per_seed_blend[seed] = {
            "valid_primary": vm["primary"], "test_primary": tm["primary"],
            "valid_delta_vs_reference": vm["primary"] - REFERENCE_VALID,
        }
        print(f"  seed={seed}  valid={vm['primary']:.8f}  (delta={vm['primary']-REFERENCE_VALID:+.8f})  test={tm['primary']:.8f}", flush=True)

    # sigmoid-mean ensemble version, for direct comparison against per-seed spread
    i63_valid_ens = np.mean(np.stack([per_seed_valid_scores[s] for s in ITER63_SEEDS]), axis=0)
    i63_test_ens = np.mean(np.stack([per_seed_test_scores[s] for s in ITER63_SEEDS]), axis=0)
    i63_valid_ens_pct = within_user_percentile(i63_valid_ens, i63_u["valid"])
    i63_test_ens_pct = within_user_percentile(i63_test_ens, i63_u["test"])
    ens_vm = blend({**fixed_valid, "i63": i63_valid_ens_pct}, BEST_WEIGHTS, users["valid"], y["valid"])
    ens_tm = blend({**fixed_test, "i63": i63_test_ens_pct}, BEST_WEIGHTS, users["test"], y["test"])
    print(f"  [ensemble] valid={ens_vm['primary']:.8f}  (delta={ens_vm['primary']-REFERENCE_VALID:+.8f})  test={ens_tm['primary']:.8f}", flush=True)

    valid_vals = [per_seed_blend[s]["valid_primary"] for s in ITER63_SEEDS]
    n_positive = sum(1 for v in valid_vals if v - REFERENCE_VALID >= PRELIMINARY_DELTA)
    n_above_promo = sum(1 for v in valid_vals if v - REFERENCE_VALID >= PROMOTION_DELTA)
    seed_confirmation = {
        "per_seed_blend_at_best_weights": per_seed_blend,
        "ensemble_blend_at_best_weights": {"valid_primary": ens_vm["primary"], "test_primary": ens_tm["primary"],
                                            "valid_delta_vs_reference": ens_vm["primary"] - REFERENCE_VALID},
        "valid_spread": {"min": float(np.min(valid_vals)), "max": float(np.max(valid_vals)), "std": float(np.std(valid_vals))},
        "seeds_clearing_preliminary_delta": n_positive, "seeds_clearing_promotion_delta": n_above_promo,
        "total_seeds": len(ITER63_SEEDS),
    }
    results["seed_confirmation"] = seed_confirmation
    save_results(results)

    print("\n=== plateau check: 2nd/3rd-best fine-grid weight points (ensemble i63 score) ===", flush=True)
    plateau = {}
    for label, w in (("best", BEST_WEIGHTS), ("second", SECOND_WEIGHTS), ("third", THIRD_WEIGHTS)):
        vm = blend({**fixed_valid, "i63": i63_valid_ens_pct}, w, users["valid"], y["valid"])
        tm = blend({**fixed_test, "i63": i63_test_ens_pct}, w, users["test"], y["test"])
        plateau[label] = {
            "weights": w, "valid_primary": vm["primary"], "test_primary": tm["primary"],
            "valid_delta_vs_reference": vm["primary"] - REFERENCE_VALID,
            "test_delta_vs_reference": tm["primary"] - REFERENCE_TEST,
        }
        print(f"  [{label}] weights={w}  valid={vm['primary']:.8f} (d={vm['primary']-REFERENCE_VALID:+.8f})  "
              f"test={tm['primary']:.8f} (d={tm['primary']-REFERENCE_TEST:+.8f})", flush=True)
    results["plateau_check"] = plateau
    save_results(results)

    print("\n=== final verdict ===", flush=True)
    best_valid_delta = ens_vm["primary"] - REFERENCE_VALID
    is_plateau = all(plateau[k]["valid_delta_vs_reference"] >= PRELIMINARY_DELTA for k in ("best", "second", "third"))
    seed_robust = n_positive == len(ITER63_SEEDS)
    if best_valid_delta >= PROMOTION_DELTA:
        final_verdict = "PROMOTE"
    elif best_valid_delta >= PRELIMINARY_DELTA:
        final_verdict = "PRELIMINARY"
    else:
        final_verdict = "REJECT"
    results["final_summary"] = {
        "best_weights": BEST_WEIGHTS,
        "best_blend_valid_ensemble": ens_vm["primary"],
        "best_blend_test_ensemble": ens_tm["primary"],
        "valid_delta_vs_3model_reference": best_valid_delta,
        "test_delta_vs_3model_reference": ens_tm["primary"] - REFERENCE_TEST,
        "is_plateau_across_top3_grid_points": is_plateau,
        "robust_across_all_5_iter63_seeds_at_preliminary_bar": seed_robust,
        "preliminary_delta_threshold": PRELIMINARY_DELTA,
        "promotion_delta_threshold": PROMOTION_DELTA,
        "final_verdict": final_verdict,
        "selection_basis": "valid-only; test reported for the record only, not used for selection",
    }
    print(f"  best valid delta = {best_valid_delta:+.8f}  plateau={is_plateau}  seed_robust={seed_robust}  VERDICT={final_verdict}", flush=True)
    results["confirm_elapsed_seconds"] = time.time() - t0
    save_results(results)
    print(f"\n=== confirm.py done in {results['confirm_elapsed_seconds']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
