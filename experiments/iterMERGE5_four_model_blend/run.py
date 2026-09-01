"""iterMERGE5: test whether iter63's own distinct rate_only LightGBM (a
DIFFERENT, smaller feature set / config than yixi's LightGBM -- num_leaves=2
linear-tree on decay_rate_2.5/decay_act_2.5/decay_tab_rate_3/last1/
lastk_rate/gap -- vs. yixi's richer LGB_CANDIDATE_COLUMNS) improves the
production blend as a 4th ensemble member alongside FM/yixi-LGB/yixi-XGB.

Pipeline:
  1. Harness-fidelity part A: reproduce iterMERGE1_verify_yixi10's 3-model
     blend reference numbers exactly (LGB valid=0.68834144, XGB
     valid=0.66755420, FM valid=0.63987792, blend valid=0.69943440,
     test=0.68432260), same code path (yixi's own features.py chain via
     load_frames(), independent from-scratch LGB/XGB training,
     make_submission.py's train_one_fm reused as a module).
  2. Harness-fidelity part B: reproduce iter63's OWN rate_only GBM
     standalone valid, using iter63_decay_tab_rate/train.py's own run()/
     prepare() functions loaded as a module (not reimplemented). NOTE: the
     dispatch note for this experiment named the target reference as
     "~0.6768913269042969, matching iterYIXI9's LGB_REFERENCE_VALID
     constant" -- this turned out on inspection to be a *different*
     constant (iterYIXI9/iterYIXI10's own intermediate LGB_REFERENCE_COLUMNS
     model, a richer feature set with decay_rate_5/decay_act_5/
     hist_watch_decay_mean_5, not iter63's rate_only model at all). The
     actual, code-verified reference for iter63's own rate_only model is
     0.6716787219047546 (visible directly in iter63_decay_tab_rate/run.log
     as "valid=0.67168" and independently corroborated by
     iterYIXI6_cross_model_feature_transfer/common.py's own
     LGB_REFERENCE_VALID = 0.6716787219047546, which its harness.py labels
     "iter63 LightGBM"). This script checks against the code-verified
     number and documents the discrepancy; see RESULT.md.
  3. Row-alignment check: both yixi's frames (iterYIXI10 chain) and iter63's
     data_ext.py splits are built by a plain two-file sequential CSV scan
     bucketed by the same (train/valid/test) date ranges, with no
     shuffling/sorting -- so their row order is expected to already agree
     position-for-position. This is verified explicitly here (not assumed):
     user_id and video_id are compared row-for-row between yixi's frames
     and iter63's splits for valid/test.
  4. Train iter63's rate_only GBM at 5 seeds (0-4) via its own train.py
     code path, sigmoid-mean ensemble (matching the FM ensembling
     convention used elsewhere in this project).
  5. within-user-percentile normalize the iter63 ensemble the same way as
     the other 3 components, then weight-search a 4-way blend on VALID
     ONLY: a coarse sweep over the iter63 component's weight (rescaling
     FM/LGB/XGB proportionally from the production 10/52/38 point), then a
     local grid refinement around the best point found.
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402

YIXI10_DIR = os.path.join(REPO_ROOT, "experiments", "iterYIXI10_video_metadata")
YIXI5_RESULTS_PATH = os.path.join(REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "results.json")
ITER63_DIR = os.path.join(REPO_ROOT, "experiments", "iter63_decay_tab_rate")
MAKE_SUBMISSION_PATH = os.path.join(REPO_ROOT, "make_submission.py")
RESULTS_PATH = os.path.join(THIS_DIR, "results.json")

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

PRODUCTION_WEIGHTS = {"fm": 0.10, "lgb": 0.52, "xgb": 0.38}
CLAIMED_VALID = 0.69943440
CLAIMED_TEST = 0.68432260
HARNESS_TOLERANCE = 1e-6

# See docstring: this is the code-verified reference for iter63's OWN
# rate_only model (NOT the dispatch note's stated 0.6768913269042969,
# which is a different, unrelated constant -- see note above and RESULT.md).
ITER63_RATE_ONLY_REFERENCE_VALID = 0.6716787219047546
ITER63_HARNESS_TOLERANCE = 1e-6

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
    payload = json.load(open(YIXI5_RESULTS_PATH, encoding="utf-8"))
    return payload["selected_on_validation"]["config"]


def fit_lgb(frames, y, users, columns, seed):
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
    values = ranked.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise AssertionError("non-finite within-user percentile")
    return values


def blend4(components, weights, users_arr, y_arr):
    b = sum(weights[k] * components[k] for k in weights)
    return evaluate(users_arr, y_arr, b)


def main():
    t0 = time.time()
    results = {"experiment": "iterMERGE5_four_model_blend"}

    # =================================================================
    print("=== [1/7] loading yixi's YIXI10 feature frames ===", flush=True)
    features_mod = load_module(os.path.join(YIXI10_DIR, "features.py"), "merge5_yixi10_features")
    frames, y, users, _meta = features_mod.load_frames()
    missing = [c for c in LGB_CANDIDATE_COLUMNS if c not in frames["train"].columns]
    if missing:
        raise AssertionError(f"missing candidate LGB columns: {missing}")
    print(f"  rows train/valid/test = {len(frames['train'])}/{len(frames['valid'])}/{len(frames['test'])}", flush=True)

    print("=== [2/7] harness-fidelity part A: LGB/XGB (seed 0) + FM (5-seed ensemble) ===", flush=True)
    lgb_cand_model = fit_lgb(frames, y, users, LGB_CANDIDATE_COLUMNS, TRAIN_SEED)
    xgb_model = fit_xgb(frames, y, users, XGB_COLUMNS, TRAIN_SEED)
    lgb_cand_valid = lgb_cand_model.predict(frames["valid"][LGB_CANDIDATE_COLUMNS])
    lgb_cand_test = lgb_cand_model.predict(frames["test"][LGB_CANDIDATE_COLUMNS])
    xgb_valid = xgb_model.predict(frames["valid"][XGB_COLUMNS])
    xgb_test = xgb_model.predict(frames["test"][XGB_COLUMNS])
    lgb_valid_metrics = evaluate(users["valid"], y["valid"], lgb_cand_valid)
    xgb_valid_metrics = evaluate(users["valid"], y["valid"], xgb_valid)
    print(f"  LGB candidate valid primary = {lgb_valid_metrics['primary']:.8f}  (ref 0.68834144)", flush=True)
    print(f"  XGB valid primary           = {xgb_valid_metrics['primary']:.8f}  (ref 0.66755420)", flush=True)

    submission = load_module(MAKE_SUBMISSION_PATH, "merge5_submission")
    splits = submission.load_ext(
        os.path.join(REPO_ROOT, "KuaiRand-Pure", "data"),
        halflives=submission.HALFLIVES, tab_halflives=submission.TAB_HALFLIVES,
    )
    encoded, dim = submission.encode_ext(
        splits, feature_set=submission.FEATURES, halflives=submission.HALFLIVES,
        tab_halflives=submission.TAB_HALFLIVES, alpha=0.5, n_buckets=20,
    )
    Xtr, ytr, utr = encoded["train"]
    Xva, yva, uva = encoded["valid"]
    Xte, yte, ute = encoded["test"]
    if not np.array_equal(np.asarray(yva), y["valid"]) or not np.array_equal(np.asarray(uva), np.asarray(users["valid"])):
        raise AssertionError("FM/native valid alignment mismatch")
    if not np.array_equal(np.asarray(yte), y["test"]) or not np.array_equal(np.asarray(ute), np.asarray(users["test"])):
        raise AssertionError("FM/native test alignment mismatch")

    fm_valid_scores, fm_test_scores = [], []
    for seed in FM_SEEDS:
        print(f"  fitting FM seed={seed}", flush=True)
        model = submission.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits["train"], dim, seed)
        fm_valid_scores.append(sigmoid(model.predict(Xva)))
        fm_test_scores.append(sigmoid(model.predict(Xte)))
    fm_valid = np.mean(np.stack(fm_valid_scores), axis=0)
    fm_test = np.mean(np.stack(fm_test_scores), axis=0)
    fm_valid_metrics = evaluate(users["valid"], y["valid"], fm_valid)
    print(f"  FM ensemble valid primary   = {fm_valid_metrics['primary']:.8f}  (ref 0.63987792)", flush=True)

    valid_components_3 = {
        "fm": within_user_percentile(fm_valid, users["valid"]),
        "lgb": within_user_percentile(lgb_cand_valid, users["valid"]),
        "xgb": within_user_percentile(xgb_valid, users["valid"]),
    }
    test_components_3 = {
        "fm": within_user_percentile(fm_test, users["test"]),
        "lgb": within_user_percentile(lgb_cand_test, users["test"]),
        "xgb": within_user_percentile(xgb_test, users["test"]),
    }
    blend3_valid = blend4(valid_components_3, PRODUCTION_WEIGHTS, users["valid"], y["valid"])
    blend3_test = blend4(test_components_3, PRODUCTION_WEIGHTS, users["test"], y["test"])

    harness_a = {
        "lgb_candidate_valid": lgb_valid_metrics["primary"],
        "xgb_valid": xgb_valid_metrics["primary"],
        "fm_valid": fm_valid_metrics["primary"],
        "blend_valid": blend3_valid["primary"],
        "blend_test": blend3_test["primary"],
        "claimed": {
            "lgb_candidate_valid": 0.68834144, "xgb_valid": 0.66755420,
            "fm_valid": 0.63987792, "blend_valid": CLAIMED_VALID, "blend_test": CLAIMED_TEST,
        },
    }
    harness_a["deltas"] = {k: harness_a[k] - harness_a["claimed"][k] for k in harness_a["claimed"]}
    harness_a["all_pass_1e-6"] = bool(all(abs(v) <= HARNESS_TOLERANCE for v in harness_a["deltas"].values()))
    results["harness_fidelity_3model"] = harness_a
    print(f"  harness A all_pass_1e-6 = {harness_a['all_pass_1e-6']}  deltas={harness_a['deltas']}", flush=True)
    save_results(results)
    if not harness_a["all_pass_1e-6"]:
        print("!!! HARNESS FIDELITY A FAILED -- stopping !!!", flush=True)
        return

    # =================================================================
    print("\n=== [3/7] harness-fidelity part B: iter63's own rate_only GBM standalone ===", flush=True)
    iter63_train = load_module(os.path.join(ITER63_DIR, "train.py"), "merge5_iter63_train")
    DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
    i63_model0, i63_va0, i63_te0, i63_cache = iter63_train.run(DATA_DIR, "rate_only", seed=TRAIN_SEED, verbose=True)
    i63_delta = i63_va0["primary"] - ITER63_RATE_ONLY_REFERENCE_VALID
    harness_b = {
        "reference_valid_code_verified": ITER63_RATE_ONLY_REFERENCE_VALID,
        "reference_valid_dispatch_note_stated": 0.6768913269042969,
        "reproduced_valid": i63_va0["primary"],
        "reproduced_test": i63_te0["primary"],
        "delta_vs_code_verified_reference": i63_delta,
        "pass_1e-6": bool(abs(i63_delta) <= ITER63_HARNESS_TOLERANCE),
        "note": (
            "The dispatch note's stated reference (0.6768913269042969) is "
            "iterYIXI9_watch_depth_history/common.py's LGB_REFERENCE_VALID, "
            "which is actually a DIFFERENT, richer intermediate model in "
            "yixi's own feature chain (LGB_REFERENCE_COLUMNS: "
            "decay_rate_5/decay_act_5/hist_watch_decay_mean_5), not iter63's "
            "rate_only model. iter63's own code-verified reference is "
            "0.6716787219047546, which matches "
            "iterYIXI6_cross_model_feature_transfer/common.py's "
            "LGB_REFERENCE_VALID exactly (that file's harness.py labels it "
            "'iter63 LightGBM'), and matches iter63_decay_tab_rate/run.log's "
            "own recorded rate_only seed-0 valid=0.67168. This script checks "
            "against the code-verified number."
        ),
    }
    results["harness_fidelity_iter63_standalone"] = harness_b
    print(f"  iter63 rate_only seed0 valid = {i63_va0['primary']:.10f}  "
          f"(code-verified ref {ITER63_RATE_ONLY_REFERENCE_VALID:.10f})  pass={harness_b['pass_1e-6']}", flush=True)
    save_results(results)
    if not harness_b["pass_1e-6"]:
        print("!!! HARNESS FIDELITY B FAILED -- stopping !!!", flush=True)
        return

    # =================================================================
    print("\n=== [4/7] row-alignment check: yixi frames vs iter63 splits (valid/test) ===", flush=True)
    i63_dfs, i63_y, i63_u = i63_cache
    # IMPORTANT: an earlier version of this check compared against
    # i63_dfs[name]["user_id"], which is a *categorical* column whose
    # categories are restricted to values seen in the TRAIN split
    # (iter63_decay_tab_rate/train.py:88, `pd.CategoricalDtype(categories=
    # dfs['train'][c].unique())`, applied via `.astype(cats[c])` at line 91).
    # Any user_id in valid/test that's unseen in train becomes NaN under that
    # cast -- a legitimate cold-start feature-encoding artifact of iter63's
    # OWN model (LightGBM treats it as a missing category), NOT a row-order
    # bug. That produced 1990/6171 "mismatches" on valid/test that were
    # actually just cold-start users, which raised a false alarm about
    # misalignment. The correct comparison uses the *trusted* identity
    # arrays: i63_u[name] (returned separately by iter63's own prepare(),
    # sourced from splits[name] before any categorical restriction -- see
    # train.py:93 `u = {name: [x[_de.IDX['user_id']] for x in splits[name]]
    # ...}`) and a freshly-derived raw video_id array from iter63's own
    # data_ext splits (same source, no categorical cast). Verified via
    # ad hoc diagnostic (/tmp/check_align11.py) before making this fix:
    # trusted_u vs i63_dfs['user_id'] -> 1990/6171 mismatches (the false
    # alarm); trusted_u vs i63_u -> 0 mismatches on both valid and test.
    i63_splits_raw = iter63_train._de.load_ext(DATA_DIR, use_cache=True)
    i63_video_raw = {
        name: np.array([x[iter63_train._de.IDX["video_id"]] for x in i63_splits_raw[name]], dtype=str)
        for name in ("valid", "test")
    }
    alignment = {}
    for name in ("valid", "test"):
        yixi_uid = np.asarray(users[name]).astype(str)
        yixi_vid = frames[name]["video_id"].astype(str).to_numpy()
        i63_uid = np.asarray(i63_u[name]).astype(str)
        i63_vid = i63_video_raw[name]
        n = len(yixi_uid)
        n_i63 = len(i63_uid)
        uid_mismatch = int((yixi_uid != i63_uid).sum()) if n == n_i63 else -1
        vid_mismatch_raw = int((yixi_vid != i63_vid).sum()) if n == n_i63 else -1
        # yixi's meta layer has a handful of rows with NaN video_id (missing
        # basic-metadata join); those show up as string mismatches here but
        # are a known, benign, already-documented artifact unrelated to row
        # order (user_id still matches at those positions).
        nan_video_yixi = int(frames[name]["video_id"].isna().sum())
        label_mismatch = int((np.asarray(y[name]) != np.asarray(i63_y[name])).sum()) if len(y[name]) == len(i63_y[name]) else -1
        alignment[name] = {
            "rows_yixi": int(n), "rows_iter63": int(n_i63),
            "row_count_match": bool(n == n_i63),
            "user_id_mismatches_trusted_arrays": uid_mismatch,
            "video_id_mismatches_raw": vid_mismatch_raw,
            "yixi_nan_video_id_rows": nan_video_yixi,
            "video_id_mismatches_excluding_known_nan": vid_mismatch_raw - nan_video_yixi,
            "label_mismatches": label_mismatch,
            "note": (
                "user_id compared using the trusted raw arrays (users[name] "
                "from yixi's build_frames() internal assertion, i63_u[name] "
                "from iter63's prepare()), not the categorical dfs['user_id'] "
                "column (which is train-vocabulary-restricted and legitimately "
                "NaNs out cold-start valid/test users -- see comment above)."
            ),
        }
        print(f"  [{name}] rows={n}  user_id_mismatches(trusted)={uid_mismatch}  "
              f"video_id_mismatches={vid_mismatch_raw} (of which {nan_video_yixi} are "
              f"known yixi NaN-video rows)  label_mismatches={label_mismatch}", flush=True)
    results["row_alignment_check"] = alignment
    save_results(results)
    aligned_ok = all(
        alignment[name]["row_count_match"]
        and alignment[name]["user_id_mismatches_trusted_arrays"] == 0
        and alignment[name]["video_id_mismatches_excluding_known_nan"] == 0
        and alignment[name]["label_mismatches"] == 0
        for name in ("valid", "test")
    )
    print(f"  row alignment OK (position-based combine is safe) = {aligned_ok}", flush=True)
    if not aligned_ok:
        raise AssertionError("row alignment check failed -- cannot safely combine scores position-wise")

    # =================================================================
    print("\n=== [5/7] training iter63 rate_only GBM at 5 seeds, sigmoid-mean ensemble ===", flush=True)
    i63_valid_seed_scores = [sigmoid(i63_model0.predict(i63_dfs["valid"]))]
    i63_test_seed_scores = [sigmoid(i63_model0.predict(i63_dfs["test"]))]
    per_seed = {
        0: {"valid_primary": i63_va0["primary"], "test_primary": i63_te0["primary"]},
    }
    for seed in ITER63_SEEDS[1:]:
        print(f"  fitting iter63 rate_only seed={seed}", flush=True)
        m, va, te, _ = iter63_train.run(DATA_DIR, "rate_only", seed=seed, _cache=i63_cache)
        i63_valid_seed_scores.append(sigmoid(m.predict(i63_dfs["valid"])))
        i63_test_seed_scores.append(sigmoid(m.predict(i63_dfs["test"])))
        per_seed[seed] = {"valid_primary": va["primary"], "test_primary": te["primary"]}
        print(f"    seed={seed}  valid={va['primary']:.6f}  test={te['primary']:.6f}", flush=True)

    i63_valid_ens = np.mean(np.stack(i63_valid_seed_scores), axis=0)
    i63_test_ens = np.mean(np.stack(i63_test_seed_scores), axis=0)
    i63_valid_ens_metrics = evaluate(i63_u["valid"], i63_y["valid"], i63_valid_ens)
    i63_test_ens_metrics = evaluate(i63_u["test"], i63_y["test"], i63_test_ens)
    results["iter63_5seed_ensemble"] = {
        "per_seed": per_seed,
        "ensemble_valid_primary": i63_valid_ens_metrics["primary"],
        "ensemble_test_primary": i63_test_ens_metrics["primary"],
    }
    print(f"  iter63 5-seed sigmoid-mean ensemble: valid={i63_valid_ens_metrics['primary']:.8f} "
          f"test={i63_test_ens_metrics['primary']:.8f}", flush=True)
    save_results(results)

    # =================================================================
    print("\n=== [6/7] 4-way within-user-percentile blend weight search (VALID ONLY) ===", flush=True)
    valid_components_4 = dict(valid_components_3)
    valid_components_4["i63"] = within_user_percentile(i63_valid_ens, i63_u["valid"])
    test_components_4 = dict(test_components_3)
    test_components_4["i63"] = within_user_percentile(i63_test_ens, i63_u["test"])

    def eval4(w):
        vm = blend4(valid_components_4, w, users["valid"], y["valid"])
        tm = blend4(test_components_4, w, users["test"], y["test"])
        return vm, tm

    # Stage A: coarse sweep over iter63 weight, rescaling FM/LGB/XGB
    # proportionally from the production 10/52/38 point.
    coarse_grid = []
    base_fm, base_lgb, base_xgb = PRODUCTION_WEIGHTS["fm"], PRODUCTION_WEIGHTS["lgb"], PRODUCTION_WEIGHTS["xgb"]
    for w_i63 in (0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30):
        scale = 1.0 - w_i63
        w = {"fm": base_fm * scale, "lgb": base_lgb * scale, "xgb": base_xgb * scale, "i63": w_i63}
        vm, tm = eval4(w)
        coarse_grid.append({"weights": dict(w), "valid_primary": vm["primary"], "test_primary": tm["primary"]})
        print(f"  coarse w_i63={w_i63:.2f}  weights={{'fm':{w['fm']:.3f},'lgb':{w['lgb']:.3f},"
              f"'xgb':{w['xgb']:.3f},'i63':{w['i63']:.3f}}}  valid={vm['primary']:.8f}", flush=True)
    results["weight_search_coarse"] = coarse_grid
    save_results(results)

    best_coarse = max(coarse_grid, key=lambda r: r["valid_primary"])
    print(f"  best coarse point: {best_coarse['weights']}  valid={best_coarse['valid_primary']:.8f}", flush=True)

    # Stage B: local grid refinement around the best coarse point.
    bw = best_coarse["weights"]
    fine_grid = []
    steps = (-0.04, -0.02, 0.0, 0.02, 0.04)
    seen = set()
    for d_i63 in steps:
        for d_fm in steps:
            for d_lgb in steps:
                w_i63 = bw["i63"] + d_i63
                w_fm = bw["fm"] + d_fm
                w_lgb = bw["lgb"] + d_lgb
                w_xgb = 1.0 - w_i63 - w_fm - w_lgb
                if min(w_i63, w_fm, w_lgb, w_xgb) < -1e-9:
                    continue
                w_i63, w_fm, w_lgb, w_xgb = (max(0.0, v) for v in (w_i63, w_fm, w_lgb, w_xgb))
                key = (round(w_i63, 4), round(w_fm, 4), round(w_lgb, 4), round(w_xgb, 4))
                if key in seen:
                    continue
                seen.add(key)
                w = {"fm": w_fm, "lgb": w_lgb, "xgb": w_xgb, "i63": w_i63}
                vm, tm = eval4(w)
                fine_grid.append({"weights": dict(w), "valid_primary": vm["primary"], "test_primary": tm["primary"]})
    fine_grid.sort(key=lambda r: -r["valid_primary"])
    results["weight_search_fine"] = fine_grid[:30]  # keep top 30 for the record
    print(f"  fine grid: {len(fine_grid)} combos evaluated", flush=True)
    for r in fine_grid[:5]:
        print(f"    valid={r['valid_primary']:.8f}  test={r['test_primary']:.8f}  weights={r['weights']}", flush=True)
    save_results(results)

    best = fine_grid[0]
    best_valid = best["valid_primary"]
    best_test = best["test_primary"]
    best_delta = best_valid - blend3_valid["primary"]

    print(f"\n=== [7/7] verdict ===", flush=True)
    if best_delta >= PROMOTION_DELTA:
        verdict = "PROMOTE"
    elif best_delta >= PRELIMINARY_DELTA:
        verdict = "PRELIMINARY"
    else:
        verdict = "REJECT"
    results["summary"] = {
        "reference_3model_blend_valid": blend3_valid["primary"],
        "reference_3model_blend_test": blend3_test["primary"],
        "best_4model_weights": best["weights"],
        "best_4model_blend_valid": best_valid,
        "best_4model_blend_test": best_test,
        "best_delta_vs_3model_valid": best_delta,
        "preliminary_delta_threshold": PRELIMINARY_DELTA,
        "promotion_delta_threshold": PROMOTION_DELTA,
        "verdict": verdict,
        "selection_basis": "valid-only; test reported for the record only, not used for selection",
    }
    print(f"  3-model reference valid = {blend3_valid['primary']:.8f}", flush=True)
    print(f"  best 4-model valid      = {best_valid:.8f}  (delta={best_delta:+.8f})", flush=True)
    print(f"  best 4-model test       = {best_test:.8f}  (not used for selection)", flush=True)
    print(f"  best weights            = {best['weights']}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)

    results["environment"] = {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "lightgbm": lgb.__version__, "xgboost": xgb.__version__,
    }
    results["elapsed_seconds"] = time.time() - t0
    save_results(results)
    print(f"\n=== done in {results['elapsed_seconds']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
