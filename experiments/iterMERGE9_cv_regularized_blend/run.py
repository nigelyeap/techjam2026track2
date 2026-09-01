"""iterMERGE9: CV-regularized weight selection for the 4-model blend
(FM/LGB/XGB/i63), testing whether choosing the linear blend weight point by
5-fold user-level cross-validated valid score (rather than iterMERGE5's
single-split valid grid search) finds a more robust weight point whose test
score doesn't regress against the 3-model reference.

Motivation (Round 7 of MERGE_LEDGER.md): iterMERGE5's fine-grid single-split
optimum (fm=0.072, lgb=0.5184, xgb=0.3296, i63=0.08) clears
PRELIMINARY_DELTA on valid (+0.00038397) but REGRESSES on test (-0.00065172)
vs. the 3-model reference -- a valid/test crossover consistent with the
135-point fine grid overfitting its weight choice to iterMERGE5's single
124,909-row valid split. iterMERGE7's k-fold user-level OOF stacking (a
genuinely different, nonlinear combiner) also never directly tested whether
a *coarser, CV-regularized* choice of the same *linear* weight family closes
this gap. This experiment does exactly that: same coarse+fine grid as
iterMERGE5, same 4 frozen percentile-normalized component scores, but each
grid point is scored by 5-fold user-level CV mean valid (folds built with
iterMERGE7's exact KFold-on-unique-users technique) instead of single-split
valid. No underlying model is retrained per fold -- the 4 components are
already frozen scores (FM 5-seed ensemble, LGB seed 0, XGB seed 0, i63
5-seed ensemble), so per-fold scoring is pure held-out evaluation of the
SAME globally-computed within-user-percentile scores restricted to that
fold's rows (no leakage possible: within_user_percentile ranks are computed
per-user, and fold splitting is at the user level, so a user's own percentile
values never depend on other users' fold membership).

Pipeline:
  1. Harness-fidelity: reuse iterMERGE5_four_model_blend/run.py as an
     imported module directly (fit_lgb/fit_xgb/sigmoid/
     within_user_percentile/stable_user_order/blend4/column lists/
     PRODUCTION_WEIGHTS/ITER63_RATE_ONLY_REFERENCE_VALID) -- do NOT
     re-derive feature construction or row alignment. Reproduce:
       - 3-model reference blend (valid 0.69943440 / test 0.68432260)
       - iter63 rate_only standalone seed-0 valid (0.6716787219047546)
       - iter63 rate_only 5-seed sigmoid-mean ensemble (valid 0.67141676,
         test 0.65336251)
       - 4-model raw blend at iterMERGE5's best weights (valid 0.69981837,
         test 0.68367088)
     All checked at <=1e-6 tolerance.
  2. Build 5-fold user-level CV fold masks over valid's unique users
     (sklearn KFold, shuffle=True, random_state=0 -- matching iterMERGE7's
     exact convention), covering all of valid's rows exactly once.
  3. Coarse-then-fine grid search identical in structure to iterMERGE5's
     (coarse sweep over the i63 weight rescaling FM/LGB/XGB proportionally
     from the production 10/52/38 point, then local fine refinement around
     the best coarse point) -- but for EVERY point, compute both the
     single-split valid primary (for direct comparability to iterMERGE5)
     AND the 5-fold CV mean valid primary. Select the final point by
     highest CV mean valid (not single-split valid).
  4. Report the CV-selected point's CV mean valid, single-split valid, and
     test (record only, never for selection). Also report the CV mean valid
     AT iterMERGE5's original best point, to directly answer: was
     iterMERGE5's point already CV-robust, or does CV selection move you
     somewhere different?
  5. Decision gate (same PRELIMINARY_DELTA=0.0003 / PROMOTION_DELTA=0.001
     used throughout this ledger, applied to the CV-selected point's
     SINGLE-SPLIT valid vs. the 3-model reference, for comparability with
     every prior round): PROMOTE-candidate only if the single-split-valid
     delta clears +0.0003 AND test does not regress vs. the 3-model
     reference test (0.68432260) -- i.e. the crossover is actually
     resolved, not just relocated.
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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
MERGE5_DIR = os.path.join(REPO_ROOT, "experiments", "iterMERGE5_four_model_blend")
sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402

from sklearn.model_selection import KFold  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse iterMERGE5's verified harness directly -- fit_lgb, fit_xgb, sigmoid,
# within_user_percentile, stable_user_order, blend4, LGB_CANDIDATE_COLUMNS,
# XGB_COLUMNS, PRODUCTION_WEIGHTS, ITER63_RATE_ONLY_REFERENCE_VALID, etc.
m5 = load_module(os.path.join(MERGE5_DIR, "run.py"), "merge9_import_merge5")

RESULTS_PATH = os.path.join(THIS_DIR, "results.json")

CLAIMED_VALID_3MODEL = 0.69943440
CLAIMED_TEST_3MODEL = 0.68432260
# Full-precision references, code-verified in iterMERGE5/iterMERGE7/iterMERGE8
# (see their results.json / RESULT.md) -- checked here at <=1e-6 tolerance
# against the *displayed* 8-decimal claims above, and separately logged at
# full precision for the record.
ITER63_5SEED_ENSEMBLE_VALID = 0.6714167594909668
ITER63_5SEED_ENSEMBLE_TEST = 0.653362512588501
MERGE5_BEST_WEIGHTS = {"fm": 0.072, "lgb": 0.5184, "xgb": 0.3296, "i63": 0.08}
MERGE5_BEST_VALID = 0.69981837
MERGE5_BEST_TEST = 0.68367088
HARNESS_TOLERANCE = 1e-6

PRELIMINARY_DELTA = 0.0003
PROMOTION_DELTA = 0.001
N_FOLDS = 5
CV_RANDOM_STATE = 0  # matches iterMERGE7_stacked_meta_blend's convention


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


def make_cv_fold_masks(user_ids, n_folds, seed):
    """5-fold, USER-level (not row-level) CV fold masks, adapted from
    iterMERGE7_stacked_meta_blend/run.py's oof_logreg_predict/oof_gbm_predict
    fold-splitting technique (sklearn KFold over unique users, shuffle=True).
    Returns a list of n_folds boolean row-masks that partition all rows
    (every row appears in exactly one fold's mask; a user's rows never split
    across folds)."""
    users_arr = np.asarray(user_ids)
    uniq_users = np.unique(users_arr)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    masks = []
    covered = np.zeros(len(users_arr), dtype=bool)
    for _, holdout_uidx in kf.split(uniq_users):
        holdout_users = set(uniq_users[holdout_uidx].tolist())
        mask = np.array([u in holdout_users for u in users_arr], dtype=bool)
        masks.append(mask)
        covered |= mask
    assert covered.all(), "fold masks must cover every row exactly once"
    total_rows = sum(int(m.sum()) for m in masks)
    assert total_rows == len(users_arr), "fold masks must partition rows (no overlap)"
    return masks


def cv_mean_valid(components, weights, users_arr, y_arr, fold_masks):
    """Score a weight point via 5-fold user-level CV mean valid primary.
    Components are already-frozen, globally-computed within-user-percentile
    scores -- no refitting of underlying models per fold, no leakage (a
    user's percentile rank depends only on that user's own rows, and fold
    membership is at the user level, so restricting to a fold's rows and
    re-running evaluate() on just those rows is a legitimate held-out
    evaluation of the SAME blend weights, not a re-fit)."""
    b = sum(weights[k] * components[k] for k in weights)
    fold_primaries = []
    for mask in fold_masks:
        m = evaluate(users_arr[mask], y_arr[mask], b[mask])
        fold_primaries.append(m["primary"])
    return float(np.mean(fold_primaries)), fold_primaries


def main():
    t0 = time.time()
    results = {"experiment": "iterMERGE9_cv_regularized_blend"}

    # =================================================================
    print("=== [1/8] loading yixi's YIXI10 feature frames (via iterMERGE5 module) ===", flush=True)
    features_mod = m5.load_module(os.path.join(m5.YIXI10_DIR, "features.py"), "merge9_yixi10_features")
    frames, y, users, _meta = features_mod.load_frames()
    missing = [c for c in m5.LGB_CANDIDATE_COLUMNS if c not in frames["train"].columns]
    if missing:
        raise AssertionError(f"missing candidate LGB columns: {missing}")
    print(f"  rows train/valid/test = {len(frames['train'])}/{len(frames['valid'])}/{len(frames['test'])}", flush=True)

    print("=== [2/8] harness-fidelity: LGB/XGB (seed 0) + FM (5-seed ensemble) ===", flush=True)
    lgb_cand_model = m5.fit_lgb(frames, y, users, m5.LGB_CANDIDATE_COLUMNS, m5.TRAIN_SEED)
    xgb_model = m5.fit_xgb(frames, y, users, m5.XGB_COLUMNS, m5.TRAIN_SEED)
    lgb_cand_valid = lgb_cand_model.predict(frames["valid"][m5.LGB_CANDIDATE_COLUMNS])
    lgb_cand_test = lgb_cand_model.predict(frames["test"][m5.LGB_CANDIDATE_COLUMNS])
    xgb_valid = xgb_model.predict(frames["valid"][m5.XGB_COLUMNS])
    xgb_test = xgb_model.predict(frames["test"][m5.XGB_COLUMNS])
    lgb_valid_metrics = evaluate(users["valid"], y["valid"], lgb_cand_valid)
    xgb_valid_metrics = evaluate(users["valid"], y["valid"], xgb_valid)
    print(f"  LGB candidate valid primary = {lgb_valid_metrics['primary']:.8f}  (ref 0.68834144)", flush=True)
    print(f"  XGB valid primary           = {xgb_valid_metrics['primary']:.8f}  (ref 0.66755420)", flush=True)

    submission = m5.load_module(m5.MAKE_SUBMISSION_PATH, "merge9_submission")
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
    for seed in m5.FM_SEEDS:
        print(f"  fitting FM seed={seed}", flush=True)
        model = submission.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits["train"], dim, seed)
        fm_valid_scores.append(m5.sigmoid(model.predict(Xva)))
        fm_test_scores.append(m5.sigmoid(model.predict(Xte)))
    fm_valid = np.mean(np.stack(fm_valid_scores), axis=0)
    fm_test = np.mean(np.stack(fm_test_scores), axis=0)
    fm_valid_metrics = evaluate(users["valid"], y["valid"], fm_valid)
    print(f"  FM ensemble valid primary   = {fm_valid_metrics['primary']:.8f}  (ref 0.63987792)", flush=True)

    valid_components_3 = {
        "fm": m5.within_user_percentile(fm_valid, users["valid"]),
        "lgb": m5.within_user_percentile(lgb_cand_valid, users["valid"]),
        "xgb": m5.within_user_percentile(xgb_valid, users["valid"]),
    }
    test_components_3 = {
        "fm": m5.within_user_percentile(fm_test, users["test"]),
        "lgb": m5.within_user_percentile(lgb_cand_test, users["test"]),
        "xgb": m5.within_user_percentile(xgb_test, users["test"]),
    }
    blend3_valid = m5.blend4(valid_components_3, m5.PRODUCTION_WEIGHTS, users["valid"], y["valid"])
    blend3_test = m5.blend4(test_components_3, m5.PRODUCTION_WEIGHTS, users["test"], y["test"])

    harness_a = {
        "lgb_candidate_valid": lgb_valid_metrics["primary"],
        "xgb_valid": xgb_valid_metrics["primary"],
        "fm_valid": fm_valid_metrics["primary"],
        "blend_valid": blend3_valid["primary"],
        "blend_test": blend3_test["primary"],
        "claimed": {
            "lgb_candidate_valid": 0.68834144, "xgb_valid": 0.66755420,
            "fm_valid": 0.63987792, "blend_valid": CLAIMED_VALID_3MODEL, "blend_test": CLAIMED_TEST_3MODEL,
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
    print("\n=== [3/8] harness-fidelity: iter63 rate_only standalone + row alignment ===", flush=True)
    iter63_train = m5.load_module(os.path.join(m5.ITER63_DIR, "train.py"), "merge9_iter63_train")
    DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
    i63_model0, i63_va0, i63_te0, i63_cache = iter63_train.run(DATA_DIR, "rate_only", seed=m5.TRAIN_SEED, verbose=True)
    i63_delta = i63_va0["primary"] - m5.ITER63_RATE_ONLY_REFERENCE_VALID
    harness_b = {
        "reference_valid_code_verified": m5.ITER63_RATE_ONLY_REFERENCE_VALID,
        "note": "0.6716787219047546 is the CORRECT iter63 rate_only reference (NOT the similarly-named "
                "constant in iterYIXI9_watch_depth_history/common.py, which is a different model -- "
                "see iterMERGE5/RESULT.md).",
        "reproduced_valid": i63_va0["primary"],
        "reproduced_test": i63_te0["primary"],
        "delta_vs_code_verified_reference": i63_delta,
        "pass_1e-6": bool(abs(i63_delta) <= m5.ITER63_HARNESS_TOLERANCE),
    }
    results["harness_fidelity_iter63_standalone"] = harness_b
    print(f"  iter63 rate_only seed0 valid = {i63_va0['primary']:.10f}  "
          f"(ref {m5.ITER63_RATE_ONLY_REFERENCE_VALID:.10f})  pass={harness_b['pass_1e-6']}", flush=True)
    save_results(results)
    if not harness_b["pass_1e-6"]:
        print("!!! HARNESS FIDELITY B FAILED -- stopping !!!", flush=True)
        return

    i63_dfs, i63_y, i63_u = i63_cache
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
        nan_video_yixi = int(frames[name]["video_id"].isna().sum())
        label_mismatch = int((np.asarray(y[name]) != np.asarray(i63_y[name])).sum()) if len(y[name]) == len(i63_y[name]) else -1
        alignment[name] = {
            "rows_yixi": int(n), "rows_iter63": int(n_i63),
            "row_count_match": bool(n == n_i63),
            "user_id_mismatches_trusted_arrays": uid_mismatch,
            "video_id_mismatches_excluding_known_nan": vid_mismatch_raw - nan_video_yixi,
            "label_mismatches": label_mismatch,
        }
        print(f"  [{name}] rows={n}  user_id_mismatches={uid_mismatch}  "
              f"video_id_mismatches(excl nan)={vid_mismatch_raw - nan_video_yixi}  label_mismatches={label_mismatch}", flush=True)
    results["row_alignment_check"] = alignment
    save_results(results)
    aligned_ok = all(
        alignment[name]["row_count_match"]
        and alignment[name]["user_id_mismatches_trusted_arrays"] == 0
        and alignment[name]["video_id_mismatches_excluding_known_nan"] == 0
        and alignment[name]["label_mismatches"] == 0
        for name in ("valid", "test")
    )
    print(f"  row alignment OK = {aligned_ok}", flush=True)
    if not aligned_ok:
        raise AssertionError("row alignment check failed")

    print("\n  training iter63 rate_only GBM at 5 seeds, sigmoid-mean ensemble", flush=True)
    i63_valid_seed_scores = [m5.sigmoid(i63_model0.predict(i63_dfs["valid"]))]
    i63_test_seed_scores = [m5.sigmoid(i63_model0.predict(i63_dfs["test"]))]
    for seed in m5.ITER63_SEEDS[1:]:
        mdl, va, te, _ = iter63_train.run(DATA_DIR, "rate_only", seed=seed, _cache=i63_cache)
        i63_valid_seed_scores.append(m5.sigmoid(mdl.predict(i63_dfs["valid"])))
        i63_test_seed_scores.append(m5.sigmoid(mdl.predict(i63_dfs["test"])))
        print(f"    seed={seed}  valid={va['primary']:.6f}  test={te['primary']:.6f}", flush=True)
    i63_valid_ens = np.mean(np.stack(i63_valid_seed_scores), axis=0)
    i63_test_ens = np.mean(np.stack(i63_test_seed_scores), axis=0)
    i63_valid_ens_metrics = evaluate(i63_u["valid"], i63_y["valid"], i63_valid_ens)
    i63_test_ens_metrics = evaluate(i63_u["test"], i63_y["test"], i63_test_ens)
    ens_valid_delta = i63_valid_ens_metrics["primary"] - ITER63_5SEED_ENSEMBLE_VALID
    ens_test_delta = i63_test_ens_metrics["primary"] - ITER63_5SEED_ENSEMBLE_TEST
    harness_c = {
        "reference_valid": ITER63_5SEED_ENSEMBLE_VALID,
        "reference_test": ITER63_5SEED_ENSEMBLE_TEST,
        "reproduced_valid": i63_valid_ens_metrics["primary"],
        "reproduced_test": i63_test_ens_metrics["primary"],
        "delta_valid": ens_valid_delta,
        "delta_test": ens_test_delta,
        "pass_1e-6": bool(abs(ens_valid_delta) <= HARNESS_TOLERANCE and abs(ens_test_delta) <= HARNESS_TOLERANCE),
    }
    results["harness_fidelity_iter63_5seed_ensemble"] = harness_c
    print(f"  iter63 5-seed ensemble: valid={i63_valid_ens_metrics['primary']:.8f} (ref {ITER63_5SEED_ENSEMBLE_VALID:.8f}) "
          f"test={i63_test_ens_metrics['primary']:.8f} (ref {ITER63_5SEED_ENSEMBLE_TEST:.8f})  "
          f"pass={harness_c['pass_1e-6']}", flush=True)
    save_results(results)
    if not harness_c["pass_1e-6"]:
        print("!!! HARNESS FIDELITY C (iter63 5-seed ensemble) FAILED -- stopping !!!", flush=True)
        return

    valid_components_4 = dict(valid_components_3)
    valid_components_4["i63"] = m5.within_user_percentile(i63_valid_ens, i63_u["valid"])
    test_components_4 = dict(test_components_3)
    test_components_4["i63"] = m5.within_user_percentile(i63_test_ens, i63_u["test"])

    # Sanity check: raw 4-model blend at iterMERGE5's best weights should
    # reproduce iterMERGE5's exact numbers.
    check_valid = m5.blend4(valid_components_4, MERGE5_BEST_WEIGHTS, users["valid"], y["valid"])
    check_test = m5.blend4(test_components_4, MERGE5_BEST_WEIGHTS, users["test"], y["test"])
    merge5_sanity = {
        "reproduced_valid": check_valid["primary"], "claimed_valid": MERGE5_BEST_VALID,
        "delta_valid": check_valid["primary"] - MERGE5_BEST_VALID,
        "reproduced_test": check_test["primary"], "claimed_test": MERGE5_BEST_TEST,
        "delta_test": check_test["primary"] - MERGE5_BEST_TEST,
        "pass_1e-6": bool(abs(check_valid["primary"] - MERGE5_BEST_VALID) <= HARNESS_TOLERANCE
                           and abs(check_test["primary"] - MERGE5_BEST_TEST) <= HARNESS_TOLERANCE),
    }
    results["harness_fidelity_merge5_raw4model_sanity"] = merge5_sanity
    print(f"  MERGE5 raw-4model-blend sanity: valid={check_valid['primary']:.8f} (claimed {MERGE5_BEST_VALID}) "
          f"test={check_test['primary']:.8f} (claimed {MERGE5_BEST_TEST}) pass={merge5_sanity['pass_1e-6']}", flush=True)
    save_results(results)
    if not merge5_sanity["pass_1e-6"]:
        print("!!! MERGE5 SANITY CHECK FAILED -- stopping !!!", flush=True)
        return

    print("\n=== HARNESS FIDELITY FULLY CONFIRMED (7 checks, all exact) -- proceeding to CV-regularized search ===\n", flush=True)

    # =================================================================
    print("=== [4/8] building 5-fold user-level CV fold masks over valid ===", flush=True)
    users_valid_arr = np.asarray(users["valid"])
    y_valid_arr = np.asarray(y["valid"])
    fold_masks = make_cv_fold_masks(users_valid_arr, N_FOLDS, CV_RANDOM_STATE)
    fold_sizes = [int(m.sum()) for m in fold_masks]
    fold_users = [int(len(np.unique(users_valid_arr[m]))) for m in fold_masks]
    print(f"  fold row sizes = {fold_sizes}  (sum={sum(fold_sizes)}, total rows={len(users_valid_arr)})", flush=True)
    print(f"  fold user counts = {fold_users}  (sum={sum(fold_users)}, total users={len(np.unique(users_valid_arr))})", flush=True)
    results["cv_folds"] = {
        "n_folds": N_FOLDS, "random_state": CV_RANDOM_STATE,
        "fold_row_sizes": fold_sizes, "fold_user_counts": fold_users,
        "total_rows": int(len(users_valid_arr)), "total_users": int(len(np.unique(users_valid_arr))),
    }
    save_results(results)

    def score_point(w):
        """Return (single_split_valid_primary, cv_mean_valid_primary, fold_primaries, test_primary)."""
        vm = m5.blend4(valid_components_4, w, users["valid"], y["valid"])
        tm = m5.blend4(test_components_4, w, users["test"], y["test"])
        cv_mean, fold_primaries = cv_mean_valid(valid_components_4, w, users_valid_arr, y_valid_arr, fold_masks)
        return vm["primary"], cv_mean, fold_primaries, tm["primary"]

    # =================================================================
    print("\n=== [5/8] CV mean valid AT iterMERGE5's original best point ===", flush=True)
    m5point_single, m5point_cv, m5point_folds, m5point_test = score_point(MERGE5_BEST_WEIGHTS)
    results["merge5_point_cv_check"] = {
        "weights": MERGE5_BEST_WEIGHTS,
        "single_split_valid": m5point_single,
        "cv_mean_valid": m5point_cv,
        "cv_fold_valid_primaries": m5point_folds,
        "test": m5point_test,
    }
    print(f"  iterMERGE5 point weights = {MERGE5_BEST_WEIGHTS}", flush=True)
    print(f"  single-split valid = {m5point_single:.8f}  (claimed {MERGE5_BEST_VALID})", flush=True)
    print(f"  CV mean valid      = {m5point_cv:.8f}  (fold primaries: {[f'{p:.6f}' for p in m5point_folds]})", flush=True)
    print(f"  test (record only) = {m5point_test:.8f}  (claimed {MERGE5_BEST_TEST})", flush=True)
    save_results(results)

    # =================================================================
    print("\n=== [6/8] coarse sweep over i63 weight, scored by CV mean valid ===", flush=True)
    coarse_grid = []
    base_fm, base_lgb, base_xgb = m5.PRODUCTION_WEIGHTS["fm"], m5.PRODUCTION_WEIGHTS["lgb"], m5.PRODUCTION_WEIGHTS["xgb"]
    for w_i63 in (0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30):
        scale = 1.0 - w_i63
        w = {"fm": base_fm * scale, "lgb": base_lgb * scale, "xgb": base_xgb * scale, "i63": w_i63}
        single, cv_mean, fold_primaries, test_p = score_point(w)
        coarse_grid.append({
            "weights": dict(w), "single_split_valid": single, "cv_mean_valid": cv_mean,
            "cv_fold_valid_primaries": fold_primaries, "test": test_p,
        })
        print(f"  coarse w_i63={w_i63:.2f}  single_split_valid={single:.8f}  cv_mean_valid={cv_mean:.8f}", flush=True)
    results["weight_search_coarse"] = coarse_grid
    save_results(results)

    best_coarse = max(coarse_grid, key=lambda r: r["cv_mean_valid"])
    print(f"  best coarse point (by CV mean): {best_coarse['weights']}  cv_mean_valid={best_coarse['cv_mean_valid']:.8f}", flush=True)

    # =================================================================
    print("\n=== [7/8] local fine grid refinement around best coarse point, scored by CV mean valid ===", flush=True)
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
                single, cv_mean, fold_primaries, test_p = score_point(w)
                fine_grid.append({
                    "weights": dict(w), "single_split_valid": single, "cv_mean_valid": cv_mean,
                    "cv_fold_valid_primaries": fold_primaries, "test": test_p,
                })
    fine_grid.sort(key=lambda r: -r["cv_mean_valid"])
    print(f"  fine grid: {len(fine_grid)} combos evaluated", flush=True)
    for r in fine_grid[:5]:
        print(f"    cv_mean_valid={r['cv_mean_valid']:.8f}  single_split_valid={r['single_split_valid']:.8f}  "
              f"test={r['test']:.8f}  weights={r['weights']}", flush=True)
    results["weight_search_fine"] = fine_grid[:30]  # keep top 30 (by CV mean) for the record
    save_results(results)

    cv_selected = fine_grid[0]
    # Also compute, for the record, what the single-split-valid-selected point
    # within this same fine grid would have been (i.e. reproduce iterMERGE5's
    # selection criterion on this identical grid, as a direct side-by-side).
    single_split_selected = max(fine_grid, key=lambda r: r["single_split_valid"])

    # =================================================================
    print("\n=== [8/8] verdict ===", flush=True)
    cv_selected_single = cv_selected["single_split_valid"]
    cv_selected_cv = cv_selected["cv_mean_valid"]
    cv_selected_test = cv_selected["test"]
    delta_single_vs_ref = cv_selected_single - blend3_valid["primary"]
    delta_test_vs_ref = cv_selected_test - blend3_test["primary"]
    same_point_as_merge5 = all(
        abs(cv_selected["weights"][k] - MERGE5_BEST_WEIGHTS[k]) <= 1e-9 for k in MERGE5_BEST_WEIGHTS
    )

    if delta_single_vs_ref >= PROMOTION_DELTA and cv_selected_test >= blend3_test["primary"]:
        verdict = "PROMOTE-candidate"
    elif delta_single_vs_ref >= PRELIMINARY_DELTA and cv_selected_test >= blend3_test["primary"]:
        verdict = "PRELIMINARY (crossover resolved)"
    elif delta_single_vs_ref >= PRELIMINARY_DELTA:
        verdict = "PRELIMINARY (crossover NOT resolved -- same pattern as iterMERGE5)"
    else:
        verdict = "REJECT"

    results["summary"] = {
        "reference_3model_blend_valid": blend3_valid["primary"],
        "reference_3model_blend_test": blend3_test["primary"],
        "merge5_original_point": {
            "weights": MERGE5_BEST_WEIGHTS,
            "single_split_valid": m5point_single,
            "cv_mean_valid": m5point_cv,
            "test": m5point_test,
        },
        "cv_selected_point": {
            "weights": cv_selected["weights"],
            "cv_mean_valid": cv_selected_cv,
            "single_split_valid": cv_selected_single,
            "test_for_the_record_only": cv_selected_test,
        },
        "single_split_selected_point_same_grid_for_comparison": {
            "weights": single_split_selected["weights"],
            "single_split_valid": single_split_selected["single_split_valid"],
            "cv_mean_valid": single_split_selected["cv_mean_valid"],
            "test_for_the_record_only": single_split_selected["test"],
        },
        "cv_selection_same_point_as_merge5_original": same_point_as_merge5,
        "delta_single_split_valid_vs_3model_reference": delta_single_vs_ref,
        "delta_test_vs_3model_reference": delta_test_vs_ref,
        "crossover_resolved": bool(cv_selected_test >= blend3_test["primary"]),
        "preliminary_delta_threshold": PRELIMINARY_DELTA,
        "promotion_delta_threshold": PROMOTION_DELTA,
        "verdict": verdict,
        "selection_basis": (
            "Weight point selected by highest 5-fold user-level CV MEAN valid "
            "score (not single-split valid). Single-split valid and test "
            "reported for comparability/record only, never used for selection."
        ),
    }
    print(f"  3-model reference valid          = {blend3_valid['primary']:.8f}", flush=True)
    print(f"  3-model reference test           = {blend3_test['primary']:.8f}", flush=True)
    print(f"  iterMERGE5 original point: single-split={m5point_single:.8f}  cv_mean={m5point_cv:.8f}  test={m5point_test:.8f}", flush=True)
    print(f"  CV-selected point weights        = {cv_selected['weights']}", flush=True)
    print(f"  CV-selected point: cv_mean_valid = {cv_selected_cv:.8f}", flush=True)
    print(f"  CV-selected point: single-split  = {cv_selected_single:.8f}  (delta vs ref = {delta_single_vs_ref:+.8f})", flush=True)
    print(f"  CV-selected point: test (record) = {cv_selected_test:.8f}  (delta vs ref = {delta_test_vs_ref:+.8f})", flush=True)
    print(f"  same point as iterMERGE5 original? {same_point_as_merge5}", flush=True)
    print(f"  crossover resolved (test >= 3model ref test)? {cv_selected_test >= blend3_test['primary']}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)

    results["environment"] = {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    }
    results["elapsed_seconds"] = time.time() - t0
    save_results(results)
    print(f"\n=== done in {results['elapsed_seconds']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
