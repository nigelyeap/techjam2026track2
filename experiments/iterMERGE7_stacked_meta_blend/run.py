"""iterMERGE7: stacked nonlinear meta-learner over the 4 within-user-
percentile component scores (FM, LGB, XGB, i63), replacing iterMERGE5's
fixed-weight linear blend.

Every merge round so far (2-6) has only tried LINEAR combinations of
component scores: fixed-weight weighted sums (raw, grid-searched, or
per-component isotonic-recalibrated before a linear sum). This round tests
a genuinely nonlinear meta-learner that can capture interactions between
components (e.g. "when FM and i63 disagree strongly, trust LGB more").

Pipeline:
  1. Harness-fidelity check: reuse iterMERGE5_four_model_blend/run.py as an
     imported module directly (its verified fit_lgb/fit_xgb/sigmoid/
     within_user_percentile/stable_user_order/column lists/row-alignment
     method) -- do NOT re-derive any of this from scratch. Reproduce the
     3-model reference, iter63 rate_only standalone, and the raw 4-model
     blend at iterMERGE5's best weights, as a sanity check that this
     pipeline agrees with iterMERGE5's before introducing anything new.
  2. K-fold (5-fold) stacking on valid: for each fold, fit a meta-learner
     on the OTHER folds' (4 percentile-normalized component scores, label)
     pairs, predict on the held-out fold. This produces out-of-fold (OOF)
     meta-predictions covering all of valid -- evaluate the primary metric
     on these OOF predictions as the valid-only selection number (never
     the in-sample-fit valid, which would be optimistic).
     Try:
       (a) shallow, heavily-regularized logistic regression (few free
           params relative to 4 input features -- low overfit risk).
       (b) optionally a shallow GBM (2-4 leaves) if time allows.
  3. Compare OOF valid vs iterMERGE5's linear optimum (0.69981837) and vs
     the 3-model reference (0.69943440).
  4. For the final reported model: refit the winning meta-learner type on
     ALL of valid (once satisfied by the OOF check this isn't trivially
     overfit), apply to test, report test for the record only (never for
     selection).
"""
from __future__ import annotations

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

import importlib.util


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse iterMERGE5's verified harness directly -- fit_lgb, fit_xgb, sigmoid,
# within_user_percentile, stable_user_order, LGB_CANDIDATE_COLUMNS,
# XGB_COLUMNS, PRODUCTION_WEIGHTS, etc.
m5 = load_module(os.path.join(MERGE5_DIR, "run.py"), "merge7_import_merge5")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
import lightgbm as lgb  # noqa: E402

RESULTS_PATH = os.path.join(THIS_DIR, "results.json")

CLAIMED_VALID_3MODEL = 0.69943440
CLAIMED_TEST_3MODEL = 0.68432260
MERGE5_BEST_WEIGHTS = {"fm": 0.072, "lgb": 0.5184, "xgb": 0.3296, "i63": 0.08}
MERGE5_BEST_VALID = 0.69981837
MERGE5_BEST_TEST = 0.68367088
HARNESS_TOLERANCE = 1e-6

PRELIMINARY_DELTA = 0.0003
PROMOTION_DELTA = 0.001
N_FOLDS = 5
RANDOM_STATE = 0


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


COMPONENT_KEYS = ["fm", "lgb", "xgb", "i63"]


def stack_features(components_dict):
    """components_dict: {key: 1D array of within-user-percentile scores}."""
    return np.column_stack([components_dict[k] for k in COMPONENT_KEYS])


def fit_logreg(X, y, C):
    model = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    model.fit(X, y)
    return model


def oof_logreg_predict(X, y, users, n_folds, C, seed):
    """K-fold OOF meta-predictions using a fold split at the USER level
    (not row level) so that no user's rows leak between train/holdout
    folds of the meta-learner -- consistent with the within-user-ranking
    nature of the primary metric."""
    users_arr = np.asarray(users)
    uniq_users = np.unique(users_arr)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    fold_id = np.full(len(y), -1, dtype=np.int64)
    for i, (train_uidx, holdout_uidx) in enumerate(kf.split(uniq_users)):
        train_users = set(uniq_users[train_uidx])
        holdout_mask = ~np.isin(users_arr, list(train_users))
        train_mask = ~holdout_mask
        model = fit_logreg(X[train_mask], y[train_mask], C)
        oof[holdout_mask] = model.predict_proba(X[holdout_mask])[:, 1]
        fold_id[holdout_mask] = i
    assert np.all(fold_id >= 0)
    return oof, fold_id


def oof_gbm_predict(X, y, users, n_folds, num_leaves, seed):
    users_arr = np.asarray(users)
    uniq_users = np.unique(users_arr)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    cols = COMPONENT_KEYS
    Xdf = pd.DataFrame(X, columns=cols)
    for i, (train_uidx, holdout_uidx) in enumerate(kf.split(uniq_users)):
        train_users = set(uniq_users[train_uidx])
        holdout_mask = ~np.isin(users_arr, list(train_users))
        train_mask = ~holdout_mask
        model = lgb.LGBMClassifier(
            num_leaves=num_leaves, learning_rate=0.05, n_estimators=200,
            min_child_samples=200, reg_lambda=1.0, subsample=0.8,
            colsample_bytree=1.0, verbosity=-1, n_jobs=-1, random_state=seed,
        )
        model.fit(Xdf[train_mask], y[train_mask])
        oof[holdout_mask] = model.predict_proba(Xdf[holdout_mask])[:, 1]
    return oof


def main():
    t0 = time.time()
    results = {"experiment": "iterMERGE7_stacked_meta_blend"}

    # =================================================================
    print("=== [1/6] loading yixi's YIXI10 feature frames (via iterMERGE5 module) ===", flush=True)
    features_mod = m5.load_module(os.path.join(m5.YIXI10_DIR, "features.py"), "merge7_yixi10_features")
    frames, y, users, _meta = features_mod.load_frames()
    missing = [c for c in m5.LGB_CANDIDATE_COLUMNS if c not in frames["train"].columns]
    if missing:
        raise AssertionError(f"missing candidate LGB columns: {missing}")
    print(f"  rows train/valid/test = {len(frames['train'])}/{len(frames['valid'])}/{len(frames['test'])}", flush=True)

    print("=== [2/6] harness-fidelity: LGB/XGB (seed 0) + FM (5-seed ensemble) ===", flush=True)
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

    submission = m5.load_module(m5.MAKE_SUBMISSION_PATH, "merge7_submission")
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
    print("\n=== [3/6] harness-fidelity: iter63 rate_only standalone + row alignment ===", flush=True)
    iter63_train = m5.load_module(os.path.join(m5.ITER63_DIR, "train.py"), "merge7_iter63_train")
    DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
    i63_model0, i63_va0, i63_te0, i63_cache = iter63_train.run(DATA_DIR, "rate_only", seed=m5.TRAIN_SEED, verbose=True)
    i63_delta = i63_va0["primary"] - m5.ITER63_RATE_ONLY_REFERENCE_VALID
    harness_b = {
        "reference_valid_code_verified": m5.ITER63_RATE_ONLY_REFERENCE_VALID,
        "note": "0.6716787219047546 is the CORRECT iter63 rate_only reference (NOT 0.6768913269042969, "
                "which is a different yixi-chain constant -- see iterMERGE5/RESULT.md).",
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
    print(f"  iter63 5-seed ensemble: valid={i63_valid_ens_metrics['primary']:.8f} test={i63_test_ens_metrics['primary']:.8f}", flush=True)

    # NOTE: iter63's own u/y arrays (i63_u/i63_y) were already verified 0
    # mismatches vs. yixi's users/y arrays above -- safe to use yixi's
    # users["valid"]/y["valid"] as the canonical arrays throughout since
    # they're position-identical.
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
          f"pass={merge5_sanity['pass_1e-6']}", flush=True)
    save_results(results)
    if not merge5_sanity["pass_1e-6"]:
        print("!!! MERGE5 SANITY CHECK FAILED -- stopping !!!", flush=True)
        return

    print("\n=== HARNESS FIDELITY FULLY CONFIRMED -- proceeding to meta-learner stacking ===\n", flush=True)

    # =================================================================
    print("=== [4/6] k-fold OOF stacking on valid ===", flush=True)
    Xstack_valid = stack_features(valid_components_4)
    ystack_valid = np.asarray(y["valid"])
    users_valid = np.asarray(users["valid"])

    oof_results = []

    # --- (a) logistic regression, sweep regularization strength C ---
    print("  (a) logistic regression OOF sweep over C", flush=True)
    for C in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        oof_pred, fold_id = oof_logreg_predict(Xstack_valid, ystack_valid, users_valid, N_FOLDS, C, RANDOM_STATE)
        m = evaluate(users_valid, ystack_valid, oof_pred)
        rec = {"method": "logreg", "C": C, "oof_valid_primary": m["primary"]}
        oof_results.append(rec)
        print(f"    C={C:<8} oof_valid_primary={m['primary']:.8f}", flush=True)
    results["oof_stacking_logreg_sweep"] = oof_results
    save_results(results)

    best_logreg = max([r for r in oof_results if r["method"] == "logreg"], key=lambda r: r["oof_valid_primary"])
    print(f"  best logreg: C={best_logreg['C']}  oof_valid={best_logreg['oof_valid_primary']:.8f}", flush=True)

    # --- (b) shallow GBM (2-4 leaves), sweep num_leaves ---
    print("  (b) shallow GBM OOF sweep over num_leaves", flush=True)
    gbm_oof_results = []
    for num_leaves in (2, 3, 4):
        oof_pred = oof_gbm_predict(Xstack_valid, ystack_valid, users_valid, N_FOLDS, num_leaves, RANDOM_STATE)
        m = evaluate(users_valid, ystack_valid, oof_pred)
        rec = {"method": "gbm", "num_leaves": num_leaves, "oof_valid_primary": m["primary"]}
        gbm_oof_results.append(rec)
        print(f"    num_leaves={num_leaves}  oof_valid_primary={m['primary']:.8f}", flush=True)
    results["oof_stacking_gbm_sweep"] = gbm_oof_results
    save_results(results)

    best_gbm = max(gbm_oof_results, key=lambda r: r["oof_valid_primary"])
    print(f"  best gbm: num_leaves={best_gbm['num_leaves']}  oof_valid={best_gbm['oof_valid_primary']:.8f}", flush=True)

    all_oof = oof_results + gbm_oof_results
    best_overall = max(all_oof, key=lambda r: r["oof_valid_primary"])
    results["oof_stacking_results"] = {
        "best_logreg": best_logreg, "best_gbm": best_gbm, "best_overall": best_overall,
        "reference_3model_valid": blend3_valid["primary"],
        "merge5_linear_optimum_valid": MERGE5_BEST_VALID,
        "delta_vs_3model": best_overall["oof_valid_primary"] - blend3_valid["primary"],
        "delta_vs_merge5_linear": best_overall["oof_valid_primary"] - MERGE5_BEST_VALID,
    }
    print(f"\n  BEST OVERALL OOF: {best_overall}", flush=True)
    print(f"  delta vs 3-model reference ({blend3_valid['primary']:.8f}) = "
          f"{best_overall['oof_valid_primary'] - blend3_valid['primary']:+.8f}", flush=True)
    print(f"  delta vs MERGE5 linear optimum ({MERGE5_BEST_VALID:.8f}) = "
          f"{best_overall['oof_valid_primary'] - MERGE5_BEST_VALID:+.8f}", flush=True)
    save_results(results)

    # =================================================================
    print("\n=== [5/6] refit winning meta-learner on ALL of valid, apply to test ===", flush=True)
    Xstack_test = stack_features(test_components_4)
    ystack_test = np.asarray(y["test"])
    users_test = np.asarray(users["test"])

    if best_overall["method"] == "logreg":
        final_model = fit_logreg(Xstack_valid, ystack_valid, best_overall["C"])
        final_valid_insample = final_model.predict_proba(Xstack_valid)[:, 1]
        final_test = final_model.predict_proba(Xstack_test)[:, 1]
        final_desc = {"method": "logreg", "C": best_overall["C"], "coef": final_model.coef_.tolist(), "intercept": final_model.intercept_.tolist()}
    else:
        Xdf_valid = pd.DataFrame(Xstack_valid, columns=COMPONENT_KEYS)
        Xdf_test = pd.DataFrame(Xstack_test, columns=COMPONENT_KEYS)
        final_model = lgb.LGBMClassifier(
            num_leaves=best_overall["num_leaves"], learning_rate=0.05, n_estimators=200,
            min_child_samples=200, reg_lambda=1.0, subsample=0.8, colsample_bytree=1.0,
            verbosity=-1, n_jobs=-1, random_state=RANDOM_STATE,
        )
        final_model.fit(Xdf_valid, ystack_valid)
        final_valid_insample = final_model.predict_proba(Xdf_valid)[:, 1]
        final_test = final_model.predict_proba(Xdf_test)[:, 1]
        final_desc = {"method": "gbm", "num_leaves": best_overall["num_leaves"],
                      "feature_importances": dict(zip(COMPONENT_KEYS, final_model.feature_importances_.tolist()))}

    m_valid_insample = evaluate(users_valid, ystack_valid, final_valid_insample)
    m_test = evaluate(users_test, ystack_test, final_test)
    refit_results = {
        "final_model": final_desc,
        "refit_all_valid_insample_primary": m_valid_insample["primary"],  # optimistic, in-sample -- NOT the selection number
        "oof_valid_primary_selection_number": best_overall["oof_valid_primary"],
        "test_primary_for_the_record_only": m_test["primary"],
        "reference_3model_test": blend3_test["primary"],
        "merge5_linear_optimum_test": MERGE5_BEST_TEST,
        "delta_test_vs_3model_reference": m_test["primary"] - blend3_test["primary"],
        "delta_test_vs_merge5_linear": m_test["primary"] - MERGE5_BEST_TEST,
    }
    results["refit_and_test"] = refit_results
    print(f"  refit-on-all-valid in-sample primary (optimistic, NOT selection) = {m_valid_insample['primary']:.8f}", flush=True)
    print(f"  OOF valid primary (the actual selection number)                 = {best_overall['oof_valid_primary']:.8f}", flush=True)
    print(f"  test primary (for the record only, never selection)             = {m_test['primary']:.8f}", flush=True)
    print(f"  delta test vs 3-model reference ({blend3_test['primary']:.8f}) = "
          f"{m_test['primary'] - blend3_test['primary']:+.8f}", flush=True)
    print(f"  delta test vs MERGE5 linear optimum ({MERGE5_BEST_TEST:.8f}) = "
          f"{m_test['primary'] - MERGE5_BEST_TEST:+.8f}", flush=True)
    save_results(results)

    # =================================================================
    print("\n=== [6/6] verdict ===", flush=True)
    oof_valid = best_overall["oof_valid_primary"]
    delta_vs_3model = oof_valid - blend3_valid["primary"]
    delta_vs_merge5 = oof_valid - MERGE5_BEST_VALID

    if delta_vs_3model >= PROMOTION_DELTA:
        verdict = "PROMOTE-candidate"
    elif delta_vs_3model >= PRELIMINARY_DELTA:
        verdict = "PRELIMINARY"
    else:
        verdict = "REJECT"

    results["summary"] = {
        "reference_3model_blend_valid": blend3_valid["primary"],
        "reference_3model_blend_test": blend3_test["primary"],
        "merge5_linear_optimum_valid": MERGE5_BEST_VALID,
        "merge5_linear_optimum_test": MERGE5_BEST_TEST,
        "best_meta_learner": best_overall,
        "oof_stacking_valid_primary": oof_valid,
        "delta_vs_3model_reference": delta_vs_3model,
        "delta_vs_merge5_linear_optimum": delta_vs_merge5,
        "beats_merge5_linear": bool(delta_vs_merge5 > 0),
        "test_primary_for_the_record": m_test["primary"],
        "closes_valid_test_crossover": bool(m_test["primary"] >= blend3_test["primary"]),
        "preliminary_delta_threshold": PRELIMINARY_DELTA,
        "promotion_delta_threshold": PROMOTION_DELTA,
        "verdict": verdict,
        "selection_basis": "K-FOLD OOF valid-only; in-sample refit valid and test reported for the record only, never for selection",
    }
    print(f"  3-model reference valid   = {blend3_valid['primary']:.8f}", flush=True)
    print(f"  MERGE5 linear optimum     = {MERGE5_BEST_VALID:.8f}", flush=True)
    print(f"  best meta-learner OOF     = {oof_valid:.8f}  (delta vs 3model={delta_vs_3model:+.8f}, vs merge5={delta_vs_merge5:+.8f})", flush=True)
    print(f"  test (record only)        = {m_test['primary']:.8f}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)

    results["environment"] = {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "lightgbm": lgb.__version__,
    }
    results["elapsed_seconds"] = time.time() - t0
    save_results(results)
    print(f"\n=== done in {results['elapsed_seconds']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
