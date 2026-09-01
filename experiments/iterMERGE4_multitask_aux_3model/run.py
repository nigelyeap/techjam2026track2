"""iterMERGE4: re-test iter67's multitask aux-label stacking idea (predicted
is_like/is_follow/is_comment/is_forward as extra GBM input columns) against
yixi's richer LightGBM (LGB_CANDIDATE_COLUMNS) and XGBoost (XGB_COLUMNS)
reference feature sets from iterMERGE1, instead of iter67's original 2-model
iter63 rate_only harness.

Pipeline:
  1. Harness-fidelity: reproduce iterMERGE1_verify_yixi10's 4 reference
     numbers exactly (LGB valid=0.68834144, XGB valid=0.66755420,
     FM valid=0.63987792, blend valid=0.69943440, test=0.68432260), same
     code path (yixi's own features.py chain via load_frames(), independent
     from-scratch LGB/XGB/FM training, make_submission.py's train_one_fm).
  2. Recover is_like/is_follow/is_comment/is_forward per row for yixi's
     train/valid/test frames, reusing iter67's aux_labels.py raw-CSV
     recovery logic. Alignment is re-derived and re-verified independently
     here (yixi's frames do not expose orig_idx themselves) by calling
     iterYIXI9_watch_depth_history/features.py's own load_raw_rows(), whose
     `rows` list -- split by the same `lengths` dict, same order -- is the
     exact row sequence yixi's frames are built from (asserted internally
     by her own build_frames()). orig_idx there is defined identically to
     iter63's (a running counter over the same two raw files, same
     file-then-file order, incremented before any date filtering).
  3. Train 4 LGBMClassifier aux predictors (mirroring iter67's config
     exactly: n_estimators=100, num_leaves=31, learning_rate=0.1,
     random_state=0) via 5-fold OOF on train (KFold(5, shuffle=True,
     random_state=0)) + a separate full-train fit for valid/test, on the
     UNION of yixi's LGB_CANDIDATE_COLUMNS and XGB_COLUMNS (one shared set
     of aux predictor columns usable by both downstream models, since the
     task is to add the same aux signal to both column sets).
  4. Re-train LightGBM (LGB_CANDIDATE_COLUMNS + aux) and XGBoost
     (XGB_COLUMNS + aux) at seed 0, individually and combined; re-run the
     3-way within-user-percentile blend at 10/52/38 (FM reused unchanged
     from step 1 -- aux columns never touch the FM component).
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
from sklearn.model_selection import KFold


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402

YIXI10_DIR = os.path.join(REPO_ROOT, "experiments", "iterYIXI10_video_metadata")
YIXI9_DIR = os.path.join(REPO_ROOT, "experiments", "iterYIXI9_watch_depth_history")
YIXI5_RESULTS_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "results.json"
)
ITER67_DIR = os.path.join(REPO_ROOT, "experiments", "iter67_multitask_gbm")
MAKE_SUBMISSION_PATH = os.path.join(REPO_ROOT, "make_submission.py")
RESULTS_PATH = os.path.join(THIS_DIR, "results.json")

FM_SEEDS = (0, 1, 2, 3, 4)
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
UNION_COLUMNS = list(dict.fromkeys(LGB_CANDIDATE_COLUMNS + XGB_COLUMNS))

LGB_CONFIG = dict(
    objective="lambdarank", lambdarank_truncation_level=50, sigmoid=2.0,
    metric="ndcg", eval_at=[5], num_leaves=2, learning_rate=0.10,
    n_estimators=500, min_child_samples=200, reg_lambda=1.0,
    verbosity=-1, n_jobs=-1, linear_tree=True,
)

WEIGHTS = {"fm": 0.10, "lgb": 0.52, "xgb": 0.38}
CLAIMED_VALID = 0.69943440
CLAIMED_TEST = 0.68432260
HARNESS_TOLERANCE = 1e-6

AUX_TARGETS = ("is_like", "is_follow", "is_comment", "is_forward")
AUX_COLS = [f"aux_{t}" for t in AUX_TARGETS]
N_FOLDS = 5

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
        eval_set=[(Xva, y["valid"][valid_order])], eval_group=[valid_groups],
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


def blend(components_valid, components_test, weights, users, y):
    bv = sum(weights[k] * components_valid[k] for k in weights)
    bt = sum(weights[k] * components_test[k] for k in weights)
    return evaluate(users["valid"], y["valid"], bv), evaluate(users["test"], y["test"], bt)


def main():
    t0 = time.time()
    results = {"experiment": "iterMERGE4_multitask_aux_3model"}

    # ---------------------------------------------------------------
    print("=== [1/6] loading yixi's YIXI10 feature frames ===", flush=True)
    features_mod = load_module(os.path.join(YIXI10_DIR, "features.py"), "merge4_yixi10_features")
    frames, y, users, _meta = features_mod.load_frames()
    missing = [c for c in UNION_COLUMNS if c not in frames["train"].columns]
    if missing:
        raise AssertionError(f"missing candidate columns: {missing}")
    print(f"  rows train/valid/test = {len(frames['train'])}/{len(frames['valid'])}/{len(frames['test'])}", flush=True)

    print("=== [2/6] harness-fidelity: LGB/XGB (seed 0) + FM (5-seed ensemble) ===", flush=True)
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

    submission = load_module(MAKE_SUBMISSION_PATH, "merge4_submission")
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
        fm_valid_scores.append(submission.sigmoid(model.predict(Xva)))
        fm_test_scores.append(submission.sigmoid(model.predict(Xte)))
    fm_valid = np.mean(np.stack(fm_valid_scores), axis=0)
    fm_test = np.mean(np.stack(fm_test_scores), axis=0)
    fm_valid_metrics = evaluate(users["valid"], y["valid"], fm_valid)
    print(f"  FM ensemble valid primary   = {fm_valid_metrics['primary']:.8f}  (ref 0.63987792)", flush=True)

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
    baseline_valid_metrics, baseline_test_metrics = blend(valid_components, test_components, WEIGHTS, users, y)

    harness = {
        "lgb_candidate_valid": lgb_valid_metrics["primary"],
        "xgb_valid": xgb_valid_metrics["primary"],
        "fm_valid": fm_valid_metrics["primary"],
        "blend_valid": baseline_valid_metrics["primary"],
        "blend_test": baseline_test_metrics["primary"],
        "claimed": {
            "lgb_candidate_valid": 0.68834144, "xgb_valid": 0.66755420,
            "fm_valid": 0.63987792, "blend_valid": CLAIMED_VALID, "blend_test": CLAIMED_TEST,
        },
    }
    harness["deltas"] = {k: harness[k] - harness["claimed"][k] for k in harness["claimed"]}
    harness["all_pass_1e-6"] = bool(all(abs(v) <= HARNESS_TOLERANCE for v in harness["deltas"].values()))
    results["harness_fidelity"] = harness
    print(f"  harness all_pass_1e-6 = {harness['all_pass_1e-6']}", flush=True)
    print(f"  deltas = {harness['deltas']}", flush=True)
    save_results(results)  # write immediately per instructions
    if not harness["all_pass_1e-6"]:
        print("!!! HARNESS FIDELITY FAILED -- stopping before aux work !!!", flush=True)
        return

    # ---------------------------------------------------------------
    print("=== [3/6] recovering aux engagement labels + verifying orig_idx alignment ===", flush=True)
    yixi9_features = load_module(os.path.join(YIXI9_DIR, "features.py"), "merge4_yixi9_features")
    rows, lengths, _distribution = yixi9_features.load_raw_rows(
        os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
    )
    ORIG = yixi9_features.ORIG
    origidx = {}
    start = 0
    for name in ("train", "valid", "test"):
        stop = start + lengths[name]
        origidx[name] = np.array([r[ORIG] for r in rows[start:stop]])
        start = stop
    for name in ("train", "valid", "test"):
        if lengths[name] != len(frames[name]):
            raise AssertionError(f"row-count mismatch for {name}: raw={lengths[name]} frames={len(frames[name])}")

    aux_labels_mod = load_module(os.path.join(ITER67_DIR, "aux_labels.py"), "merge4_aux_labels")
    aux_full = aux_labels_mod.load_aux_labels(os.path.join(REPO_ROOT, "KuaiRand-Pure", "data"))

    alignment = {}
    for name in ("train", "valid", "test"):
        recon = aux_full["long_view"][origidx[name]]
        trusted = y[name].astype(np.int8)
        n_mismatch = int((recon != trusted).sum())
        alignment[name] = {"rows": int(len(recon)), "mismatches": n_mismatch}
        print(f"  alignment [{name}]: rows={len(recon)} mismatches={n_mismatch}", flush=True)
        if n_mismatch != 0:
            raise AssertionError(f"orig_idx alignment broken for {name}: {n_mismatch} mismatches")
    print("  PASS: orig_idx alignment exact on all splits (0 mismatches)", flush=True)
    results["alignment_check"] = alignment
    save_results(results)

    aux = {name: {tgt: aux_full[tgt][origidx[name]] for tgt in AUX_TARGETS} for name in ("train", "valid", "test")}
    prevalence = {tgt: float(aux["train"][tgt].mean()) for tgt in AUX_TARGETS}
    print(f"  train prevalence: {prevalence}", flush=True)
    results["aux_prevalence_train"] = prevalence

    # ---------------------------------------------------------------
    print(f"=== [4/6] training 4 aux LGBMClassifiers (mirrors iter67: n_estimators=100, "
          f"num_leaves=31, lr=0.1, {N_FOLDS}-fold OOF) on UNION_COLUMNS ===", flush=True)
    Xtr_full = frames["train"][UNION_COLUMNS]
    Xva_full = frames["valid"][UNION_COLUMNS]
    Xte_full = frames["test"][UNION_COLUMNS]
    oof_cols = {tgt: np.zeros(len(Xtr_full), dtype=np.float64) for tgt in AUX_TARGETS}
    va_cols = {tgt: np.zeros(len(Xva_full), dtype=np.float64) for tgt in AUX_TARGETS}
    te_cols = {tgt: np.zeros(len(Xte_full), dtype=np.float64) for tgt in AUX_TARGETS}

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    aux_diag = {}
    for tgt in AUX_TARGETS:
        ytgt = aux["train"][tgt].astype(np.float32)
        for fold, (tr_idx, ho_idx) in enumerate(kf.split(Xtr_full)):
            clf = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.1,
                                      random_state=0, verbosity=-1, n_jobs=-1)
            clf.fit(Xtr_full.iloc[tr_idx], ytgt[tr_idx])
            oof_cols[tgt][ho_idx] = clf.predict_proba(Xtr_full.iloc[ho_idx])[:, 1]
        clf_full = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.1,
                                       random_state=0, verbosity=-1, n_jobs=-1)
        clf_full.fit(Xtr_full, ytgt)
        va_cols[tgt] = clf_full.predict_proba(Xva_full)[:, 1]
        te_cols[tgt] = clf_full.predict_proba(Xte_full)[:, 1]
        aux_diag[tgt] = {"oof_mean_pred": float(oof_cols[tgt].mean()), "actual_rate": float(ytgt.mean())}
        print(f"  {tgt}: OOF mean_pred={oof_cols[tgt].mean():.4f} actual_rate={ytgt.mean():.4f}", flush=True)
    results["aux_classifier_diagnostics"] = aux_diag
    save_results(results)

    for tgt in AUX_TARGETS:
        frames["train"][f"aux_{tgt}"] = oof_cols[tgt]
        frames["valid"][f"aux_{tgt}"] = va_cols[tgt]
        frames["test"][f"aux_{tgt}"] = te_cols[tgt]

    # ---------------------------------------------------------------
    print("=== [5/6] retraining LGB+aux and XGB+aux (seed 0), individual + combined ablations ===", flush=True)
    lgb_aux_model = fit_lgb(frames, y, users, LGB_CANDIDATE_COLUMNS + AUX_COLS, TRAIN_SEED)
    xgb_aux_model = fit_xgb(frames, y, users, XGB_COLUMNS + AUX_COLS, TRAIN_SEED)
    lgb_aux_valid = lgb_aux_model.predict(frames["valid"][LGB_CANDIDATE_COLUMNS + AUX_COLS])
    lgb_aux_test = lgb_aux_model.predict(frames["test"][LGB_CANDIDATE_COLUMNS + AUX_COLS])
    xgb_aux_valid = xgb_aux_model.predict(frames["valid"][XGB_COLUMNS + AUX_COLS])
    xgb_aux_test = xgb_aux_model.predict(frames["test"][XGB_COLUMNS + AUX_COLS])

    lgb_aux_valid_metrics = evaluate(users["valid"], y["valid"], lgb_aux_valid)
    xgb_aux_valid_metrics = evaluate(users["valid"], y["valid"], xgb_aux_valid)
    lgb_aux_test_metrics = evaluate(users["test"], y["test"], lgb_aux_test)
    xgb_aux_test_metrics = evaluate(users["test"], y["test"], xgb_aux_test)

    lgb_importance = dict(zip(lgb_aux_model.feature_name_, [int(v) for v in lgb_aux_model.feature_importances_]))
    xgb_booster = xgb_aux_model.get_booster()
    xgb_score = xgb_booster.get_score(importance_type="weight")

    standalone = {
        "lgb_baseline_valid": lgb_valid_metrics["primary"],
        "lgb_aux_valid": lgb_aux_valid_metrics["primary"],
        "lgb_aux_delta": lgb_aux_valid_metrics["primary"] - lgb_valid_metrics["primary"],
        "xgb_baseline_valid": xgb_valid_metrics["primary"],
        "xgb_aux_valid": xgb_aux_valid_metrics["primary"],
        "xgb_aux_delta": xgb_aux_valid_metrics["primary"] - xgb_valid_metrics["primary"],
        "lgb_aux_feature_importance": {c: lgb_importance.get(c, 0) for c in AUX_COLS},
        "xgb_aux_feature_importance": {c: xgb_score.get(c, 0) for c in AUX_COLS},
    }
    standalone["lgb_baseline_test"] = evaluate(users["test"], y["test"], lgb_cand_test)["primary"]
    standalone["lgb_aux_test"] = lgb_aux_test_metrics["primary"]
    standalone["xgb_baseline_test"] = evaluate(users["test"], y["test"], xgb_test)["primary"]
    standalone["xgb_aux_test"] = xgb_aux_test_metrics["primary"]
    print(f"  LGB: baseline={standalone['lgb_baseline_valid']:.8f} +aux={standalone['lgb_aux_valid']:.8f} "
          f"delta={standalone['lgb_aux_delta']:+.8f}", flush=True)
    print(f"  XGB: baseline={standalone['xgb_baseline_valid']:.8f} +aux={standalone['xgb_aux_valid']:.8f} "
          f"delta={standalone['xgb_aux_delta']:+.8f}", flush=True)
    print(f"  LGB aux importances: {standalone['lgb_aux_feature_importance']}", flush=True)
    print(f"  XGB aux importances: {standalone['xgb_aux_feature_importance']}", flush=True)
    results["standalone_ablation"] = standalone
    save_results(results)

    # Combined blend variants (FM component reused unchanged from harness-fidelity step)
    variants = {
        "baseline_baseline": (lgb_cand_valid, lgb_cand_test, xgb_valid, xgb_test),
        "lgbaux_baseline": (lgb_aux_valid, lgb_aux_test, xgb_valid, xgb_test),
        "baseline_xgbaux": (lgb_cand_valid, lgb_cand_test, xgb_aux_valid, xgb_aux_test),
        "lgbaux_xgbaux": (lgb_aux_valid, lgb_aux_test, xgb_aux_valid, xgb_aux_test),
    }
    blend_results = {}
    for name, (lv, lt, xv, xt) in variants.items():
        vc = {
            "fm": within_user_percentile(fm_valid, users["valid"]),
            "lgb": within_user_percentile(lv, users["valid"]),
            "xgb": within_user_percentile(xv, users["valid"]),
        }
        tc = {
            "fm": within_user_percentile(fm_test, users["test"]),
            "lgb": within_user_percentile(lt, users["test"]),
            "xgb": within_user_percentile(xt, users["test"]),
        }
        vm, tm = blend(vc, tc, WEIGHTS, users, y)
        blend_results[name] = {
            "valid_primary": vm["primary"], "test_primary": tm["primary"],
            "valid_delta_vs_baseline": vm["primary"] - baseline_valid_metrics["primary"],
        }
        print(f"  blend[{name}]: valid={vm['primary']:.8f} (delta={blend_results[name]['valid_delta_vs_baseline']:+.8f}) "
              f"test={tm['primary']:.8f}", flush=True)

    results["blend_baseline"] = {"valid_primary": baseline_valid_metrics["primary"], "test_primary": baseline_test_metrics["primary"]}
    results["blend_ablation"] = blend_results

    best_variant = max(blend_results, key=lambda k: blend_results[k]["valid_primary"])
    best_delta = blend_results[best_variant]["valid_delta_vs_baseline"]
    if best_variant == "baseline_baseline":
        best_delta = 0.0
    if best_delta >= PROMOTION_DELTA:
        verdict = "PROMOTE"
    elif best_delta >= PRELIMINARY_DELTA:
        verdict = "PRELIMINARY"
    else:
        verdict = "REJECT"
    results["verdict"] = {
        "decision": verdict, "best_variant": best_variant, "best_valid_delta": best_delta,
        "preliminary_delta_threshold": PRELIMINARY_DELTA, "promotion_delta_threshold": PROMOTION_DELTA,
        "selection_basis": "valid-only; test reported for the record only",
    }
    print(f"\n=== VERDICT: {verdict} (best={best_variant}, delta={best_delta:+.8f}) ===", flush=True)

    results["environment"] = {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "lightgbm": lgb.__version__, "xgboost": xgb.__version__,
    }
    results["elapsed_seconds"] = time.time() - t0
    save_results(results)
    print(f"\n=== [6/6] done in {results['elapsed_seconds']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
