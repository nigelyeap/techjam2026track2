"""iterMERGE6: does isotonic-regression calibration (iter66_calibrated_blend's
method) applied per-component to the iterMERGE5 4-model (FM/LGB/XGB/i63)
blend produce a blend that is BOTH a genuine valid improvement AND doesn't
regress test -- unlike iterMERGE5's raw weight-search blend, which found a
valid-only gain (+0.00038397) with a consistent test regression (-0.0004 to
-0.0009)?

Strong prior going in (must be confronted empirically, not assumed away):
iter66_calibrated_blend/RESULT.md already found isotonic calibration of
iter63's OWN rate_only GBM raw score collapses 122,613 unique raw valid
scores into only 37 distinct calibrated levels -- a step-function artifact
that destroyed within-user ranking (GBM standalone valid 0.67168 -> 0.54189,
blend 0.67606 -> 0.63877). That was on iter63's small num_leaves=2 GBM
against a 2-model alpha=0.14/global-minmax blend. This experiment asks
whether the same collapse mechanism reproduces for the 4 different,
generally richer components (FM, yixi's LightGBM, yixi's XGBoost, i63) used
in the current within-user-percentile production blend, or whether it
happens to behave differently for at least one of them.

Pipeline (mirrors iterMERGE5_four_model_blend/run.py's already-verified
harness -- imported as a module and reused directly, not re-derived):
  1. Harness-fidelity: reproduce the 3-model reference (LGB/XGB/FM valid,
     blend valid/test) and iter63's own rate_only standalone valid
     (code-verified reference 0.6716787219047546 -- NOT the dispatch-note
     trap constant 0.6768913269042969, per iterMERGE5/RESULT.md).
  2. Row-alignment check (reused from iterMERGE5, trusted uncast arrays).
  3. Train all 4 components exactly as iterMERGE5 does (LGB/XGB seed 0,
     FM 5-seed sigmoid-mean ensemble, i63 5-seed sigmoid-mean ensemble),
     but ALSO capture each component's TRAIN-split raw score (iterMERGE5
     only needed valid/test).
  4. Per-component isotonic calibration: fit sklearn IsotonicRegression
     (out_of_bounds='clip') on (component_train_raw_score, y_train) --
     exactly iter66_calibrated_blend/run_isotonic.py's method (fit on
     TRAIN only, apply to valid/test; train is fully held out from
     valid/test so there is no leakage into either evaluation split or
     into the weight-search step that follows). Record unique-level counts
     on valid/test per component (iter66's diagnostic for the collapse
     failure mode).
  5. Two blending variants of the calibrated scores, both weight-searched
     on VALID ONLY with a coarse-then-fine grid around iterMERGE5's found
     optimum (fm=0.072/lgb=0.5184/xgb=0.3296/i63=0.08) as a starting
     region, but searched independently in calibrated-space:
       (a) primary: calibrated score -> within-user-percentile rank (same
           normalization convention as the current production blend) ->
           weighted sum.
       (b) secondary/robustness: calibrated score used directly (already
           label-probability-scaled by isotonic) -> weighted sum, no
           percentile step -- mirrors iter66_calibrated_blend's own
           original substitution design.
  6. Verdict vs PRELIMINARY_DELTA=0.0003 / PROMOTION_DELTA=0.001 on valid;
     test reported for the record only, never for selection, then checked
     post hoc for the specific two-split-consistency question this
     experiment exists to answer.
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
from sklearn.isotonic import IsotonicRegression

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
from evaluate import evaluate  # noqa: E402

MERGE5_DIR = os.path.join(REPO_ROOT, "experiments", "iterMERGE5_four_model_blend")
YIXI10_DIR = os.path.join(REPO_ROOT, "experiments", "iterYIXI10_video_metadata")
ITER63_DIR = os.path.join(REPO_ROOT, "experiments", "iter63_decay_tab_rate")
MAKE_SUBMISSION_PATH = os.path.join(REPO_ROOT, "make_submission.py")
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
RESULTS_PATH = os.path.join(THIS_DIR, "results.json")

FM_SEEDS = (0, 1, 2, 3, 4)
ITER63_SEEDS = (0, 1, 2, 3, 4)
TRAIN_SEED = 0

PRODUCTION_WEIGHTS_3 = {"fm": 0.10, "lgb": 0.52, "xgb": 0.38}
CLAIMED_VALID_3 = 0.69943440
CLAIMED_TEST_3 = 0.68432260
HARNESS_TOLERANCE = 1e-6

ITER63_RATE_ONLY_REFERENCE_VALID = 0.6716787219047546
ITER63_HARNESS_TOLERANCE = 1e-6

MERGE5_BEST_WEIGHTS = {"fm": 0.072, "lgb": 0.5184, "xgb": 0.3296, "i63": 0.08}
MERGE5_BEST_VALID = 0.69981837
MERGE5_BEST_TEST = 0.68367088

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


def blend(components, weights, users_arr, y_arr):
    b = sum(weights[k] * components[k] for k in weights)
    return evaluate(users_arr, y_arr, b)


def weight_search(components_valid, components_test, users, y, base_weights, label):
    """Coarse-then-fine grid search on VALID only, mirroring iterMERGE5's
    pattern, starting from base_weights as the search center."""

    def eval_w(w):
        vm = blend(components_valid, w, users["valid"], y["valid"])
        tm = blend(components_test, w, users["test"], y["test"])
        return vm, tm

    keys = list(base_weights.keys())
    # Coarse: sweep i63 weight, rescale the rest proportionally from base.
    coarse_grid = []
    base_i63 = base_weights["i63"]
    rest_keys = [k for k in keys if k != "i63"]
    rest_total = sum(base_weights[k] for k in rest_keys)
    for w_i63 in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.30):
        scale = (1.0 - w_i63) / rest_total if rest_total > 0 else 0.0
        w = {k: base_weights[k] * scale for k in rest_keys}
        w["i63"] = w_i63
        vm, tm = eval_w(w)
        coarse_grid.append({"weights": dict(w), "valid_primary": vm["primary"], "test_primary": tm["primary"]})
        print(f"    [{label}] coarse w_i63={w_i63:.2f}  valid={vm['primary']:.8f}", flush=True)
    best_coarse = max(coarse_grid, key=lambda r: r["valid_primary"])
    print(f"    [{label}] best coarse: {best_coarse['weights']}  valid={best_coarse['valid_primary']:.8f}", flush=True)

    # Fine: local grid refinement around best coarse point (and also around
    # base_weights itself, since base_weights is the trusted MERGE5 raw
    # optimum -- calibrated space might prefer either neighborhood).
    fine_grid = []
    steps = (-0.06, -0.03, 0.0, 0.03, 0.06)
    seen = set()
    for center in (best_coarse["weights"], base_weights):
        bw = center
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
                    vm, tm = eval_w(w)
                    fine_grid.append({"weights": dict(w), "valid_primary": vm["primary"], "test_primary": tm["primary"]})
    fine_grid.sort(key=lambda r: -r["valid_primary"])
    print(f"    [{label}] fine grid: {len(fine_grid)} combos, best valid={fine_grid[0]['valid_primary']:.8f} "
          f"test={fine_grid[0]['test_primary']:.8f}  weights={fine_grid[0]['weights']}", flush=True)
    return {"coarse": coarse_grid, "fine_top30": fine_grid[:30], "best": fine_grid[0]}


def main():
    t0 = time.time()
    results = {"experiment": "iterMERGE6_calibrated_4model"}

    print("=== [1/8] importing iterMERGE5's verified harness code as a module ===", flush=True)
    m5 = load_module(os.path.join(MERGE5_DIR, "run.py"), "merge6_import_merge5")
    print("  imported: fit_lgb, fit_xgb, sigmoid, within_user_percentile, stable_user_order, "
          "LGB_CANDIDATE_COLUMNS, XGB_COLUMNS", flush=True)

    print("\n=== [2/8] loading yixi's YIXI10 feature frames ===", flush=True)
    features_mod = load_module(os.path.join(YIXI10_DIR, "features.py"), "merge6_yixi10_features")
    frames, y, users, _meta = features_mod.load_frames()
    print(f"  rows train/valid/test = {len(frames['train'])}/{len(frames['valid'])}/{len(frames['test'])}", flush=True)

    print("\n=== [3/8] harness-fidelity part A: LGB/XGB (seed 0) + FM (5-seed ensemble) ===", flush=True)
    lgb_cand_model = m5.fit_lgb(frames, y, users, m5.LGB_CANDIDATE_COLUMNS, TRAIN_SEED)
    xgb_model = m5.fit_xgb(frames, y, users, m5.XGB_COLUMNS, TRAIN_SEED)
    lgb_cand_train = lgb_cand_model.predict(frames["train"][m5.LGB_CANDIDATE_COLUMNS])
    lgb_cand_valid = lgb_cand_model.predict(frames["valid"][m5.LGB_CANDIDATE_COLUMNS])
    lgb_cand_test = lgb_cand_model.predict(frames["test"][m5.LGB_CANDIDATE_COLUMNS])
    xgb_train = xgb_model.predict(frames["train"][m5.XGB_COLUMNS])
    xgb_valid = xgb_model.predict(frames["valid"][m5.XGB_COLUMNS])
    xgb_test = xgb_model.predict(frames["test"][m5.XGB_COLUMNS])
    lgb_valid_metrics = evaluate(users["valid"], y["valid"], lgb_cand_valid)
    xgb_valid_metrics = evaluate(users["valid"], y["valid"], xgb_valid)
    print(f"  LGB candidate valid primary = {lgb_valid_metrics['primary']:.8f}  (ref 0.68834144)", flush=True)
    print(f"  XGB valid primary           = {xgb_valid_metrics['primary']:.8f}  (ref 0.66755420)", flush=True)

    submission = load_module(MAKE_SUBMISSION_PATH, "merge6_submission")
    splits = submission.load_ext(DATA_DIR, halflives=submission.HALFLIVES, tab_halflives=submission.TAB_HALFLIVES)
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

    fm_train_scores, fm_valid_scores, fm_test_scores = [], [], []
    for seed in FM_SEEDS:
        print(f"  fitting FM seed={seed}", flush=True)
        model = submission.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits["train"], dim, seed)
        fm_train_scores.append(m5.sigmoid(model.predict(Xtr)))
        fm_valid_scores.append(m5.sigmoid(model.predict(Xva)))
        fm_test_scores.append(m5.sigmoid(model.predict(Xte)))
    fm_train = np.mean(np.stack(fm_train_scores), axis=0)
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
    blend3_valid = blend(valid_components_3, PRODUCTION_WEIGHTS_3, users["valid"], y["valid"])
    blend3_test = blend(test_components_3, PRODUCTION_WEIGHTS_3, users["test"], y["test"])

    harness_a = {
        "lgb_candidate_valid": lgb_valid_metrics["primary"],
        "xgb_valid": xgb_valid_metrics["primary"],
        "fm_valid": fm_valid_metrics["primary"],
        "blend_valid": blend3_valid["primary"],
        "blend_test": blend3_test["primary"],
        "claimed": {
            "lgb_candidate_valid": 0.68834144, "xgb_valid": 0.66755420,
            "fm_valid": 0.63987792, "blend_valid": CLAIMED_VALID_3, "blend_test": CLAIMED_TEST_3,
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

    print("\n=== [4/8] harness-fidelity part B: iter63's own rate_only GBM standalone ===", flush=True)
    iter63_train = load_module(os.path.join(ITER63_DIR, "train.py"), "merge6_iter63_train")
    i63_model0, i63_va0, i63_te0, i63_cache = iter63_train.run(DATA_DIR, "rate_only", seed=TRAIN_SEED, verbose=True)
    i63_delta = i63_va0["primary"] - ITER63_RATE_ONLY_REFERENCE_VALID
    harness_b = {
        "reference_valid_code_verified": ITER63_RATE_ONLY_REFERENCE_VALID,
        "reference_valid_dispatch_note_trap_constant": 0.6768913269042969,
        "reproduced_valid": i63_va0["primary"],
        "reproduced_test": i63_te0["primary"],
        "delta_vs_code_verified_reference": i63_delta,
        "pass_1e-6": bool(abs(i63_delta) <= ITER63_HARNESS_TOLERANCE),
    }
    results["harness_fidelity_iter63_standalone"] = harness_b
    print(f"  iter63 rate_only seed0 valid = {i63_va0['primary']:.10f}  "
          f"(code-verified ref {ITER63_RATE_ONLY_REFERENCE_VALID:.10f})  pass={harness_b['pass_1e-6']}", flush=True)
    save_results(results)
    if not harness_b["pass_1e-6"]:
        print("!!! HARNESS FIDELITY B FAILED -- stopping !!!", flush=True)
        return

    print("\n=== [5/8] row-alignment check (reused method: trusted uncast arrays) ===", flush=True)
    i63_dfs, i63_y, i63_u = i63_cache
    i63_splits_raw = iter63_train._de.load_ext(DATA_DIR, use_cache=True)
    i63_video_raw = {
        name: np.array([x[iter63_train._de.IDX["video_id"]] for x in i63_splits_raw[name]], dtype=str)
        for name in ("train", "valid", "test")
    }
    alignment = {}
    for name in ("train", "valid", "test"):
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
            "rows_yixi": int(n), "rows_iter63": int(n_i63), "row_count_match": bool(n == n_i63),
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
        for name in ("train", "valid", "test")
    )
    print(f"  row alignment OK on all 3 splits (position-based combine safe) = {aligned_ok}", flush=True)
    if not aligned_ok:
        raise AssertionError("row alignment check failed -- cannot safely combine scores position-wise")

    print("\n=== [6/8] training iter63 rate_only GBM at 5 seeds (sigmoid-mean ensemble, incl. TRAIN scores) ===", flush=True)
    i63_train_seed_scores = [m5.sigmoid(i63_model0.predict(i63_dfs["train"]))]
    i63_valid_seed_scores = [m5.sigmoid(i63_model0.predict(i63_dfs["valid"]))]
    i63_test_seed_scores = [m5.sigmoid(i63_model0.predict(i63_dfs["test"]))]
    per_seed = {0: {"valid_primary": i63_va0["primary"], "test_primary": i63_te0["primary"]}}
    for seed in ITER63_SEEDS[1:]:
        print(f"  fitting iter63 rate_only seed={seed}", flush=True)
        mdl, va, te, _ = iter63_train.run(DATA_DIR, "rate_only", seed=seed, _cache=i63_cache)
        i63_train_seed_scores.append(m5.sigmoid(mdl.predict(i63_dfs["train"])))
        i63_valid_seed_scores.append(m5.sigmoid(mdl.predict(i63_dfs["valid"])))
        i63_test_seed_scores.append(m5.sigmoid(mdl.predict(i63_dfs["test"])))
        per_seed[seed] = {"valid_primary": va["primary"], "test_primary": te["primary"]}
        print(f"    seed={seed}  valid={va['primary']:.6f}  test={te['primary']:.6f}", flush=True)

    i63_train_ens = np.mean(np.stack(i63_train_seed_scores), axis=0)
    i63_valid_ens = np.mean(np.stack(i63_valid_seed_scores), axis=0)
    i63_test_ens = np.mean(np.stack(i63_test_seed_scores), axis=0)
    i63_valid_ens_metrics = evaluate(i63_u["valid"], i63_y["valid"], i63_valid_ens)
    i63_test_ens_metrics = evaluate(i63_u["test"], i63_y["test"], i63_test_ens)
    results["iter63_5seed_ensemble"] = {
        "per_seed": per_seed,
        "ensemble_valid_primary": i63_valid_ens_metrics["primary"],
        "ensemble_test_primary": i63_test_ens_metrics["primary"],
    }
    print(f"  iter63 5-seed ensemble: valid={i63_valid_ens_metrics['primary']:.8f} test={i63_test_ens_metrics['primary']:.8f}", flush=True)
    save_results(results)

    # Uncalibrated (raw) 4-model reference point, for a clean before/after
    # comparison in the same run (reproduces iterMERGE5's best point).
    raw_valid_components_4 = dict(valid_components_3, i63=m5.within_user_percentile(i63_valid_ens, i63_u["valid"]))
    raw_test_components_4 = dict(test_components_3, i63=m5.within_user_percentile(i63_test_ens, i63_u["test"]))
    raw_merge5_point_valid = blend(raw_valid_components_4, MERGE5_BEST_WEIGHTS, users["valid"], y["valid"])
    raw_merge5_point_test = blend(raw_test_components_4, MERGE5_BEST_WEIGHTS, users["test"], y["test"])
    results["raw_4model_at_merge5_best_weights"] = {
        "valid_primary": raw_merge5_point_valid["primary"], "test_primary": raw_merge5_point_test["primary"],
        "reference_from_merge5_RESULT_md": {"valid": MERGE5_BEST_VALID, "test": MERGE5_BEST_TEST},
        "delta_valid": raw_merge5_point_valid["primary"] - MERGE5_BEST_VALID,
        "delta_test": raw_merge5_point_test["primary"] - MERGE5_BEST_TEST,
    }
    print(f"  sanity: raw 4-model @ MERGE5 best weights -> valid={raw_merge5_point_valid['primary']:.8f} "
          f"(ref {MERGE5_BEST_VALID:.8f})  test={raw_merge5_point_test['primary']:.8f} (ref {MERGE5_BEST_TEST:.8f})", flush=True)
    save_results(results)

    # =================================================================
    print("\n=== [7/8] isotonic calibration per component (fit on TRAIN, apply to valid/test) ===", flush=True)
    y_train = np.asarray(y["train"])
    raw_train = {"fm": fm_train, "lgb": lgb_cand_train, "xgb": xgb_train, "i63": i63_train_ens}
    raw_valid = {"fm": fm_valid, "lgb": lgb_cand_valid, "xgb": xgb_valid, "i63": i63_valid_ens}
    raw_test = {"fm": fm_test, "lgb": lgb_cand_test, "xgb": xgb_test, "i63": i63_test_ens}

    iso_models = {}
    calibration_diag = {}
    calibrated_valid_raw = {}
    calibrated_test_raw = {}
    for name in ("fm", "lgb", "xgb", "i63"):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_train[name], y_train)
        iso_models[name] = iso
        cal_va = iso.predict(raw_valid[name])
        cal_te = iso.predict(raw_test[name])
        calibrated_valid_raw[name] = cal_va
        calibrated_test_raw[name] = cal_te
        standalone_raw_va = evaluate(users["valid"], y["valid"], raw_valid[name])["primary"]
        standalone_cal_va = evaluate(users["valid"], y["valid"], cal_va)["primary"]
        standalone_raw_te = evaluate(users["test"], y["test"], raw_test[name])["primary"]
        standalone_cal_te = evaluate(users["test"], y["test"], cal_te)["primary"]
        calibration_diag[name] = {
            "unique_raw_train": int(len(np.unique(raw_train[name]))),
            "unique_raw_valid": int(len(np.unique(raw_valid[name]))),
            "unique_calibrated_valid": int(len(np.unique(cal_va))),
            "unique_calibrated_test": int(len(np.unique(cal_te))),
            "standalone_valid_raw": standalone_raw_va,
            "standalone_valid_calibrated": standalone_cal_va,
            "standalone_valid_delta": standalone_cal_va - standalone_raw_va,
            "standalone_test_raw": standalone_raw_te,
            "standalone_test_calibrated": standalone_cal_te,
            "standalone_test_delta": standalone_cal_te - standalone_raw_te,
        }
        print(f"  [{name}] unique raw(train/valid)={calibration_diag[name]['unique_raw_train']}/"
              f"{calibration_diag[name]['unique_raw_valid']}  unique calibrated(valid/test)="
              f"{calibration_diag[name]['unique_calibrated_valid']}/{calibration_diag[name]['unique_calibrated_test']}  "
              f"standalone valid raw->cal: {standalone_raw_va:.5f} -> {standalone_cal_va:.5f} "
              f"({calibration_diag[name]['standalone_valid_delta']:+.5f})", flush=True)
    results["calibration_fit"] = calibration_diag
    save_results(results)

    # =================================================================
    print("\n=== [8/8] weight search: calibrated 4-way blend (VALID ONLY) ===", flush=True)

    # Variant (a): calibrated score -> within-user-percentile -> weighted sum
    # (matches the current production blend's normalization convention).
    cal_pct_valid = {name: m5.within_user_percentile(calibrated_valid_raw[name], users["valid"]) for name in raw_train}
    cal_pct_test = {name: m5.within_user_percentile(calibrated_test_raw[name], users["test"]) for name in raw_train}
    search_a = weight_search(cal_pct_valid, cal_pct_test, users, y, MERGE5_BEST_WEIGHTS, "variant-a-percentile")

    # Variant (b): calibrated score used directly (already ~[0,1] probability
    # scale from isotonic), no percentile step -- mirrors iter66's original
    # substitution design as a robustness check.
    search_b = weight_search(calibrated_valid_raw, calibrated_test_raw, users, y, MERGE5_BEST_WEIGHTS, "variant-b-direct")

    results["weight_search_variant_a_percentile"] = search_a
    results["weight_search_variant_b_direct"] = search_b
    save_results(results)

    best_a = search_a["best"]
    best_b = search_b["best"]
    overall_best = best_a if best_a["valid_primary"] >= best_b["valid_primary"] else best_b
    overall_variant = "a_percentile" if overall_best is best_a else "b_direct"

    ref_valid = blend3_valid["primary"]
    ref_test = blend3_test["primary"]
    best_delta_valid = overall_best["valid_primary"] - ref_valid
    best_delta_test = overall_best["test_primary"] - ref_test

    if best_delta_valid >= PROMOTION_DELTA:
        verdict = "PROMOTE-candidate" if best_delta_test >= 0 else "PROMOTE-candidate-but-test-regresses"
    elif best_delta_valid >= PRELIMINARY_DELTA:
        verdict = "PRELIMINARY" if best_delta_test >= 0 else "PRELIMINARY-but-test-regresses"
    else:
        verdict = "REJECT"

    summary = {
        "reference_3model_blend_valid": ref_valid,
        "reference_3model_blend_test": ref_test,
        "merge5_raw_4model_best_valid": MERGE5_BEST_VALID,
        "merge5_raw_4model_best_test": MERGE5_BEST_TEST,
        "best_overall_variant": overall_variant,
        "best_overall_weights": overall_best["weights"],
        "best_overall_valid": overall_best["valid_primary"],
        "best_overall_test": overall_best["test_primary"],
        "best_delta_vs_3model_valid": best_delta_valid,
        "best_delta_vs_3model_test": best_delta_test,
        "does_it_beat_valid_only_gain_threshold": bool(best_delta_valid >= PRELIMINARY_DELTA),
        "does_it_also_avoid_test_regression": bool(best_delta_test >= 0),
        "two_split_consistent_gain": bool(best_delta_valid >= PRELIMINARY_DELTA and best_delta_test >= 0),
        "preliminary_delta_threshold": PRELIMINARY_DELTA,
        "promotion_delta_threshold": PROMOTION_DELTA,
        "verdict": verdict,
        "selection_basis": "valid-only for weight search; test reported for the record and for the two-split-consistency question this experiment targets, never used for weight selection",
    }
    results["verdict"] = summary
    print(f"\n  3-model reference valid/test   = {ref_valid:.8f} / {ref_test:.8f}", flush=True)
    print(f"  best calibrated 4-model        = {overall_best['valid_primary']:.8f} / {overall_best['test_primary']:.8f}  "
          f"(variant={overall_variant}, weights={overall_best['weights']})", flush=True)
    print(f"  delta vs 3-model ref            = valid {best_delta_valid:+.8f}  test {best_delta_test:+.8f}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)

    results["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    results["elapsed_seconds"] = time.time() - t0
    save_results(results)
    print(f"\n=== done in {results['elapsed_seconds']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
