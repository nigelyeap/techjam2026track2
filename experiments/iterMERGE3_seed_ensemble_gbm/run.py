"""iterMERGE3_seed_ensemble_gbm

Currently only the FM component of the production blend (root make_submission.py)
is a 5-seed sigmoid-mean ensemble; LightGBM and XGBoost are each trained at a
single seed (seed 0) only. This experiment tests whether seed-ensembling LGB
and XGB (seeds 0-4 each, same feature sets/configs, just varying random_state)
reduces variance and improves the 3-way within-user-percentile blend.

Step 1 (mandatory): harness-fidelity check against the 4 reference numbers,
using seed 0 of the same LGB/XGB fits this script trains anyway, plus the
unchanged FM 5-seed ensemble -- exact same code path as iterMERGE1/verify.py
and root make_submission.py (both loaded here as modules, functions reused
verbatim, no reimplementation).

Step 2: train LGB seeds 0-4 and XGB seeds 0-4 (FM ensemble trained once,
unchanged). For each model, compare single-seed(0), plain-mean-of-5-seeds,
and sigmoid-mean-of-5-seeds on standalone valid/test.

Step 3: re-blend at the production 10/52/38 weights using the seed-ensembled
LGB/XGB (best-performing ensembling convention per model) + unchanged FM
ensemble, then a small local grid search around 10/52/38 on VALID ONLY.

Results appended to results.json incrementally (each stage overwrites/adds
its own key and is flushed to disk immediately after that stage completes).
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
MAKE_SUBMISSION_PATH = os.path.join(REPO_ROOT, "make_submission.py")
RESULTS_PATH = os.path.join(THIS_DIR, "results.json")

SEEDS = (0, 1, 2, 3, 4)
REFERENCE = {
    "lgb_valid": 0.68834144,
    "xgb_valid": 0.66755420,
    "fm_valid": 0.63987792,
    "blend_valid": 0.69943440,
    "blend_test": 0.68432260,
}
TOLERANCE = 1e-6
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


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"experiment": "iterMERGE3_seed_ensemble_gbm"}


def save_stage(key, payload):
    results = load_results()
    results[key] = jsonable(payload)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  [saved results.json stage: {key}]", flush=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def main():
    t0 = time.time()
    print("=== [1/6] loading YIXI10 feature frames ===", flush=True)
    features_mod = load_module(os.path.join(YIXI10_DIR, "features.py"), "merge3_features")
    frames, y, users, _meta = features_mod.load_frames()
    print(
        f"  train/valid/test rows = {len(frames['train'])}/{len(frames['valid'])}/{len(frames['test'])}"
        f"  ({time.time()-t0:.0f}s)",
        flush=True,
    )

    print("=== [2/6] loading make_submission.py as module (reuse _fit_lgb/_fit_xgb/train_one_fm) ===", flush=True)
    submission = load_module(MAKE_SUBMISSION_PATH, "merge3_submission")
    LGB_CANDIDATE_COLUMNS = submission.LGB_CANDIDATE_COLUMNS
    XGB_COLUMNS = submission.XGB_COLUMNS
    WEIGHTS = dict(submission.WEIGHTS)

    print("=== [3/6] training LightGBM seeds 0-4 ===", flush=True)
    lgb_valid_seed, lgb_test_seed = {}, {}
    for seed in SEEDS:
        ts = time.time()
        model = submission._fit_lgb(frames, y, users, LGB_CANDIDATE_COLUMNS, seed)
        lgb_valid_seed[seed] = model.predict(frames["valid"][LGB_CANDIDATE_COLUMNS])
        lgb_test_seed[seed] = model.predict(frames["test"][LGB_CANDIDATE_COLUMNS])
        va = evaluate(users["valid"], y["valid"], lgb_valid_seed[seed])["primary"]
        print(f"  LGB seed={seed} valid={va:.8f}  ({time.time()-ts:.0f}s)", flush=True)

    print("=== [4/6] training XGBoost seeds 0-4 ===", flush=True)
    xgb_valid_seed, xgb_test_seed = {}, {}
    for seed in SEEDS:
        ts = time.time()
        model = submission._fit_xgb(frames, y, users, XGB_COLUMNS, seed)
        xgb_valid_seed[seed] = model.predict(frames["valid"][XGB_COLUMNS])
        xgb_test_seed[seed] = model.predict(frames["test"][XGB_COLUMNS])
        va = evaluate(users["valid"], y["valid"], xgb_valid_seed[seed])["primary"]
        print(f"  XGB seed={seed} valid={va:.8f}  ({time.time()-ts:.0f}s)", flush=True)

    print("=== [5/6] training FM 5-seed ensemble (unchanged production protocol) ===", flush=True)
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
    assert np.array_equal(np.asarray(yva), y["valid"]), "FM/native valid label mismatch"
    assert np.array_equal(np.asarray(uva), np.asarray(users["valid"])), "FM/native valid user mismatch"
    assert np.array_equal(np.asarray(yte), y["test"]), "FM/native test label mismatch"
    assert np.array_equal(np.asarray(ute), np.asarray(users["test"])), "FM/native test user mismatch"

    fm_valid_seed_scores, fm_test_seed_scores = [], []
    for seed in SEEDS:
        ts = time.time()
        model = submission.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits["train"], dim, seed)
        fm_valid_seed_scores.append(sigmoid(model.predict(Xva)))
        fm_test_seed_scores.append(sigmoid(model.predict(Xte)))
        print(f"  FM seed={seed} done  ({time.time()-ts:.0f}s)", flush=True)
    fm_valid = np.mean(np.stack(fm_valid_seed_scores), axis=0)
    fm_test = np.mean(np.stack(fm_test_seed_scores), axis=0)
    fm_valid_metric = evaluate(users["valid"], y["valid"], fm_valid)
    fm_test_metric = evaluate(users["test"], y["test"], fm_test)
    print(f"  FM ensemble valid primary = {fm_valid_metric['primary']:.8f}", flush=True)

    # ---- Stage A: harness-fidelity check (seed 0 of the above + FM ensemble) ----
    print("=== [6a/6] harness-fidelity check ===", flush=True)

    def wup(scores, uid):
        return submission.within_user_percentile(scores, uid)

    va0 = {
        "fm": wup(fm_valid, users["valid"]),
        "lgb": wup(lgb_valid_seed[0], users["valid"]),
        "xgb": wup(xgb_valid_seed[0], users["valid"]),
    }
    te0 = {
        "fm": wup(fm_test, users["test"]),
        "lgb": wup(lgb_test_seed[0], users["test"]),
        "xgb": wup(xgb_test_seed[0], users["test"]),
    }
    blend_valid0 = sum(WEIGHTS[k] * va0[k] for k in WEIGHTS)
    blend_test0 = sum(WEIGHTS[k] * te0[k] for k in WEIGHTS)
    bv0 = evaluate(users["valid"], y["valid"], blend_valid0)["primary"]
    bt0 = evaluate(users["test"], y["test"], blend_test0)["primary"]

    lgb0_valid = evaluate(users["valid"], y["valid"], lgb_valid_seed[0])["primary"]
    xgb0_valid = evaluate(users["valid"], y["valid"], xgb_valid_seed[0])["primary"]

    fidelity = {
        "lgb_valid": {"ours": lgb0_valid, "ref": REFERENCE["lgb_valid"], "delta": lgb0_valid - REFERENCE["lgb_valid"]},
        "xgb_valid": {"ours": xgb0_valid, "ref": REFERENCE["xgb_valid"], "delta": xgb0_valid - REFERENCE["xgb_valid"]},
        "fm_valid": {"ours": fm_valid_metric["primary"], "ref": REFERENCE["fm_valid"], "delta": fm_valid_metric["primary"] - REFERENCE["fm_valid"]},
        "blend_valid": {"ours": bv0, "ref": REFERENCE["blend_valid"], "delta": bv0 - REFERENCE["blend_valid"]},
        "blend_test": {"ours": bt0, "ref": REFERENCE["blend_test"], "delta": bt0 - REFERENCE["blend_test"]},
        "tolerance": TOLERANCE,
        "all_pass": bool(
            abs(lgb0_valid - REFERENCE["lgb_valid"]) <= TOLERANCE
            and abs(xgb0_valid - REFERENCE["xgb_valid"]) <= TOLERANCE
            and abs(fm_valid_metric["primary"] - REFERENCE["fm_valid"]) <= TOLERANCE
            and abs(bv0 - REFERENCE["blend_valid"]) <= TOLERANCE
            and abs(bt0 - REFERENCE["blend_test"]) <= TOLERANCE
        ),
    }
    save_stage("harness_fidelity", fidelity)
    print(json.dumps(jsonable(fidelity), indent=2), flush=True)
    if not fidelity["all_pass"]:
        print("!!! HARNESS FIDELITY FAILED -- stopping before trusting anything new !!!", flush=True)
        return

    # ---- Stage B: per-model seed-ensemble comparison ----
    print("=== [6b/6] per-model seed-ensemble comparison (plain mean vs sigmoid mean) ===", flush=True)

    def ensemble_variants(valid_seed, test_seed, uid_valid, uid_test, y_valid, y_test):
        raw_valid = np.stack([valid_seed[s] for s in SEEDS])
        raw_test = np.stack([test_seed[s] for s in SEEDS])
        plain_valid = np.mean(raw_valid, axis=0)
        plain_test = np.mean(raw_test, axis=0)
        sig_valid = np.mean(sigmoid(raw_valid), axis=0)
        sig_test = np.mean(sigmoid(raw_test), axis=0)
        out = {
            "per_seed_valid": {int(s): evaluate(uid_valid, y_valid, valid_seed[s])["primary"] for s in SEEDS},
            "plain_mean": {
                "valid": evaluate(uid_valid, y_valid, plain_valid)["primary"],
                "test": evaluate(uid_test, y_test, plain_test)["primary"],
            },
            "sigmoid_mean": {
                "valid": evaluate(uid_valid, y_valid, sig_valid)["primary"],
                "test": evaluate(uid_test, y_test, sig_test)["primary"],
            },
        }
        return out, plain_valid, plain_test, sig_valid, sig_test

    lgb_cmp, lgb_plain_valid, lgb_plain_test, lgb_sig_valid, lgb_sig_test = ensemble_variants(
        lgb_valid_seed, lgb_test_seed, users["valid"], users["test"], y["valid"], y["test"]
    )
    xgb_cmp, xgb_plain_valid, xgb_plain_test, xgb_sig_valid, xgb_sig_test = ensemble_variants(
        xgb_valid_seed, xgb_test_seed, users["valid"], users["test"], y["valid"], y["test"]
    )
    seed_ensemble_cmp = {"lgb": lgb_cmp, "xgb": xgb_cmp, "fm_convention": "sigmoid_mean (unchanged production)"}
    save_stage("seed_ensemble_comparison", seed_ensemble_cmp)
    print(json.dumps(jsonable(seed_ensemble_cmp), indent=2), flush=True)

    # pick, per model, whichever convention scores higher on valid standalone
    lgb_use_sigmoid = lgb_cmp["sigmoid_mean"]["valid"] > lgb_cmp["plain_mean"]["valid"]
    xgb_use_sigmoid = xgb_cmp["sigmoid_mean"]["valid"] > xgb_cmp["plain_mean"]["valid"]
    lgb_ens_valid = lgb_sig_valid if lgb_use_sigmoid else lgb_plain_valid
    lgb_ens_test = lgb_sig_test if lgb_use_sigmoid else lgb_plain_test
    xgb_ens_valid = xgb_sig_valid if xgb_use_sigmoid else xgb_plain_valid
    xgb_ens_test = xgb_sig_test if xgb_use_sigmoid else xgb_plain_test

    # ---- Stage C: re-blend at production weights with seed-ensembled LGB/XGB ----
    print("=== [6c/6] re-blend at production 10/52/38 weights with seed-ensembled LGB/XGB ===", flush=True)
    va_new = {
        "fm": wup(fm_valid, users["valid"]),
        "lgb": wup(lgb_ens_valid, users["valid"]),
        "xgb": wup(xgb_ens_valid, users["valid"]),
    }
    te_new = {
        "fm": wup(fm_test, users["test"]),
        "lgb": wup(lgb_ens_test, users["test"]),
        "xgb": wup(xgb_ens_test, users["test"]),
    }
    blend_valid_new = sum(WEIGHTS[k] * va_new[k] for k in WEIGHTS)
    blend_test_new = sum(WEIGHTS[k] * te_new[k] for k in WEIGHTS)
    bv_new = evaluate(users["valid"], y["valid"], blend_valid_new)["primary"]
    bt_new = evaluate(users["test"], y["test"], blend_test_new)["primary"]

    production_weights_result = {
        "lgb_convention_used": "sigmoid_mean" if lgb_use_sigmoid else "plain_mean",
        "xgb_convention_used": "sigmoid_mean" if xgb_use_sigmoid else "plain_mean",
        "weights": WEIGHTS,
        "blend_valid": bv_new,
        "blend_test": bt_new,
        "reference_blend_valid": REFERENCE["blend_valid"],
        "reference_blend_test": REFERENCE["blend_test"],
        "delta_valid": bv_new - REFERENCE["blend_valid"],
        "delta_test": bt_new - REFERENCE["blend_test"],
    }
    save_stage("production_weights_seed_ensembled", production_weights_result)
    print(json.dumps(jsonable(production_weights_result), indent=2), flush=True)

    # ---- Stage D: local grid search around 10/52/38 on VALID ONLY ----
    print("=== [6d/6] local grid search around 10/52/38 on VALID ONLY ===", flush=True)
    grid_results = []
    fm_w0, lgb_w0, xgb_w0 = WEIGHTS["fm"], WEIGHTS["lgb"], WEIGHTS["xgb"]
    deltas = [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06]
    seen = set()
    for dfm in deltas:
        for dlgb in deltas:
            fm_w = fm_w0 + dfm
            lgb_w = lgb_w0 + dlgb
            xgb_w = 1.0 - fm_w - lgb_w
            if fm_w < 0 or lgb_w < 0 or xgb_w < 0:
                continue
            key = (round(fm_w, 4), round(lgb_w, 4), round(xgb_w, 4))
            if key in seen:
                continue
            seen.add(key)
            bv = fm_w * va_new["fm"] + lgb_w * va_new["lgb"] + xgb_w * va_new["xgb"]
            metric = evaluate(users["valid"], y["valid"], bv)["primary"]
            grid_results.append({"fm": key[0], "lgb": key[1], "xgb": key[2], "valid_primary": metric})
    grid_results.sort(key=lambda r: -r["valid_primary"])
    best = grid_results[0]
    best_bt = None
    if best["fm"] == fm_w0 and best["lgb"] == lgb_w0 and best["xgb"] == xgb_w0:
        best_bt = bt_new
    else:
        bt_best = best["fm"] * te_new["fm"] + best["lgb"] * te_new["lgb"] + best["xgb"] * te_new["xgb"]
        best_bt = evaluate(users["test"], y["test"], bt_best)["primary"]

    grid_search = {
        "n_combos": len(grid_results),
        "top10": grid_results[:10],
        "best": best,
        "best_test_primary": best_bt,
        "production_weights_valid": bv_new,
        "delta_best_vs_production_weights": best["valid_primary"] - bv_new,
    }
    save_stage("grid_search", grid_search)
    print(json.dumps(jsonable(grid_search), indent=2), flush=True)

    # ---- Final verdict ----
    final_valid = max(bv_new, best["valid_primary"])
    final_delta = final_valid - REFERENCE["blend_valid"]
    if final_delta > PROMOTION_DELTA:
        verdict = "PROMOTE"
    elif final_delta > PRELIMINARY_DELTA:
        verdict = "PRELIMINARY"
    else:
        verdict = "REJECT"

    summary = {
        "reference_blend_valid": REFERENCE["blend_valid"],
        "reference_blend_test": REFERENCE["blend_test"],
        "seed_ensembled_production_weights_valid": bv_new,
        "seed_ensembled_production_weights_test": bt_new,
        "best_grid_valid": best["valid_primary"],
        "best_grid_weights": {"fm": best["fm"], "lgb": best["lgb"], "xgb": best["xgb"]},
        "best_grid_test": best_bt,
        "final_delta_valid": final_delta,
        "preliminary_delta_threshold": PRELIMINARY_DELTA,
        "promotion_delta_threshold": PROMOTION_DELTA,
        "verdict": verdict,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
        },
        "total_runtime_sec": time.time() - t0,
    }
    save_stage("summary", summary)
    print("\n=== FINAL SUMMARY ===", flush=True)
    print(json.dumps(jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
