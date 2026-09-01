"""iterMERGE8: test whether the 3 decay-rate axes (author-popularity,
duration-bucket, hour-of-day -- iter71/72/73's Laplace-smoothed half-life-3
construction, already REJECTed by iterMERGE2 against yixi's rich 30+-column
LightGBM/XGBoost reference sets) show any signal against iter63's own much
SMALLER `rate_only` feature set (only 6 columns: duration_ms,
decay_rate_2.5, decay_act_2.5, lastk_rate, gap, decay_tab_rate_3 -- plus
categoricals user_id/video_id/author_id/tab/last1). Hypothesis: a
`num_leaves=2` linear tree choosing among only 6 competing numeric features
faces far less competition for splits than one choosing among 30+, so a
weak feature that never wins a split among many strong candidates might
still win among few.

Reuses, as imported modules, not reimplemented:
  - experiments/iterMERGE2_decay_axis_transfer/common.py: `attach_axis_rate`
    (Laplace-smoothed rate construction + row/user/label alignment checks),
    `AXIS_DATA_EXT_PATHS`, `ALPHA`, `HALFLIFE`.
  - experiments/iter63_decay_tab_rate/train.py: `prepare()`, `run()`,
    `_sort_by_user()`, `_de` (its own data_ext.py).
  - experiments/iterMERGE5_four_model_blend/run.py: `fit_lgb`, `fit_xgb`,
    `xgb_config`, `stable_user_order`, `within_user_percentile`, `sigmoid`,
    `blend4`, `LGB_CANDIDATE_COLUMNS`, `XGB_COLUMNS`, `LGB_CONFIG`,
    `PRODUCTION_WEIGHTS`.
  - make_submission.py: `train_one_fm`, `load_ext`, `encode_ext`.
  - experiments/iterYIXI10_video_metadata/features.py: `load_frames`.

Pipeline:
  [1] Harness-fidelity: reproduce iter63 rate_only standalone seed 0
      (0.6716787219047546), the 5-seed sigmoid-mean ensemble
      (0.67141676 valid / 0.65336251 test), the 3-model reference
      (0.69943440 valid / 0.68432260 test), and the raw 4-model blend at
      iterMERGE5's best weights (fm=0.072/lgb=0.5184/xgb=0.3296/i63=0.08
      -> valid=0.69981837/test=0.68367088).
  [2] For each of the 3 axes (author, durbucket, hour): attach the axis's
      Laplace-smoothed decay-rate column onto a fresh copy of iter63's own
      `rate_only` dfs (verified row/user/label-aligned via
      `common.attach_axis_rate`), retrain at seed 0 using iter63's own
      `run()` with `_cache` override, compare valid delta and LightGBM
      `feature_importance()` split usage for the new column.
  [3] If any axis clears standalone signal (nonzero split usage or valid
      delta beyond noise), retrain that variant at 5 seeds and re-run the
      4-model blend weight search (reusing iterMERGE5/run.py's fit_lgb/
      fit_xgb/within_user_percentile/blend4 machinery) with the enhanced
      i63 replacing the original i63 as the 4th component.
  [4] If all 3 axes show zero split usage / exact-zero delta, stop --
      clean REJECT, no blend-level testing (matches iterMERGE2's own
      scoping and this round's explicit instruction to save time).
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
ITER63_DIR = os.path.join(REPO_ROOT, "experiments", "iter63_decay_tab_rate")
MERGE2_DIR = os.path.join(REPO_ROOT, "experiments", "iterMERGE2_decay_axis_transfer")
MERGE5_DIR = os.path.join(REPO_ROOT, "experiments", "iterMERGE5_four_model_blend")
MAKE_SUBMISSION_PATH = os.path.join(REPO_ROOT, "make_submission.py")
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
RESULTS_PATH = os.path.join(THIS_DIR, "results.json")

FM_SEEDS = (0, 1, 2, 3, 4)
ITER63_SEEDS = (0, 1, 2, 3, 4)
TRAIN_SEED = 0

# --- Reference constants, all code-verified in iterMERGE5 (see its RESULT.md) ---
CLAIMED_3MODEL_VALID = 0.69943440
CLAIMED_3MODEL_TEST = 0.68432260
ITER63_RATE_ONLY_SEED0_REFERENCE = 0.6716787219047546
ITER63_5SEED_ENSEMBLE_VALID_REF = 0.6714167594909668
ITER63_5SEED_ENSEMBLE_TEST_REF = 0.653362512588501
BEST_4MODEL_WEIGHTS = {"fm": 0.072, "lgb": 0.5184, "xgb": 0.3296, "i63": 0.08}
CLAIMED_4MODEL_VALID = 0.6998183727264404
CLAIMED_4MODEL_TEST = 0.6836708784103394

PRELIMINARY_DELTA = 0.0003
PROMOTION_DELTA = 0.001
HARNESS_TOLERANCE = 1e-6
NONDETERMINISM_TOLERANCE = 2e-4  # observed cross-run n_jobs=-1 float noise in iterMERGE5


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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


AXES = ("author", "durbucket", "hour")


def main():
    t0 = time.time()
    results = {"experiment": "iterMERGE8_decay_axis_on_i63"}

    print("=== loading reused modules ===", flush=True)
    merge2_common = load_module(os.path.join(MERGE2_DIR, "common.py"), "merge8_merge2_common")
    merge5_run = load_module(os.path.join(MERGE5_DIR, "run.py"), "merge8_merge5_run")
    iter63_train = load_module(os.path.join(ITER63_DIR, "train.py"), "merge8_iter63_train")
    features_mod = load_module(os.path.join(YIXI10_DIR, "features.py"), "merge8_yixi10_features")
    submission = load_module(MAKE_SUBMISSION_PATH, "merge8_submission")

    # =================================================================
    print("\n=== [1a/5] harness-fidelity: iter63 rate_only standalone seed 0 ===", flush=True)
    i63_model0, i63_va0, i63_te0, i63_cache = iter63_train.run(
        DATA_DIR, "rate_only", seed=TRAIN_SEED, verbose=True
    )
    delta_seed0 = i63_va0["primary"] - ITER63_RATE_ONLY_SEED0_REFERENCE
    hf_seed0 = {
        "reference": ITER63_RATE_ONLY_SEED0_REFERENCE,
        "reproduced_valid": i63_va0["primary"],
        "reproduced_test": i63_te0["primary"],
        "delta": delta_seed0,
        "pass": bool(abs(delta_seed0) <= HARNESS_TOLERANCE),
    }
    results["harness_iter63_seed0"] = hf_seed0
    print(f"  seed0 valid={i63_va0['primary']:.10f} ref={ITER63_RATE_ONLY_SEED0_REFERENCE:.10f} "
          f"delta={delta_seed0:.2e} pass={hf_seed0['pass']}", flush=True)
    save_results(results)
    if not hf_seed0["pass"]:
        print("!!! HARNESS FIDELITY (iter63 seed0) FAILED -- stopping !!!", flush=True)
        return

    print("\n=== [1b/5] harness-fidelity: iter63 rate_only 5-seed ensemble ===", flush=True)
    i63_dfs, i63_y, i63_u = i63_cache
    i63_valid_seed_scores = [sigmoid(i63_model0.predict(i63_dfs["valid"]))]
    i63_test_seed_scores = [sigmoid(i63_model0.predict(i63_dfs["test"]))]
    i63_per_seed = {0: {"valid_primary": i63_va0["primary"], "test_primary": i63_te0["primary"]}}
    i63_models = {0: i63_model0}
    for seed in ITER63_SEEDS[1:]:
        m, va, te, _ = iter63_train.run(DATA_DIR, "rate_only", seed=seed, _cache=i63_cache)
        i63_valid_seed_scores.append(sigmoid(m.predict(i63_dfs["valid"])))
        i63_test_seed_scores.append(sigmoid(m.predict(i63_dfs["test"])))
        i63_per_seed[seed] = {"valid_primary": va["primary"], "test_primary": te["primary"]}
        i63_models[seed] = m
        print(f"    seed={seed} valid={va['primary']:.6f} test={te['primary']:.6f}", flush=True)

    i63_valid_ens = np.mean(np.stack(i63_valid_seed_scores), axis=0)
    i63_test_ens = np.mean(np.stack(i63_test_seed_scores), axis=0)
    i63_valid_ens_metrics = evaluate(i63_u["valid"], i63_y["valid"], i63_valid_ens)
    i63_test_ens_metrics = evaluate(i63_u["test"], i63_y["test"], i63_test_ens)
    delta_ens_valid = i63_valid_ens_metrics["primary"] - ITER63_5SEED_ENSEMBLE_VALID_REF
    delta_ens_test = i63_test_ens_metrics["primary"] - ITER63_5SEED_ENSEMBLE_TEST_REF
    hf_ens = {
        "per_seed": i63_per_seed,
        "ensemble_valid_primary": i63_valid_ens_metrics["primary"],
        "ensemble_test_primary": i63_test_ens_metrics["primary"],
        "reference_valid": ITER63_5SEED_ENSEMBLE_VALID_REF,
        "reference_test": ITER63_5SEED_ENSEMBLE_TEST_REF,
        "delta_valid": delta_ens_valid,
        "delta_test": delta_ens_test,
        "pass": bool(abs(delta_ens_valid) <= HARNESS_TOLERANCE and abs(delta_ens_test) <= HARNESS_TOLERANCE),
    }
    results["harness_iter63_5seed_ensemble"] = hf_ens
    print(f"  5-seed ensemble valid={i63_valid_ens_metrics['primary']:.8f} "
          f"(ref {ITER63_5SEED_ENSEMBLE_VALID_REF:.8f}) test={i63_test_ens_metrics['primary']:.8f} "
          f"(ref {ITER63_5SEED_ENSEMBLE_TEST_REF:.8f}) pass={hf_ens['pass']}", flush=True)
    save_results(results)
    if not hf_ens["pass"]:
        print("!!! HARNESS FIDELITY (iter63 5-seed ensemble) FAILED -- stopping !!!", flush=True)
        return

    # =================================================================
    print("\n=== [2/5] harness-fidelity: 3-model reference (FM/LGB/XGB) ===", flush=True)
    frames, y, users, _meta = features_mod.load_frames()
    lgb_cand_model = merge5_run.fit_lgb(frames, y, users, merge5_run.LGB_CANDIDATE_COLUMNS, TRAIN_SEED)
    xgb_model = merge5_run.fit_xgb(frames, y, users, merge5_run.XGB_COLUMNS, TRAIN_SEED)
    lgb_cand_valid = lgb_cand_model.predict(frames["valid"][merge5_run.LGB_CANDIDATE_COLUMNS])
    lgb_cand_test = lgb_cand_model.predict(frames["test"][merge5_run.LGB_CANDIDATE_COLUMNS])
    xgb_valid = xgb_model.predict(frames["valid"][merge5_run.XGB_COLUMNS])
    xgb_test = xgb_model.predict(frames["test"][merge5_run.XGB_COLUMNS])
    lgb_valid_metrics = evaluate(users["valid"], y["valid"], lgb_cand_valid)
    xgb_valid_metrics = evaluate(users["valid"], y["valid"], xgb_valid)

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

    fm_valid_scores, fm_test_scores = [], []
    for seed in FM_SEEDS:
        print(f"  fitting FM seed={seed}", flush=True)
        model = submission.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits["train"], dim, seed)
        fm_valid_scores.append(sigmoid(model.predict(Xva)))
        fm_test_scores.append(sigmoid(model.predict(Xte)))
    fm_valid = np.mean(np.stack(fm_valid_scores), axis=0)
    fm_test = np.mean(np.stack(fm_test_scores), axis=0)
    fm_valid_metrics = evaluate(users["valid"], y["valid"], fm_valid)

    valid_components_3 = {
        "fm": merge5_run.within_user_percentile(fm_valid, users["valid"]),
        "lgb": merge5_run.within_user_percentile(lgb_cand_valid, users["valid"]),
        "xgb": merge5_run.within_user_percentile(xgb_valid, users["valid"]),
    }
    test_components_3 = {
        "fm": merge5_run.within_user_percentile(fm_test, users["test"]),
        "lgb": merge5_run.within_user_percentile(lgb_cand_test, users["test"]),
        "xgb": merge5_run.within_user_percentile(xgb_test, users["test"]),
    }
    blend3_valid = merge5_run.blend4(valid_components_3, merge5_run.PRODUCTION_WEIGHTS, users["valid"], y["valid"])
    blend3_test = merge5_run.blend4(test_components_3, merge5_run.PRODUCTION_WEIGHTS, users["test"], y["test"])

    hf3 = {
        "lgb_candidate_valid": lgb_valid_metrics["primary"],
        "xgb_valid": xgb_valid_metrics["primary"],
        "fm_valid": fm_valid_metrics["primary"],
        "blend_valid": blend3_valid["primary"],
        "blend_test": blend3_test["primary"],
        "claimed_blend_valid": CLAIMED_3MODEL_VALID,
        "claimed_blend_test": CLAIMED_3MODEL_TEST,
        "delta_blend_valid": blend3_valid["primary"] - CLAIMED_3MODEL_VALID,
        "delta_blend_test": blend3_test["primary"] - CLAIMED_3MODEL_TEST,
    }
    hf3["pass"] = bool(
        abs(hf3["delta_blend_valid"]) <= HARNESS_TOLERANCE
        and abs(hf3["delta_blend_test"]) <= HARNESS_TOLERANCE
    )
    results["harness_3model"] = hf3
    print(f"  3-model blend valid={blend3_valid['primary']:.8f} (ref {CLAIMED_3MODEL_VALID}) "
          f"test={blend3_test['primary']:.8f} (ref {CLAIMED_3MODEL_TEST}) pass={hf3['pass']}", flush=True)
    save_results(results)
    if not hf3["pass"]:
        print("!!! HARNESS FIDELITY (3-model) FAILED -- stopping !!!", flush=True)
        return

    # =================================================================
    print("\n=== [3/5] harness-fidelity: raw 4-model blend at iterMERGE5 best weights ===", flush=True)
    # Row-alignment: already exhaustively verified in iterMERGE5/run.py
    # (0 mismatches, both splits) using the identical code path (yixi
    # features.py frames vs iter63's own data_ext splits). Reproduce the
    # same check here rather than assuming it still holds.
    i63_splits_raw = iter63_train._de.load_ext(DATA_DIR, use_cache=True)
    i63_video_raw = {
        name: np.array([x[iter63_train._de.IDX["video_id"]] for x in i63_splits_raw[name]], dtype=str)
        for name in ("valid", "test")
    }
    for name in ("valid", "test"):
        yixi_uid = np.asarray(users[name]).astype(str)
        yixi_vid = frames[name]["video_id"].astype(str).to_numpy()
        i63_uid = np.asarray(i63_u[name]).astype(str)
        i63_vid = i63_video_raw[name]
        uid_mismatch = int((yixi_uid != i63_uid).sum())
        nan_video_yixi = int(frames[name]["video_id"].isna().sum())
        vid_mismatch_raw = int((yixi_vid != i63_vid).sum())
        label_mismatch = int((np.asarray(y[name]) != np.asarray(i63_y[name])).sum())
        ok = uid_mismatch == 0 and (vid_mismatch_raw - nan_video_yixi) == 0 and label_mismatch == 0
        print(f"  [{name}] uid_mismatch={uid_mismatch} vid_mismatch(excl known nan)="
              f"{vid_mismatch_raw - nan_video_yixi} label_mismatch={label_mismatch} ok={ok}", flush=True)
        if not ok:
            raise AssertionError(f"row alignment check failed for split={name}")

    valid_components_4 = dict(valid_components_3)
    valid_components_4["i63"] = merge5_run.within_user_percentile(i63_valid_ens, i63_u["valid"])
    test_components_4 = dict(test_components_3)
    test_components_4["i63"] = merge5_run.within_user_percentile(i63_test_ens, i63_u["test"])

    blend4_valid = merge5_run.blend4(valid_components_4, BEST_4MODEL_WEIGHTS, users["valid"], y["valid"])
    blend4_test = merge5_run.blend4(test_components_4, BEST_4MODEL_WEIGHTS, users["test"], y["test"])
    delta4_valid = blend4_valid["primary"] - CLAIMED_4MODEL_VALID
    delta4_test = blend4_test["primary"] - CLAIMED_4MODEL_TEST
    hf4 = {
        "weights": BEST_4MODEL_WEIGHTS,
        "reproduced_valid": blend4_valid["primary"],
        "reproduced_test": blend4_test["primary"],
        "claimed_valid": CLAIMED_4MODEL_VALID,
        "claimed_test": CLAIMED_4MODEL_TEST,
        "delta_valid": delta4_valid,
        "delta_test": delta4_test,
        # Wider tolerance: iterMERGE5 documented ~1e-4-scale cross-run float
        # noise between run.py's original grid search and confirm.py's
        # independent refit (both used n_jobs=-1 LightGBM/XGBoost), even
        # with fixed random_state.
        "pass": bool(abs(delta4_valid) <= NONDETERMINISM_TOLERANCE and abs(delta4_test) <= NONDETERMINISM_TOLERANCE),
    }
    results["harness_4model_raw_blend"] = hf4
    print(f"  4-model blend valid={blend4_valid['primary']:.8f} (ref {CLAIMED_4MODEL_VALID:.8f}) "
          f"test={blend4_test['primary']:.8f} (ref {CLAIMED_4MODEL_TEST:.8f}) pass={hf4['pass']}", flush=True)
    save_results(results)
    if not hf4["pass"]:
        print("!!! HARNESS FIDELITY (4-model raw blend) FAILED -- stopping !!!", flush=True)
        return

    print("\n*** ALL HARNESS-FIDELITY CHECKS PASSED ***", flush=True)
    results["harness_fidelity_all_pass"] = True
    save_results(results)

    # =================================================================
    print("\n=== [4/5] standalone ablations: attach each axis to i63's rate_only set (seed 0) ===", flush=True)
    ablations = {}
    any_signal = False
    for axis in AXES:
        print(f"\n  -- axis={axis} --", flush=True)
        axis_dfs = dict(i63_dfs)  # shallow copy: attach_axis_rate reassigns
                                   # per-split keys to fresh DataFrame copies,
                                   # so this does not mutate i63_dfs itself.
        col_name = merge2_common.attach_axis_rate(axis_dfs, i63_y, i63_u, axis)
        model, va, te, _ = iter63_train.run(
            DATA_DIR, "rate_only", seed=TRAIN_SEED, verbose=False,
            _cache=(axis_dfs, i63_y, i63_u),
        )
        gain = model.booster_.feature_importance(importance_type="gain")
        names = model.booster_.feature_name()
        split_count = model.booster_.feature_importance(importance_type="split")
        gain_by_name = dict(zip(names, gain))
        split_by_name = dict(zip(names, split_count))
        new_col_gain = float(gain_by_name.get(col_name, 0.0))
        new_col_splits = int(split_by_name.get(col_name, 0))
        delta_valid = va["primary"] - i63_va0["primary"]
        clears = bool(new_col_splits > 0 or abs(delta_valid) > 1e-9)
        any_signal = any_signal or clears
        row = {
            "axis": axis,
            "column": col_name,
            "valid_primary": va["primary"],
            "test_primary": te["primary"],
            "delta_valid_vs_i63_seed0": delta_valid,
            "new_column_gain": new_col_gain,
            "new_column_split_count": new_col_splits,
            "total_gain": float(np.sum(gain)),
            "new_column_gain_fraction": (new_col_gain / float(np.sum(gain))) if np.sum(gain) else 0.0,
            "clears_standalone_signal_gate": clears,
        }
        ablations[axis] = row
        print(f"    valid={va['primary']:.10f} delta={delta_valid:+.10f} "
              f"new_col_splits={new_col_splits} new_col_gain_frac={row['new_column_gain_fraction']:.6f} "
              f"clears_gate={clears}", flush=True)
    results["standalone_ablations"] = ablations
    results["any_axis_shows_standalone_signal"] = any_signal
    save_results(results)

    # =================================================================
    if not any_signal:
        print("\n=== [5/5] verdict: all 3 axes show zero split usage / exact-zero delta ===", flush=True)
        print("  CLEAN REJECT -- skipping 5-seed retrain and blend-level testing per protocol", flush=True)
        results["verdict"] = "REJECT"
        results["verdict_reason"] = (
            "All 3 decay-rate axes (author, durbucket, hour) show zero "
            "LightGBM split usage and exact-zero (<1e-9) valid delta when "
            "added to iter63's own minimal 6-column rate_only feature set, "
            "extending iterMERGE2's finding (same 4/4-null signature) to a "
            "3rd, much smaller-capacity harness. No blend-level retest "
            "performed, per protocol (gate requires standalone signal)."
        )
        results["elapsed_seconds"] = time.time() - t0
        results["environment"] = {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "lightgbm": lgb.__version__, "xgboost": xgb.__version__,
        }
        save_results(results)
        print(f"\n=== done in {results['elapsed_seconds']:.1f}s -- VERDICT: REJECT ===", flush=True)
        return

    # =================================================================
    print("\n=== [5/5] signal detected -- 5-seed retrain + 4-model blend re-search ===", flush=True)
    signal_axes = [a for a, r in ablations.items() if r["clears_standalone_signal_gate"]]
    print(f"  axes clearing standalone gate: {signal_axes}", flush=True)
    blend_results = {}
    for axis in signal_axes:
        print(f"\n  -- 5-seed retrain: axis={axis} --", flush=True)
        axis_dfs = dict(i63_dfs)
        col_name = merge2_common.attach_axis_rate(axis_dfs, i63_y, i63_u, axis)
        axis_cache = (axis_dfs, i63_y, i63_u)
        valid_seed_scores, test_seed_scores, per_seed = [], [], {}
        for seed in ITER63_SEEDS:
            m, va, te, _ = iter63_train.run(DATA_DIR, "rate_only", seed=seed, _cache=axis_cache)
            valid_seed_scores.append(sigmoid(m.predict(axis_dfs["valid"])))
            test_seed_scores.append(sigmoid(m.predict(axis_dfs["test"])))
            per_seed[seed] = {"valid_primary": va["primary"], "test_primary": te["primary"]}
            print(f"    seed={seed} valid={va['primary']:.6f} test={te['primary']:.6f}", flush=True)
        v_ens = np.mean(np.stack(valid_seed_scores), axis=0)
        t_ens = np.mean(np.stack(test_seed_scores), axis=0)
        v_ens_metrics = evaluate(i63_u["valid"], i63_y["valid"], v_ens)
        t_ens_metrics = evaluate(i63_u["test"], i63_y["test"], t_ens)
        print(f"  {axis} 5-seed ensemble: valid={v_ens_metrics['primary']:.8f} test={t_ens_metrics['primary']:.8f}", flush=True)

        vc4 = dict(valid_components_3)
        vc4["i63"] = merge5_run.within_user_percentile(v_ens, i63_u["valid"])
        tc4 = dict(test_components_3)
        tc4["i63"] = merge5_run.within_user_percentile(t_ens, i63_u["test"])

        coarse_grid = []
        base_fm, base_lgb, base_xgb = merge5_run.PRODUCTION_WEIGHTS["fm"], merge5_run.PRODUCTION_WEIGHTS["lgb"], merge5_run.PRODUCTION_WEIGHTS["xgb"]
        for w_i63 in (0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30):
            scale = 1.0 - w_i63
            w = {"fm": base_fm * scale, "lgb": base_lgb * scale, "xgb": base_xgb * scale, "i63": w_i63}
            vm = merge5_run.blend4(vc4, w, users["valid"], y["valid"])
            tm = merge5_run.blend4(tc4, w, users["test"], y["test"])
            coarse_grid.append({"weights": dict(w), "valid_primary": vm["primary"], "test_primary": tm["primary"]})
        best_coarse = max(coarse_grid, key=lambda r: r["valid_primary"])

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
                    vm = merge5_run.blend4(vc4, w, users["valid"], y["valid"])
                    tm = merge5_run.blend4(tc4, w, users["test"], y["test"])
                    fine_grid.append({"weights": dict(w), "valid_primary": vm["primary"], "test_primary": tm["primary"]})
        fine_grid.sort(key=lambda r: -r["valid_primary"])
        best = fine_grid[0]
        best_delta_vs_3model = best["valid_primary"] - blend3_valid["primary"]
        best_delta_vs_merge5_4model = best["valid_primary"] - CLAIMED_4MODEL_VALID
        print(f"  {axis} best 4-model weights={best['weights']} valid={best['valid_primary']:.8f} "
              f"(delta vs 3-model={best_delta_vs_3model:+.8f}, vs MERGE5 4-model={best_delta_vs_merge5_4model:+.8f})", flush=True)

        blend_results[axis] = {
            "per_seed": per_seed,
            "ensemble_valid_primary": v_ens_metrics["primary"],
            "ensemble_test_primary": t_ens_metrics["primary"],
            "coarse_grid": coarse_grid,
            "fine_grid_top10": fine_grid[:10],
            "best_weights": best["weights"],
            "best_valid": best["valid_primary"],
            "best_test": best["test_primary"],
            "delta_vs_3model_reference": best_delta_vs_3model,
            "delta_vs_merge5_4model_reference": best_delta_vs_merge5_4model,
        }
    results["blend_retest"] = blend_results
    save_results(results)

    best_overall_delta = max(r["delta_vs_3model_reference"] for r in blend_results.values())
    if best_overall_delta >= PROMOTION_DELTA:
        verdict = "PROMOTE-candidate"
    elif best_overall_delta >= PRELIMINARY_DELTA:
        verdict = "PRELIMINARY"
    else:
        verdict = "REJECT"
    results["verdict"] = verdict
    results["elapsed_seconds"] = time.time() - t0
    results["environment"] = {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "lightgbm": lgb.__version__, "xgboost": xgb.__version__,
    }
    save_results(results)
    print(f"\n=== done in {results['elapsed_seconds']:.1f}s -- VERDICT: {verdict} ===", flush=True)


if __name__ == "__main__":
    main()
