"""Harness-fidelity check: reproduce yixi's reference LightGBM/XGBoost valid
numbers exactly, and reproduce her current-best 3-model blend, before
trusting anything new added in this directory.

Must be run before run_ablation.py.  Writes harness_results.json.
"""

from __future__ import annotations

import os
import time

import numpy as np

import common as c


def main():
    t0 = time.time()
    print("loading iterYIXI9 reference frames (source of LGB/XGB reference cols)", flush=True)
    frames, y, users, meta9 = c.load_reference_frames("yixi9")
    print(f"  train={len(frames['train'])} valid={len(frames['valid'])} test={len(frames['test'])}", flush=True)

    print("fitting reference LightGBM (seed=0)...", flush=True)
    lgb_model, lgb_scores, lgb_metrics, lgb_gain = c.fit_lgb(
        frames, y, users, c.LGB_REFERENCE_COLUMNS, seed=0
    )
    lgb_delta = lgb_metrics["primary"] - c.LGB_REFERENCE_VALID
    print(f"  LGB primary={lgb_metrics['primary']:.10f} target={c.LGB_REFERENCE_VALID:.10f} delta={lgb_delta:+.2e}", flush=True)

    print("fitting reference XGBoost (rate col, seed=0)...", flush=True)
    xgb_model, xgb_scores, xgb_metrics, xgb_gain = c.fit_xgb(
        frames, y, users, c.XGB_REFERENCE_COLUMNS, seed=0
    )
    xgb_delta = xgb_metrics["primary"] - c.XGB_REFERENCE_VALID
    print(f"  XGB(rate) primary={xgb_metrics['primary']:.10f} target={c.XGB_REFERENCE_VALID:.10f} delta={xgb_delta:+.2e}", flush=True)

    print("fitting current-production XGBoost (raw count col, seed=0)...", flush=True)
    xgb_cur_model, xgb_cur_scores, xgb_cur_metrics, xgb_cur_gain = c.fit_xgb(
        frames, y, users, c.CURRENT_XGB_COLUMNS, seed=0
    )
    xgb_cur_delta = xgb_cur_metrics["primary"] - c.CURRENT_XGB_VALID
    print(f"  XGB(count) primary={xgb_cur_metrics['primary']:.10f} target={c.CURRENT_XGB_VALID:.10f} delta={xgb_cur_delta:+.2e}", flush=True)

    tol = 1e-8
    lgb_ok = abs(lgb_delta) <= tol
    xgb_ok = abs(xgb_delta) <= tol
    xgb_cur_ok = abs(xgb_cur_delta) <= tol
    print(f"fidelity: LGB ok={lgb_ok} XGB(rate) ok={xgb_ok} XGB(count) ok={xgb_cur_ok}", flush=True)

    print("loading FM frozen predictions + iterYIXI10 best columns for blend fidelity", flush=True)
    frames10, y10, users10, meta10 = c.load_reference_frames("yixi10")
    yixi8_common = c.load_module(c.YIXI8_COMMON_PATH, "yixi8_common_for_merge2")
    frozen = yixi8_common.load_frozen()
    fm_scores_valid = frozen["fm"]

    if not np.array_equal(np.asarray(users10["valid"]), np.asarray(users["valid"])):
        raise AssertionError("user_id alignment mismatch between yixi9 and yixi10 valid splits")
    if not np.array_equal(np.asarray(y10["valid"]), np.asarray(y["valid"])):
        raise AssertionError("label alignment mismatch between yixi9 and yixi10 valid splits")
    if not np.array_equal(np.asarray(frozen["users"]), np.asarray(users10["valid"]).astype(str)):
        raise AssertionError("frozen FM user_id alignment mismatch against yixi10 valid split")
    if not np.array_equal(np.asarray(frozen["labels"], dtype=np.float32), np.asarray(y10["valid"], dtype=np.float32)):
        raise AssertionError("frozen FM label alignment mismatch against yixi10 valid split")

    print("fitting best-blend LightGBM (yixi10 columns incl. meta_upload_type, seed=0)...", flush=True)
    lgb_best_model, lgb_best_scores, lgb_best_metrics, lgb_best_gain = c.fit_lgb(
        frames10, y10, users10, c.LGB_BEST_COLUMNS, seed=0
    )
    print(f"  LGB(best) primary={lgb_best_metrics['primary']:.10f}", flush=True)

    print("fitting best-blend XGBoost (current-production cols, seed=0)...", flush=True)
    xgb_best_model, xgb_best_scores, xgb_best_metrics, xgb_best_gain = c.fit_xgb(
        frames10, y10, users10, c.XGB_BEST_COLUMNS, seed=0
    )
    print(f"  XGB(best) primary={xgb_best_metrics['primary']:.10f}", flush=True)

    blend_mod = c.load_module(c.YIXI5_BLEND_PATH, "yixi5_blend_for_merge2")
    components = blend_mod.normalize_components(
        fm_scores_valid, lgb_best_scores, xgb_best_scores, users10["valid"], "within_user_percentile"
    )
    combined = blend_mod.combined_scores(components, c.BEST_WEIGHTS)
    from evaluate import evaluate

    blend_metrics = evaluate(users10["valid"], y10["valid"], combined)
    blend_delta = blend_metrics["primary"] - c.BEST_BLEND_VALID
    print(f"  BLEND primary={blend_metrics['primary']:.10f} target={c.BEST_BLEND_VALID:.10f} delta={blend_delta:+.2e}", flush=True)
    blend_ok = abs(blend_delta) <= 5e-4  # looser: RESULT.md value rounded to 8 dp by hand

    all_ok = bool(lgb_ok and xgb_ok and xgb_cur_ok and blend_ok)
    print(f"ALL FIDELITY CHECKS PASSED = {all_ok}", flush=True)

    payload = {
        "environment": c.environment(),
        "elapsed_seconds": time.time() - t0,
        "lgb_reference": {
            "valid": c.metric_dict(lgb_metrics),
            "target": c.LGB_REFERENCE_VALID,
            "delta": float(lgb_delta),
            "ok": lgb_ok,
        },
        "xgb_reference_rate": {
            "valid": c.metric_dict(xgb_metrics),
            "target": c.XGB_REFERENCE_VALID,
            "delta": float(xgb_delta),
            "ok": xgb_ok,
        },
        "xgb_current_count": {
            "valid": c.metric_dict(xgb_cur_metrics),
            "target": c.CURRENT_XGB_VALID,
            "delta": float(xgb_cur_delta),
            "ok": xgb_cur_ok,
        },
        "best_blend": {
            "valid": c.metric_dict(blend_metrics),
            "target": c.BEST_BLEND_VALID,
            "delta": float(blend_delta),
            "ok": blend_ok,
            "weights": c.BEST_WEIGHTS,
        },
        "all_ok": all_ok,
    }
    out_path = os.path.join(c.THIS_DIR, "harness_results.json")
    c.write_json(out_path, payload)
    print(f"wrote {out_path}", flush=True)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
