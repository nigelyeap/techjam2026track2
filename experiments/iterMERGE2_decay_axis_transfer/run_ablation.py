"""Six single-seed standalone ablations: each of the three causal decay-rate
axes (author, durbucket, hour) added to yixi's LGB reference columns and,
separately, to her XGB reference columns. Must be run after run_harness.py
has passed (harness_results.json with all_ok=True).

Writes ablation_results.json.
"""

from __future__ import annotations

import os
import time

import common as c

AXES = ("author", "durbucket", "hour")


def main():
    harness_path = os.path.join(c.THIS_DIR, "harness_results.json")
    if not os.path.exists(harness_path):
        raise SystemExit("run_harness.py must complete successfully first")
    harness = c.read_json(harness_path)
    if not harness.get("all_ok"):
        raise SystemExit("harness fidelity check did not pass; aborting ablation")

    t0 = time.time()
    results = {"environment": c.environment(), "axes": {}}

    print("loading iterYIXI9 reference frames (shared base for all 6 ablations)", flush=True)
    base_frames, y, users, meta = c.load_reference_frames("yixi9")

    for axis in AXES:
        print(f"=== axis={axis} ===", flush=True)
        frames = {split: base_frames[split].copy() for split in ("train", "valid", "test")}
        col_name = c.attach_axis_rate(frames, y, users, axis)
        print(f"  attached column {col_name}", flush=True)

        lgb_cols = c.LGB_REFERENCE_COLUMNS + [col_name]
        xgb_cols = c.XGB_REFERENCE_COLUMNS + [col_name]

        print(f"  fitting LGB reference+{col_name} (seed=0)...", flush=True)
        _, lgb_scores, lgb_metrics, lgb_gain = c.fit_lgb(frames, y, users, lgb_cols, seed=0)
        lgb_delta = lgb_metrics["primary"] - c.LGB_REFERENCE_VALID
        print(f"    LGB primary={lgb_metrics['primary']:.8f} delta={lgb_delta:+.6f}", flush=True)

        print(f"  fitting XGB reference+{col_name} (seed=0)...", flush=True)
        _, xgb_scores, xgb_metrics, xgb_gain = c.fit_xgb(frames, y, users, xgb_cols, seed=0)
        xgb_delta = xgb_metrics["primary"] - c.XGB_REFERENCE_VALID
        print(f"    XGB primary={xgb_metrics['primary']:.8f} delta={xgb_delta:+.6f}", flush=True)

        results["axes"][axis] = {
            "column": col_name,
            "lgb": {
                "valid": c.metric_dict(lgb_metrics),
                "delta_vs_reference": float(lgb_delta),
                "clears_preliminary": bool(lgb_delta >= c.PRELIMINARY_DELTA),
                "feature_gain_fraction": lgb_gain.get(col_name, 0.0),
            },
            "xgb": {
                "valid": c.metric_dict(xgb_metrics),
                "delta_vs_reference": float(xgb_delta),
                "clears_preliminary": bool(xgb_delta >= c.PRELIMINARY_DELTA),
                "feature_gain_fraction": xgb_gain.get(col_name, 0.0),
            },
        }
        del frames
        import gc

        gc.collect()

    results["elapsed_seconds"] = time.time() - t0
    any_promising = any(
        results["axes"][axis]["lgb"]["clears_preliminary"]
        or results["axes"][axis]["xgb"]["clears_preliminary"]
        for axis in AXES
    )
    results["any_promising"] = any_promising
    out_path = os.path.join(c.THIS_DIR, "ablation_results.json")
    c.write_json(out_path, results)
    print(f"wrote {out_path}; any_promising={any_promising}", flush=True)


if __name__ == "__main__":
    main()
