"""Build/audit the 6i watch-depth feature frame before any model fit."""

from __future__ import annotations

import json
import os

import numpy as np

import features


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "feature_results.json")


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    frames, _, _, metadata = features.load_frames(features.DATA_DIR, use_cache=True)
    summaries = {}
    for feature in features.FEATURES:
        summaries[feature] = {}
        for split in ("train", "valid", "test"):
            values = frames[split][feature].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            summaries[feature][split] = {
                "rows": len(values),
                "finite_rows": len(finite),
                "coverage": float(len(finite) / len(values)),
                "min": float(np.min(finite)),
                "q50": float(np.quantile(finite, 0.5)),
                "q99": float(np.quantile(finite, 0.99)),
                "max": float(np.max(finite)),
            }
    results = {
        "experiment": "iterYIXI9_watch_depth_feature_audit",
        "selection_role": "none; run before model validation",
        "metadata": metadata,
        "feature_summaries": summaries,
        "test_metrics_computed": False,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(jsonable(results), f, indent=2, sort_keys=True)
        f.write("\n")
    distribution = metadata["watch_fraction_distribution"]
    causality = metadata["causality"]
    print(
        f"raw_gt_1={distribution['raw_fraction_gt_1_rate']:.6f} "
        f"zero_duration={distribution['zero_duration_rows_excluded_from_history_updates']}",
        flush=True,
    )
    print(
        f"causality rows={causality['rows_checked']} "
        f"max_error={causality['global_max_abs_error']:.3g} "
        f"real_tie_groups={causality['real_exact_timestamp_groups']}",
        flush=True,
    )
    for feature in features.FEATURES:
        print(
            f"{feature}: valid coverage="
            f"{summaries[feature]['valid']['coverage']:.6f}",
            flush=True,
        )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
