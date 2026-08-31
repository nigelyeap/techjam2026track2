"""Build and audit the intrinsic metadata frame without fitting a model."""

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
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    frames, _, _, metadata = features.load_frames()
    expected = set(features.CATEGORICAL_COLUMNS + features.CONTINUOUS_COLUMNS)
    for name, frame in frames.items():
        if not expected.issubset(frame.columns):
            raise AssertionError(f"missing metadata columns in {name}")
    results = {
        "experiment": "iterYIXI10_intrinsic_metadata_audit",
        "selection_role": "none; policy and feature audit before model scores",
        "metadata": metadata,
        "test_metrics_computed": False,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(jsonable(results), f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        f"join={metadata['source']['join_unmatched_rows']} unmatched "
        f"age={metadata['summaries']['video_age_days']['train']['min']:.0f}.."
        f"{metadata['summaries']['video_age_days']['test']['max']:.0f} "
        f"statistic_file_read={metadata['source']['statistic_file_read']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
