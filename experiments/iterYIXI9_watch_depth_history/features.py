"""Causal historical watch-depth features for YIXI Section 6i.

Current-row play time is never exposed.  Each row reads state accumulated
from strictly smaller ``time_ms`` values for the same user; all exact-time
ties are treated as simultaneous and update state only after every tied row
has read it.  Earlier interactions on the same date are intentionally usable.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import importlib.util
import os
import pickle
from typing import Any

import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
BASE_FEATURES_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI6_cross_model_feature_transfer", "features.py"
)
CACHE_PATH = os.path.join(THIS_DIR, ".watch_depth_frames_v1.pkl")
RAW_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
MS_PER_DAY = 86_400_000.0
HALFLIVES = (2.5, 5.0)
TAB_HALFLIFE = 3.0
FEATURES = (
    "hist_watch_decay_mean_2.5",
    "hist_watch_decay_mean_5",
    "hist_watch_last5_mean",
    "hist_watch_tab_decay_mean_3",
)

# Compact raw-row tuple layout.
DATE, USER, TAB, PLAY, DURATION, TIME, ORIG, LABEL = range(8)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_rows(data_dir: str):
    import sys

    sys.path.insert(0, REPO_ROOT)
    from data import SPLITS

    all_rows = []
    distribution_values = []
    zero_duration = 0
    negative_duration = 0
    negative_play = 0
    original_index = 0
    for filename in RAW_FILES:
        with open(os.path.join(data_dir, filename), newline="") as f:
            for raw in csv.DictReader(f):
                play = float(raw["play_time_ms"])
                duration = float(raw["duration_ms"])
                negative_play += int(play < 0)
                negative_duration += int(duration < 0)
                zero_duration += int(duration == 0)
                if duration > 0:
                    distribution_values.append(play / duration)
                all_rows.append(
                    (
                        int(raw["date"]),
                        raw["user_id"],
                        raw["tab"],
                        play,
                        duration,
                        int(raw["time_ms"]),
                        original_index,
                        1 if raw["long_view"] != "0" else 0,
                    )
                )
                original_index += 1

    splits = {}
    for name, (lo, hi) in SPLITS.items():
        splits[name] = [row for row in all_rows if lo <= row[DATE] <= hi]
    rows = splits["train"] + splits["valid"] + splits["test"]
    lengths = {name: len(splits[name]) for name in ("train", "valid", "test")}
    raw_fraction = np.asarray(distribution_values, dtype=np.float64)
    clipped = np.clip(raw_fraction, 0.0, 1.0)
    distribution = {
        "rows": len(all_rows),
        "defined_fraction_rows": int(len(raw_fraction)),
        "zero_duration_rows_excluded_from_history_updates": int(zero_duration),
        "negative_duration_rows": int(negative_duration),
        "negative_play_rows": int(negative_play),
        "raw_quantiles": {
            str(q): float(np.quantile(raw_fraction, q))
            for q in (0.0, 0.01, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0)
        },
        "clipped_quantiles": {
            str(q): float(np.quantile(clipped, q))
            for q in (0.0, 0.01, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        },
        "raw_fraction_gt_1_rows": int(np.sum(raw_fraction > 1.0)),
        "raw_fraction_gt_1_rate": float(np.mean(raw_fraction > 1.0)),
        "raw_fraction_gt_2_rate": float(np.mean(raw_fraction > 2.0)),
        "raw_fraction_gt_5_rate": float(np.mean(raw_fraction > 5.0)),
        "policy": {
            "defined": "play_time_ms / duration_ms when duration_ms > 0",
            "clipping": "clip to [0, 1] because completion saturates at 1 and the raw upper tail reaches 68.31",
            "duration_ms_equal_0": "undefined; row stays in interaction ordering but does not update watch-value numerator/denominator",
            "current_row_play_time": "never read into its own feature; update occurs only after feature snapshot",
        },
    }
    return rows, lengths, distribution


def watch_values(rows):
    values = np.full(len(rows), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        if row[DURATION] > 0:
            values[index] = np.clip(row[PLAY] / row[DURATION], 0.0, 1.0)
    return values


def compute_features(rows, values):
    n = len(rows)
    outputs = {name: np.full(n, np.nan, dtype=np.float64) for name in FEATURES}
    by_user = collections.defaultdict(list)
    for index, row in enumerate(rows):
        by_user[row[USER]].append(index)

    same_timestamp_groups = []
    same_date_earlier_targets = []
    for indices in by_user.values():
        indices.sort(key=lambda index: (rows[index][TIME], rows[index][ORIG]))
        user_num = np.zeros(len(HALFLIVES), dtype=np.float64)
        user_den = np.zeros(len(HALFLIVES), dtype=np.float64)
        user_last_time = None
        tab_state = {}
        last_five = collections.deque(maxlen=5)
        prior_dates = set()

        start = 0
        while start < len(indices):
            stop = start + 1
            timestamp = rows[indices[start]][TIME]
            while stop < len(indices) and rows[indices[stop]][TIME] == timestamp:
                stop += 1
            group = indices[start:stop]
            if len(group) > 1:
                same_timestamp_groups.append(list(group))

            if user_last_time is not None:
                gap_days = (timestamp - user_last_time) / MS_PER_DAY
                for position, halflife in enumerate(HALFLIVES):
                    factor = 0.5 ** (gap_days / halflife)
                    user_num[position] *= factor
                    user_den[position] *= factor
            user_snapshot = np.divide(
                user_num,
                user_den,
                out=np.full_like(user_num, np.nan),
                where=user_den > 0,
            )
            valid_last_five = [value for value in last_five if np.isfinite(value)]
            last_five_snapshot = (
                float(np.mean(valid_last_five)) if valid_last_five else np.nan
            )
            for index in group:
                outputs["hist_watch_decay_mean_2.5"][index] = user_snapshot[0]
                outputs["hist_watch_decay_mean_5"][index] = user_snapshot[1]
                outputs["hist_watch_last5_mean"][index] = last_five_snapshot
                if rows[index][DATE] in prior_dates:
                    same_date_earlier_targets.append(index)

            groups_by_tab = collections.defaultdict(list)
            for index in group:
                groups_by_tab[rows[index][TAB]].append(index)
            for tab, tab_indices in groups_by_tab.items():
                last_time, numerator, denominator = tab_state.get(
                    tab, (None, 0.0, 0.0)
                )
                if last_time is not None:
                    gap_days = (timestamp - last_time) / MS_PER_DAY
                    factor = 0.5 ** (gap_days / TAB_HALFLIFE)
                    numerator *= factor
                    denominator *= factor
                snapshot = numerator / denominator if denominator > 0 else np.nan
                for index in tab_indices:
                    outputs["hist_watch_tab_decay_mean_3"][index] = snapshot
                valid = [values[index] for index in tab_indices if np.isfinite(values[index])]
                if valid:
                    numerator += float(np.sum(valid))
                    denominator += float(len(valid))
                tab_state[tab] = (timestamp, numerator, denominator)

            valid_group = [values[index] for index in group if np.isfinite(values[index])]
            if valid_group:
                user_num += float(np.sum(valid_group))
                user_den += float(len(valid_group))
            for index in group:
                last_five.append(values[index])
                prior_dates.add(rows[index][DATE])
            user_last_time = timestamp
            start = stop

    debug = {
        "by_user": by_user,
        "same_timestamp_groups": same_timestamp_groups,
        "same_date_earlier_targets": same_date_earlier_targets,
    }
    return outputs, debug


def manual_features(target, rows, values, user_indices):
    target_row = rows[target]
    prior = [index for index in user_indices if rows[index][TIME] < target_row[TIME]]
    prior.sort(key=lambda index: (rows[index][TIME], rows[index][ORIG]))

    result = {}
    for halflife, name in (
        (2.5, "hist_watch_decay_mean_2.5"),
        (5.0, "hist_watch_decay_mean_5"),
    ):
        valid = [index for index in prior if np.isfinite(values[index])]
        if valid:
            weights = np.asarray(
                [
                    0.5
                    ** (((target_row[TIME] - rows[index][TIME]) / MS_PER_DAY) / halflife)
                    for index in valid
                ],
                dtype=np.float64,
            )
            result[name] = float(
                np.sum(weights * values[np.asarray(valid)]) / np.sum(weights)
            )
        else:
            result[name] = np.nan

    recent = prior[-5:]
    recent_values = [values[index] for index in recent if np.isfinite(values[index])]
    result["hist_watch_last5_mean"] = (
        float(np.mean(recent_values)) if recent_values else np.nan
    )
    tab_prior = [
        index
        for index in prior
        if rows[index][TAB] == target_row[TAB] and np.isfinite(values[index])
    ]
    if tab_prior:
        weights = np.asarray(
            [
                0.5
                ** (
                    ((target_row[TIME] - rows[index][TIME]) / MS_PER_DAY)
                    / TAB_HALFLIFE
                )
                for index in tab_prior
            ],
            dtype=np.float64,
        )
        result["hist_watch_tab_decay_mean_3"] = float(
            np.sum(weights * values[np.asarray(tab_prior)]) / np.sum(weights)
        )
    else:
        result["hist_watch_tab_decay_mean_3"] = np.nan
    result["strict_prior_count"] = len(prior)
    result["same_timestamp_excluded"] = sum(
        rows[index][TIME] == target_row[TIME] and index != target
        for index in user_indices
    )
    result["same_date_earlier_included"] = sum(
        rows[index][DATE] == target_row[DATE] and rows[index][TIME] < target_row[TIME]
        for index in user_indices
    )
    result["same_date_not_earlier_excluded"] = sum(
        rows[index][DATE] == target_row[DATE]
        and rows[index][TIME] >= target_row[TIME]
        and index != target
        for index in user_indices
    )
    return result


def verify_causality(rows, values, outputs, debug, lengths):
    rng = np.random.default_rng(20260901)
    valid_start = lengths["train"]
    valid_stop = valid_start + lengths["valid"]
    finite_targets = np.flatnonzero(
        np.isfinite(outputs["hist_watch_decay_mean_2.5"][valid_start:valid_stop])
    ) + valid_start
    random_targets = rng.choice(finite_targets, size=8, replace=False).tolist()
    same_date_pool = [
        index
        for index in debug["same_date_earlier_targets"]
        if valid_start <= index < valid_stop
    ]
    same_date_targets = (
        rng.choice(same_date_pool, size=min(4, len(same_date_pool)), replace=False).tolist()
        if same_date_pool
        else []
    )
    tied_targets = [
        index
        for group in debug["same_timestamp_groups"]
        for index in group[:1]
        if valid_start <= index < valid_stop
    ][:4]
    targets = list(dict.fromkeys(random_targets + same_date_targets + tied_targets))

    checks = []
    global_max_error = 0.0
    for target in targets:
        user_indices = debug["by_user"][rows[target][USER]]
        manual = manual_features(target, rows, values, user_indices)
        errors = {}
        for name in FEATURES:
            actual = outputs[name][target]
            expected = manual[name]
            if np.isnan(actual) and np.isnan(expected):
                error = 0.0
            else:
                error = abs(actual - expected)
            errors[name] = float(error)
            global_max_error = max(global_max_error, float(error))
        checks.append(
            {
                "row_index": int(target),
                "date": int(rows[target][DATE]),
                "time_ms": int(rows[target][TIME]),
                "strict_prior_count": manual["strict_prior_count"],
                "same_timestamp_rows_explicitly_excluded": manual[
                    "same_timestamp_excluded"
                ],
                "same_date_earlier_rows_included": manual[
                    "same_date_earlier_included"
                ],
                "same_date_equal_or_later_rows_excluded": manual[
                    "same_date_not_earlier_excluded"
                ],
                "errors": errors,
                "max_abs_error": max(errors.values()),
            }
        )
    if global_max_error >= 1e-10:
        raise AssertionError(f"brute-force causal check failed: {global_max_error}")
    if not any(row["same_date_earlier_rows_included"] > 0 for row in checks):
        raise AssertionError("same-date earlier inclusion was not exercised")

    # Synthetic exact-time stress test guarantees coverage even if real ties
    # are absent or do not land in validation.
    synthetic = [
        (20220410, "tie", "0", 20.0, 100.0, 1000, 0, 0),
        (20220410, "tie", "0", 80.0, 100.0, 1000, 1, 1),
        (20220410, "tie", "0", 50.0, 100.0, 2000, 2, 1),
    ]
    synthetic_values = watch_values(synthetic)
    synthetic_outputs, _ = compute_features(synthetic, synthetic_values)
    for name in FEATURES:
        if not np.isnan(synthetic_outputs[name][0]) or not np.isnan(
            synthetic_outputs[name][1]
        ):
            raise AssertionError(f"same-timestamp leakage in synthetic {name}")
        if abs(synthetic_outputs[name][2] - 0.5) >= 1e-12:
            raise AssertionError(f"synthetic prior aggregation failed for {name}")

    return {
        "passed": True,
        "method": "independent brute-force recomputation over time_ms < target_time_ms",
        "random_seed": 20260901,
        "checks": checks,
        "rows_checked": len(checks),
        "global_max_abs_error": float(global_max_error),
        "real_exact_timestamp_groups": len(debug["same_timestamp_groups"]),
        "real_exact_timestamp_rows": int(
            sum(len(group) for group in debug["same_timestamp_groups"])
        ),
        "real_tied_targets_checked": len(tied_targets),
        "synthetic_same_timestamp_stress_test": "passed; tied rows saw no values from one another and the later row saw their mean",
        "same_date_earlier_inclusion_exercised": True,
        "current_row_outcome_access": "none; snapshot precedes group update",
    }


def split_array(values, lengths):
    output = {}
    start = 0
    for name in ("train", "valid", "test"):
        stop = start + lengths[name]
        output[name] = values[start:stop]
        start = stop
    if start != len(values):
        raise AssertionError("split lengths do not cover values")
    return output


def build_frames(data_dir: str):
    rows, lengths, distribution = load_raw_rows(data_dir)
    values = watch_values(rows)
    print("  computing timestamp-causal watch-depth histories", flush=True)
    outputs, debug = compute_features(rows, values)
    causality = verify_causality(rows, values, outputs, debug, lengths)

    base_features = load_module(BASE_FEATURES_PATH, "yixi6_features_for_yixi9")
    frames, y, users, base_metadata = base_features.load_frames(data_dir)
    if lengths != {name: len(frames[name]) for name in ("train", "valid", "test")}:
        raise AssertionError("raw/base split-length mismatch")
    start = 0
    for name in ("train", "valid", "test"):
        stop = start + lengths[name]
        raw_users = np.asarray([row[USER] for row in rows[start:stop]])
        raw_labels = np.asarray([row[LABEL] for row in rows[start:stop]], dtype=np.float32)
        if not np.array_equal(raw_users, np.asarray(users[name])):
            raise AssertionError(f"raw/base user alignment failed for {name}")
        if not np.array_equal(raw_labels, y[name]):
            raise AssertionError(f"raw/base label alignment failed for {name}")
        start = stop
    for feature_name, values_array in outputs.items():
        parts = split_array(values_array, lengths)
        for name in ("train", "valid", "test"):
            frames[name][feature_name] = parts[name]

    metadata = {
        "lengths": lengths,
        "watch_fraction_distribution": distribution,
        "causality": causality,
        "feature_definitions": {
            "hist_watch_decay_mean_2.5": "timestamp-decayed mean clipped watch fraction, user history, 2.5-day half-life",
            "hist_watch_decay_mean_5": "timestamp-decayed mean clipped watch fraction, user history, 5-day half-life",
            "hist_watch_last5_mean": "mean defined clipped fractions among the five strictly-prior interactions; invalid-duration rows occupy window positions but do not enter the mean",
            "hist_watch_tab_decay_mean_3": "timestamp-decayed mean clipped watch fraction for matching user/tab history, 3-day half-life",
        },
        "ordering": "per user by time_ms; all exact time_ms ties read before group update; orig_idx orders tied rows only for future last-five windows",
        "base_feature_metadata": base_metadata,
        "provenance": {
            "base_features": os.path.relpath(BASE_FEATURES_PATH, REPO_ROOT),
            "base_features_sha256": file_sha256(BASE_FEATURES_PATH),
            "raw_files": {
                filename: file_sha256(os.path.join(data_dir, filename))
                for filename in RAW_FILES
            },
        },
    }
    return frames, y, users, metadata


def load_frames(data_dir: str = DATA_DIR, use_cache: bool = True):
    if use_cache and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    payload = build_frames(data_dir)
    if use_cache:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


if __name__ == "__main__":
    _, _, _, metadata = load_frames(DATA_DIR, use_cache=False)
    print(metadata["watch_fraction_distribution"])
    print(metadata["causality"])
