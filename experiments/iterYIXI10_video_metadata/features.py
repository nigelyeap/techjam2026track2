"""Leakage-safe intrinsic video metadata for YIXI Section 6j.

Only ``video_features_basic_pure.csv`` is read.  The aggregate statistic file
is deliberately neither opened nor joined.  Upload age is calculated for each
impression from its own interaction date and the video's static upload date.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
BASE_FEATURES_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI9_watch_depth_history", "features.py"
)
BASIC_FILENAME = "video_features_basic_pure.csv"
CACHE_PATH = os.path.join(THIS_DIR, ".metadata_frames_v1.pkl")
RAW_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
MISSING = "__MISSING__"

CATEGORICAL_COLUMNS = (
    "meta_video_type",
    "meta_upload_type",
    "meta_music_type",
    "meta_tag",
    "meta_music_id",
)
CONTINUOUS_COLUMNS = ("video_age_days", "aspect_ratio")
PROHIBITED_AGGREGATES = (
    "play_cnt",
    "like_cnt",
    "complete_play_cnt",
    "show_cnt",
    "play_user_num",
    "play_duration",
    "valid_play_cnt",
    "long_time_play_cnt",
    "comment_cnt",
    "follow_cnt",
    "share_cnt",
)


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


def parse_upload_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_basic(data_dir: str):
    path = os.path.join(data_dir, BASIC_FILENAME)
    metadata = {}
    source_columns = None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        source_columns = list(reader.fieldnames or [])
        for row in reader:
            video_id = row["video_id"]
            if video_id in metadata:
                raise AssertionError(f"duplicate basic metadata video_id {video_id}")
            width = float(row["server_width"])
            height = float(row["server_height"])
            if not np.isfinite(width) or not np.isfinite(height) or height <= 0:
                raise AssertionError(f"invalid dimensions for video {video_id}")
            metadata[video_id] = {
                "upload_date": parse_upload_date(row["upload_dt"]),
                "meta_video_type": row["video_type"] or MISSING,
                "meta_upload_type": row["upload_type"] or MISSING,
                "meta_music_type": row["music_type"] or MISSING,
                "meta_tag": row["tag"] or MISSING,
                "meta_music_id": row["music_id"] or MISSING,
                "aspect_ratio": width / height,
            }
    if len(metadata) != 7583:
        raise AssertionError(f"unexpected basic metadata row count {len(metadata)}")
    if set(source_columns or []).intersection(PROHIBITED_AGGREGATES):
        raise AssertionError("aggregate statistic column unexpectedly found in basic source")
    return metadata, source_columns


def load_impression_keys(data_dir: str):
    splits = {name: [] for name in SPLITS}
    for filename in RAW_FILES:
        with open(os.path.join(data_dir, filename), newline="") as f:
            for row in csv.DictReader(f):
                date = int(row["date"])
                record = (
                    row["video_id"],
                    date,
                    row["user_id"],
                    1 if row["long_view"] != "0" else 0,
                )
                for name, (lo, hi) in SPLITS.items():
                    if lo <= date <= hi:
                        splits[name].append(record)
                        break
    return splits


def values_for_split(keys, metadata):
    values = {column: [] for column in CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS}
    missing_video_ids = []
    negative_ages = []
    spot_checks = []
    for position, (video_id, date_int, _user_id, _label) in enumerate(keys):
        item = metadata.get(video_id)
        if item is None:
            missing_video_ids.append(video_id)
            for column in CATEGORICAL_COLUMNS:
                values[column].append(MISSING)
            values["video_age_days"].append(np.nan)
            values["aspect_ratio"].append(np.nan)
            continue
        interaction_date = datetime.strptime(str(date_int), "%Y%m%d").date()
        age = (interaction_date - item["upload_date"]).days
        if age < 0:
            negative_ages.append((video_id, date_int, str(item["upload_date"]), age))
        for column in CATEGORICAL_COLUMNS:
            values[column].append(item[column])
        values["video_age_days"].append(float(age))
        values["aspect_ratio"].append(float(item["aspect_ratio"]))
        if position in (0, len(keys) // 4, len(keys) // 2, 3 * len(keys) // 4, len(keys) - 1):
            spot_checks.append(
                {
                    "row": int(position),
                    "video_id": video_id,
                    "interaction_date": int(date_int),
                    "upload_date": str(item["upload_date"]),
                    "video_age_days": int(age),
                }
            )
    if missing_video_ids:
        raise AssertionError(f"unmatched video ids: {len(missing_video_ids)}")
    if negative_ages:
        raise AssertionError(f"impressions before upload: {negative_ages[:3]}")
    return values, spot_checks


def summarize(values):
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.number):
        finite = array[np.isfinite(array.astype(np.float64))].astype(np.float64)
        return {
            "rows": int(len(array)),
            "finite_rows": int(len(finite)),
            "min": float(np.min(finite)),
            "q01": float(np.quantile(finite, 0.01)),
            "q50": float(np.quantile(finite, 0.50)),
            "q99": float(np.quantile(finite, 0.99)),
            "max": float(np.max(finite)),
            "unique": int(len(np.unique(finite))),
        }
    series = pd.Series(array, dtype="string")
    return {
        "rows": int(len(series)),
        "missing_sentinel_rows": int((series == MISSING).sum()),
        "unique_including_missing": int(series.nunique(dropna=False)),
        "top_values": {
            str(key): int(value)
            for key, value in series.value_counts(dropna=False).head(10).items()
        },
    }


def build_frames(data_dir: str):
    basic, source_columns = load_basic(data_dir)
    keys = load_impression_keys(data_dir)
    base = load_module(BASE_FEATURES_PATH, "yixi9_features_for_yixi10")
    frames, y, users, base_metadata = base.load_frames(data_dir)
    lengths = {name: len(keys[name]) for name in SPLITS}
    expected_lengths = {name: len(frames[name]) for name in SPLITS}
    if lengths != expected_lengths:
        raise AssertionError(f"raw/base length mismatch: {lengths} vs {expected_lengths}")

    raw_values = {}
    all_spot_checks = {}
    for name in SPLITS:
        frame_video_ids = frames[name]["video_id"].astype("string").to_numpy()
        raw_video_ids = np.asarray([row[0] for row in keys[name]], dtype=str)
        known_video = ~pd.isna(frame_video_ids)
        if not np.array_equal(frame_video_ids[known_video].astype(str), raw_video_ids[known_video]):
            raise AssertionError(f"known-video alignment failed for {name}")
        raw_users = np.asarray([row[2] for row in keys[name]])
        raw_labels = np.asarray([row[3] for row in keys[name]], dtype=np.float32)
        if not np.array_equal(raw_users, np.asarray(users[name])):
            raise AssertionError(f"user alignment failed for {name}")
        if not np.array_equal(raw_labels, y[name]):
            raise AssertionError(f"label alignment failed for {name}")
        raw_values[name], all_spot_checks[name] = values_for_split(keys[name], basic)

    # Vocabularies are learned from train only.  Unseen validation/test levels
    # become categorical missing values instead of gaining future knowledge.
    category_audit = {}
    for column in CATEGORICAL_COLUMNS:
        train_values = pd.Series(raw_values["train"][column], dtype="string")
        categories = list(pd.unique(train_values))
        dtype = pd.CategoricalDtype(categories=categories)
        category_audit[column] = {
            "train_categories": len(categories),
            "splits": {},
        }
        known = set(categories)
        for name in SPLITS:
            split_values = pd.Series(raw_values[name][column], dtype="string")
            unseen = ~split_values.isin(known)
            frames[name][column] = split_values.astype(dtype)
            category_audit[column]["splits"][name] = {
                "raw_unique": int(split_values.nunique(dropna=False)),
                "missing_sentinel_rows": int((split_values == MISSING).sum()),
                "unseen_rows_mapped_to_nan": int(unseen.sum()),
                "unseen_rate": float(unseen.mean()),
            }

    for column in CONTINUOUS_COLUMNS:
        for name in SPLITS:
            frames[name][column] = np.asarray(raw_values[name][column], dtype=np.float64)

    summaries = {
        column: {name: summarize(raw_values[name][column]) for name in SPLITS}
        for column in CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS
    }
    all_ages = np.concatenate(
        [np.asarray(raw_values[name]["video_age_days"]) for name in SPLITS]
    )
    if np.min(all_ages) < 0 or np.max(all_ages) != 29:
        raise AssertionError("unexpected video-age range")
    all_ratios = np.concatenate(
        [np.asarray(raw_values[name]["aspect_ratio"]) for name in SPLITS]
    )
    if not np.all(np.isfinite(all_ratios)):
        raise AssertionError("non-finite aspect ratio")

    available = set(frames["train"].columns)
    prohibited_present = sorted(available.intersection(PROHIBITED_AGGREGATES))
    if prohibited_present:
        raise AssertionError(f"prohibited aggregate reached frame: {prohibited_present}")
    metadata = {
        "lengths": lengths,
        "source": {
            "file_read": BASIC_FILENAME,
            "sha256": file_sha256(os.path.join(data_dir, BASIC_FILENAME)),
            "columns": source_columns,
            "video_rows": len(basic),
            "unique_video_ids": len(basic),
            "join_unmatched_rows": 0,
            "statistic_file_read": False,
        },
        "leakage_policy": {
            "allowed_source": BASIC_FILENAME,
            "prohibited_source": "video_features_statistic_pure.csv",
            "prohibited_aggregate_columns": list(PROHIBITED_AGGREGATES),
            "prohibited_columns_in_frames": prohibited_present,
            "upload_age": "row interaction_date minus static upload_dt; no negative ages",
            "categorical_vocabularies": "train-only; unseen future levels map to NaN",
        },
        "feature_definitions": {
            "video_age_days": "interaction date minus upload date in whole days, computed per impression",
            "meta_video_type": "static video_type, native categorical",
            "meta_upload_type": "static upload_type, native categorical",
            "meta_music_type": "static music_type, native categorical",
            "meta_tag": "static full tag string, native categorical",
            "aspect_ratio": "static server_width / server_height, raw continuous",
            "meta_music_id": "static music_id, native high-cardinality categorical; tested separately",
        },
        "category_audit": category_audit,
        "summaries": summaries,
        "age_spot_checks": all_spot_checks,
        "all_metadata_static_per_video": True,
        "base_feature_metadata": base_metadata,
        "provenance": {
            "base_features": os.path.relpath(BASE_FEATURES_PATH, REPO_ROOT),
            "base_features_sha256": file_sha256(BASE_FEATURES_PATH),
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
    print(metadata)
