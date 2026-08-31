"""Exact causal feature union needed by YIXI Section 6f.

The tab count/rate comes directly from iter63's extended data module.  The
5-day user pair is computed by that same module's unchanged user-decay
function, which is identical to the implementation reused by YIXI4.  This
module only aligns those already-established definitions into one native
DataFrame; it does not redesign either feature.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
ITER63_TRAIN_PATH = os.path.join(
    REPO_ROOT, "experiments", "iter63_decay_tab_rate", "train.py"
)
ITER63_DATA_PATH = os.path.join(
    REPO_ROOT, "experiments", "iter63_decay_tab_rate", "data_ext.py"
)
YIXI2_FEATURES_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI2_feature_depth", "features.py"
)
CACHE_PATH = os.path.join(THIS_DIR, ".cross_transfer_frames_v1.pkl")

CAT_COLS = ["user_id", "video_id", "author_id", "tab", "last1"]
ALPHA = 0.5


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


iter63 = load_module(ITER63_TRAIN_PATH, "iter63_train_for_yixi6_features")
de = iter63._de


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_array(values: np.ndarray, lengths: dict[str, int]):
    output = {}
    start = 0
    for name in ("train", "valid", "test"):
        stop = start + lengths[name]
        output[name] = values[start:stop]
        start = stop
    if start != len(values):
        raise AssertionError("split lengths do not cover feature array")
    return output


def _pick_validation_targets(flat, valid_start: int, valid_stop: int) -> list[int]:
    candidates = []
    for idx in np.linspace(valid_start, valid_stop - 1, 31, dtype=int):
        row = flat[idx]
        date, user_id, tab = row[de.IDX["date"]], row[de.IDX["user_id"]], row[de.IDX["tab"]]
        has_user_history = any(
            prior[de.IDX["user_id"]] == user_id and prior[de.IDX["date"]] < date
            for prior in flat[:valid_start]
        )
        has_tab_history = any(
            prior[de.IDX["user_id"]] == user_id
            and prior[de.IDX["tab"]] == tab
            and prior[de.IDX["date"]] < date
            for prior in flat[:valid_start]
        )
        if has_user_history and has_tab_history:
            candidates.append(int(idx))
        if len(candidates) == 3:
            break
    if len(candidates) != 3:
        raise AssertionError("could not find three validation rows with both histories")
    return candidates


def verify_causality(flat, lengths, h5_pos, h5_total) -> dict[str, Any]:
    """Independent direct sums using strictly earlier dates only."""
    valid_start = lengths["train"]
    valid_stop = valid_start + lengths["valid"]
    targets = _pick_validation_targets(flat, valid_start, valid_stop)
    tab_pos_col = de._tab_halflife_col(3, de.HALFLIVES, de.TAB_HALFLIVES)
    tab_total_col = de._tab_halflife_total_col(3, de.HALFLIVES, de.TAB_HALFLIVES)
    checks = []
    global_max_error = 0.0
    same_date_excluded = 0

    for idx in targets:
        row = flat[idx]
        date = row[de.IDX["date"]]
        user_id = row[de.IDX["user_id"]]
        tab = row[de.IDX["tab"]]
        target_ord = de._date_to_ordinal(date)
        earlier_user = [
            prior
            for prior in flat
            if prior[de.IDX["user_id"]] == user_id and prior[de.IDX["date"]] < date
        ]
        earlier_tab = [
            prior
            for prior in earlier_user
            if prior[de.IDX["tab"]] == tab
        ]
        same_date_excluded += sum(
            prior[de.IDX["user_id"]] == user_id and prior[de.IDX["date"]] == date
            for prior in flat
        )
        same_date_excluded += sum(
            prior[de.IDX["user_id"]] == user_id
            and prior[de.IDX["tab"]] == tab
            and prior[de.IDX["date"]] == date
            for prior in flat
        )

        manual_h5_pos = sum(
            float(prior[de.IDX["label"]])
            * 0.5 ** ((target_ord - de._date_to_ordinal(prior[de.IDX["date"]])) / 5.0)
            for prior in earlier_user
        )
        manual_h5_total = sum(
            0.5 ** ((target_ord - de._date_to_ordinal(prior[de.IDX["date"]])) / 5.0)
            for prior in earlier_user
        )
        manual_tab_pos = sum(
            float(prior[de.IDX["label"]])
            * 0.5 ** ((target_ord - de._date_to_ordinal(prior[de.IDX["date"]])) / 3.0)
            for prior in earlier_tab
        )
        manual_tab_total = sum(
            0.5 ** ((target_ord - de._date_to_ordinal(prior[de.IDX["date"]])) / 3.0)
            for prior in earlier_tab
        )
        errors = {
            "h5_pos": abs(manual_h5_pos - h5_pos[idx, 0]),
            "h5_total": abs(manual_h5_total - h5_total[idx, 0]),
            "tab_h3_pos": abs(manual_tab_pos - row[tab_pos_col]),
            "tab_h3_total": abs(manual_tab_total - row[tab_total_col]),
        }
        max_error = float(max(errors.values()))
        global_max_error = max(global_max_error, max_error)
        checks.append(
            {
                "row_index": idx,
                "date": int(date),
                "errors": {name: float(value) for name, value in errors.items()},
                "max_abs_error": max_error,
            }
        )

    if global_max_error >= 1e-10:
        raise AssertionError(f"cross-transfer causality check failed: {global_max_error}")
    return {
        "passed": True,
        "method": "independent direct sums over strictly earlier dates",
        "checks": checks,
        "global_max_abs_error": float(global_max_error),
        "same_date_matching_rows_explicitly_excluded": int(same_date_excluded),
    }


def build_frames(data_dir: str):
    print("  rebuilding iter63 causal rows without writing shared caches", flush=True)
    splits = de.load_ext(data_dir, use_cache=False)
    lengths = {name: len(splits[name]) for name in ("train", "valid", "test")}
    flat = splits["train"] + splits["valid"] + splits["test"]

    print("  computing exact 5-day user pair with imported decay function", flush=True)
    h5_pos, h5_total = de.compute_decay_features(flat, [5.0])
    causality = verify_causality(flat, lengths, h5_pos, h5_total)
    h5_pos_split = split_array(h5_pos[:, 0], lengths)
    h5_total_split = split_array(h5_total[:, 0], lengths)

    frames = {}
    y = {}
    users = {}
    for name in ("train", "valid", "test"):
        rows = [iter63._row_to_dict(row, "plus_rate") for row in splits[name]]
        frame = pd.DataFrame(rows)
        frame["decay_rate_5"] = (
            h5_pos_split[name] + ALPHA
        ) / (h5_total_split[name] + 2 * ALPHA)
        frame["decay_act_5"] = h5_total_split[name]
        frame["decay_rate_x_log_activity"] = (
            frame["decay_rate_2.5"].to_numpy(dtype=np.float64)
            * np.log1p(frame["decay_act_2.5"].to_numpy(dtype=np.float64))
        )
        frames[name] = frame
        y[name] = np.asarray(
            [row[de.IDX["label"]] for row in splits[name]], dtype=np.float32
        )
        users[name] = np.asarray([row[de.IDX["user_id"]] for row in splits[name]])

    categories = {
        column: pd.CategoricalDtype(categories=frames["train"][column].unique())
        for column in CAT_COLS
    }
    for frame in frames.values():
        for column in CAT_COLS:
            frame[column] = frame[column].astype(categories[column])

    metadata = {
        "lengths": lengths,
        "causality": causality,
        "provenance": {
            "iter63_train": os.path.relpath(ITER63_TRAIN_PATH, REPO_ROOT),
            "iter63_train_sha256": file_sha256(ITER63_TRAIN_PATH),
            "iter63_data": os.path.relpath(ITER63_DATA_PATH, REPO_ROOT),
            "iter63_data_sha256": file_sha256(ITER63_DATA_PATH),
            "yixi4_feature_source": os.path.relpath(YIXI2_FEATURES_PATH, REPO_ROOT),
            "yixi4_feature_source_sha256": file_sha256(YIXI2_FEATURES_PATH),
            "definitions_redesigned": False,
        },
        "available_columns": list(frames["train"].columns),
    }
    return frames, y, users, metadata


def load_frames(data_dir: str, use_cache: bool = True):
    if use_cache and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    payload = build_frames(data_dir)
    if use_cache:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


if __name__ == "__main__":
    data_dir = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
    _, _, _, metadata = load_frames(data_dir, use_cache=False)
    print(metadata["causality"])
