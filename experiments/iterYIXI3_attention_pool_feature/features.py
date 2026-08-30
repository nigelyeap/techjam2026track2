"""Causal lightweight attention-pooling features for YIXI Section 6c.

The scaled-dot-product implementation and its frozen train-only item
embeddings are imported directly from iter32. This module changes the
downstream representation only: the pooled scalar is passed to a native tree
ranker instead of bucketed into an FM field.

All generated caches live in this experiment directory. Existing iter1-iter44
folders are read-only.
"""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import os
import pickle
from typing import Any

import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
YIXI2_FEATURES_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI2_feature_depth", "features.py"
)
ITER32_FEATURES_PATH = os.path.join(
    REPO_ROOT, "experiments", "iter32_sequence_attention", "data_ext.py"
)
CACHE_PATH = os.path.join(THIS_DIR, ".attention_cache_v1.pkl")

ATTENTION_WINDOWS = (20, 40)
ITEM_EMBEDDING_DIM = 8
ITEM_EMBEDDING_EPOCHS = 8
ITEM_EMBEDDING_LR = 0.005
ITEM_EMBEDDING_SEED = 0
H5_HALFLIFE = 5.0
ALPHA = 0.5
CACHE_VERSION = 1


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


yixi2 = load_module(YIXI2_FEATURES_PATH, "yixi2_features_for_yixi3")
iter32 = load_module(ITER32_FEATURES_PATH, "iter32_attention_for_yixi3")
de = yixi2.de


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_array(values: np.ndarray, lengths: dict[str, int]) -> dict[str, np.ndarray]:
    result = {}
    start = 0
    for name in ("train", "valid", "test"):
        end = start + lengths[name]
        result[name] = values[start:end]
        start = end
    assert start == len(values)
    return result


def compute_uniform_history_rates(rows: list[tuple]) -> dict[str, np.ndarray]:
    """Non-attention control: uniform label mean over the same causal windows."""
    result = {
        f"uniform_rate_{window}": np.full(len(rows), -1.0, dtype=np.float64)
        for window in ATTENTION_WINDOWS
    }
    by_user: dict[Any, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        by_user[row[de.IDX["user_id"]]].append(index)

    for indices in by_user.values():
        indices.sort(
            key=lambda index: (
                rows[index][de.IDX["time_ms"]],
                rows[index][de.IDX["orig_idx"]],
            )
        )
        history = collections.deque(maxlen=max(ATTENTION_WINDOWS))
        for index in indices:
            if history:
                labels = np.asarray(history, dtype=np.float64)
                for window in ATTENTION_WINDOWS:
                    result[f"uniform_rate_{window}"][index] = float(
                        np.mean(labels[-window:])
                    )
            history.append(float(rows[index][de.IDX["label"]]))
    return result


def verify_h5_causality(
    rows: list[tuple], train_len: int, valid_len: int, pos: np.ndarray, total: np.ndarray
) -> dict[str, Any]:
    """Independent direct-sum check of the reused 5-day decay pair."""
    start, end = train_len, train_len + valid_len
    candidates = np.flatnonzero(total[start:end, 0] > 0) + start
    positions = np.linspace(0, len(candidates) - 1, 3, dtype=int)
    checks = []
    max_error = 0.0
    same_date_excluded = 0
    for position in positions:
        target_index = int(candidates[position])
        target = rows[target_index]
        target_user = target[de.IDX["user_id"]]
        target_date = target[de.IDX["date"]]
        target_ordinal = de._date_to_ordinal(target_date)
        expected_pos = 0.0
        expected_total = 0.0
        target_same_date = 0
        for index, row in enumerate(rows):
            if index == target_index or row[de.IDX["user_id"]] != target_user:
                continue
            row_date = row[de.IDX["date"]]
            if row_date == target_date:
                target_same_date += 1
            if row_date >= target_date:
                continue
            gap = target_ordinal - de._date_to_ordinal(row_date)
            weight = 0.5 ** (gap / H5_HALFLIFE)
            expected_total += weight
            if row[de.IDX["label"]] == 1:
                expected_pos += weight
        error = max(
            abs(float(pos[target_index, 0]) - expected_pos),
            abs(float(total[target_index, 0]) - expected_total),
        )
        max_error = max(max_error, error)
        same_date_excluded += target_same_date
        checks.append(
            {
                "row_index": target_index,
                "date": int(target_date),
                "same_date_matching_rows_excluded": target_same_date,
                "max_abs_error": error,
            }
        )
    passed = max_error < 1e-10 and same_date_excluded > 0
    if not passed:
        raise AssertionError("5-day decay causal verification failed")
    return {
        "checks": checks,
        "max_abs_error": max_error,
        "same_date_matching_rows_excluded": same_date_excluded,
        "semantics": "strict prior dates only",
        "passed": passed,
    }


def _manual_attention_rate(
    rows: list[tuple], history_indices: list[int], target_index: int, item_embeddings, window: int
) -> float:
    history_indices = history_indices[-window:]
    zero = np.zeros(ITEM_EMBEDDING_DIM, dtype=np.float64)
    candidate = item_embeddings.get(rows[target_index][de.IDX["video_id"]], zero)
    history_embeddings = np.stack(
        [item_embeddings.get(rows[index][de.IDX["video_id"]], zero) for index in history_indices]
    )
    history_labels = np.asarray(
        [rows[index][de.IDX["label"]] for index in history_indices], dtype=np.float64
    )
    logits = history_embeddings.dot(candidate) / np.sqrt(ITEM_EMBEDDING_DIM)
    weights = np.exp(logits - np.max(logits))
    weights /= np.sum(weights)
    return float(np.dot(weights, history_labels))


def verify_attention_causality(
    rows: list[tuple],
    train_len: int,
    valid_len: int,
    attention: dict[str, np.ndarray],
    uniform: dict[str, np.ndarray],
    item_embeddings,
) -> dict[str, Any]:
    """Independently reconstruct selected validation rows from prior history."""
    by_user: dict[Any, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        by_user[row[de.IDX["user_id"]]].append(index)
    for indices in by_user.values():
        indices.sort(
            key=lambda index: (
                rows[index][de.IDX["time_ms"]],
                rows[index][de.IDX["orig_idx"]],
            )
        )

    start, end = train_len, train_len + valid_len
    covered = np.flatnonzero(attention["attn_rate_40"][start:end] >= 0) + start
    positions = np.linspace(0, len(covered) - 1, 5, dtype=int)
    checks = []
    max_error = 0.0
    same_time_nonprior_excluded = 0
    for position in positions:
        target_index = int(covered[position])
        target = rows[target_index]
        target_key = (target[de.IDX["time_ms"]], target[de.IDX["orig_idx"]])
        user_indices = by_user[target[de.IDX["user_id"]]]
        prior = [
            index
            for index in user_indices
            if (rows[index][de.IDX["time_ms"]], rows[index][de.IDX["orig_idx"]])
            < target_key
        ]
        same_time_nonprior = sum(
            rows[index][de.IDX["time_ms"]] == target[de.IDX["time_ms"]]
            and rows[index][de.IDX["orig_idx"]] >= target[de.IDX["orig_idx"]]
            for index in user_indices
        )
        same_time_nonprior_excluded += same_time_nonprior
        row_errors = []
        for window in ATTENTION_WINDOWS:
            tail = prior[-window:]
            expected_attention = _manual_attention_rate(
                rows, tail, target_index, item_embeddings, window
            )
            expected_uniform = float(
                np.mean([rows[index][de.IDX["label"]] for index in tail])
            )
            row_errors.extend(
                [
                    abs(attention[f"attn_rate_{window}"][target_index] - expected_attention),
                    abs(uniform[f"uniform_rate_{window}"][target_index] - expected_uniform),
                ]
            )
        error = float(max(row_errors))
        max_error = max(max_error, error)
        checks.append(
            {
                "row_index": target_index,
                "history_rows_available": len(prior),
                "same_time_nonprior_rows_excluded": same_time_nonprior,
                "max_abs_error": error,
            }
        )

    zero_history_checked = 0
    for indices in by_user.values():
        first = indices[0]
        for window in ATTENTION_WINDOWS:
            if attention[f"attn_rate_{window}"][first] != -1.0:
                raise AssertionError("attention zero-history sentinel mismatch")
            if uniform[f"uniform_rate_{window}"][first] != -1.0:
                raise AssertionError("uniform zero-history sentinel mismatch")
        zero_history_checked += 1
        if zero_history_checked == 20:
            break

    passed = max_error < 1e-12 and zero_history_checked == 20
    if not passed:
        raise AssertionError("attention causal verification failed")
    return {
        "checks": checks,
        "max_abs_error": max_error,
        "same_time_nonprior_rows_excluded": same_time_nonprior_excluded,
        "zero_history_sentinels_checked": zero_history_checked,
        "semantics": "strict prior (time_ms, orig_idx) only",
        "passed": passed,
    }


def numeric_diagnostic(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "rows": int(len(values)),
        "finite_fraction": float(np.mean(np.isfinite(values))),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _cache_identity() -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "yixi2_sha256": file_sha256(YIXI2_FEATURES_PATH),
        "iter32_sha256": file_sha256(ITER32_FEATURES_PATH),
        "windows": list(ATTENTION_WINDOWS),
        "embedding_dim": ITEM_EMBEDDING_DIM,
        "embedding_epochs": ITEM_EMBEDDING_EPOCHS,
        "embedding_lr": ITEM_EMBEDDING_LR,
        "embedding_seed": ITEM_EMBEDDING_SEED,
        "h5_halflife": H5_HALFLIFE,
    }


def prepare_feature_frames(data_dir: str):
    """Return native frames with exact h5 and lightweight attention features."""
    base_frames, y, users = yixi2.iter44.prepare(data_dir)
    splits = de.load_ext(data_dir)
    lengths = {name: len(splits[name]) for name in ("train", "valid", "test")}
    flat = splits["train"] + splits["valid"] + splits["test"]

    for name in ("train", "valid", "test"):
        split_labels = np.asarray(
            [row[de.IDX["label"]] for row in splits[name]], dtype=np.float32
        )
        split_users = np.asarray([row[de.IDX["user_id"]] for row in splits[name]])
        assert np.array_equal(split_labels, y[name]), f"{name} label alignment mismatch"
        assert np.array_equal(split_users, np.asarray(users[name])), (
            f"{name} user alignment mismatch"
        )

    identity = _cache_identity()
    payload = None
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            candidate = pickle.load(f)
        if candidate.get("identity") == identity and candidate.get("lengths") == lengths:
            payload = candidate
            print("  loaded YIXI3 attention cache", flush=True)

    if payload is None:
        print("  computing exact reused 5-day user decay", flush=True)
        decay_pos, decay_total = de.compute_decay_features(flat, [H5_HALFLIFE])
        h5_causality = verify_h5_causality(
            flat, lengths["train"], lengths["valid"], decay_pos, decay_total
        )

        print("  fitting frozen train-only item embeddings", flush=True)
        item_embeddings = iter32.pretrain_item_embeddings(
            splits["train"],
            k=ITEM_EMBEDDING_DIM,
            epochs=ITEM_EMBEDDING_EPOCHS,
            lr=ITEM_EMBEDDING_LR,
            seed=ITEM_EMBEDDING_SEED,
        )
        print("  computing scaled-dot attention pools (K=20,40)", flush=True)
        attention = iter32.compute_attention_features(
            flat,
            item_embeddings,
            ITEM_EMBEDDING_DIM,
            windows=ATTENTION_WINDOWS,
            decay_halflives=(),
        )
        uniform = compute_uniform_history_rates(flat)
        attention_causality = verify_attention_causality(
            flat,
            lengths["train"],
            lengths["valid"],
            attention,
            uniform,
            item_embeddings,
        )
        embedding_norms = np.asarray(
            [np.linalg.norm(vector) for vector in item_embeddings.values()], dtype=np.float64
        )
        payload = {
            "identity": identity,
            "lengths": lengths,
            "decay_pos": decay_pos,
            "decay_total": decay_total,
            "attention": attention,
            "uniform": uniform,
            "causality": {
                "h5_decay": h5_causality,
                "attention": attention_causality,
                "passed": h5_causality["passed"] and attention_causality["passed"],
            },
            "embedding_diagnostics": {
                "items": int(len(item_embeddings)),
                "mean_l2_norm": float(np.mean(embedding_norms)),
                "std_l2_norm": float(np.std(embedding_norms)),
            },
        }
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  wrote {CACHE_PATH}", flush=True)

    if not payload["causality"]["passed"]:
        raise AssertionError("cached causal verification did not pass")

    frames = {name: base_frames[name].copy() for name in ("train", "valid", "test")}
    h5_rate = (payload["decay_pos"][:, 0] + ALPHA) / (
        payload["decay_total"][:, 0] + 2 * ALPHA
    )
    values = {
        "decay_rate_5": h5_rate,
        "decay_act_5": payload["decay_total"][:, 0],
        **payload["attention"],
        **payload["uniform"],
    }
    split_values = {column: split_array(array, lengths) for column, array in values.items()}
    for name, frame in frames.items():
        for column, by_split in split_values.items():
            frame[column] = by_split[name]

    train = frames["train"]
    diagnostics: dict[str, Any] = {
        column: numeric_diagnostic(train[column].to_numpy()) for column in values
    }
    for window in ATTENTION_WINDOWS:
        attn = train[f"attn_rate_{window}"].to_numpy(dtype=np.float64)
        uniform = train[f"uniform_rate_{window}"].to_numpy(dtype=np.float64)
        covered = attn >= 0
        diagnostics[f"attn_rate_{window}"]["coverage"] = float(np.mean(covered))
        diagnostics[f"attn_rate_{window}"]["mean_abs_delta_vs_uniform"] = float(
            np.mean(np.abs(attn[covered] - uniform[covered]))
        )
        diagnostics[f"attn_rate_{window}"]["corr_with_uniform"] = float(
            np.corrcoef(attn[covered], uniform[covered])[0, 1]
        )
        diagnostics[f"attn_rate_{window}"]["corr_with_decay_rate_5"] = float(
            np.corrcoef(
                attn[covered], train["decay_rate_5"].to_numpy(dtype=np.float64)[covered]
            )[0, 1]
        )
        diagnostics[f"attn_rate_{window}"]["corr_with_lastk_rate"] = float(
            np.corrcoef(
                attn[covered], train["lastk_rate"].to_numpy(dtype=np.float64)[covered]
            )[0, 1]
        )

    base_columns = list(yixi2.iter44.CAT_COLS + yixi2.iter44.NUM_COLS)
    h5_columns = [
        column
        for column in base_columns
        if column not in ("decay_rate_2.5", "decay_act_2.5")
    ] + ["decay_rate_5", "decay_act_5"]
    metadata = {
        "identity": identity,
        "lengths": lengths,
        "base_columns": base_columns,
        "h5_columns": h5_columns,
        "attention_columns": [f"attn_rate_{window}" for window in ATTENTION_WINDOWS],
        "uniform_control_columns": [
            f"uniform_rate_{window}" for window in ATTENTION_WINDOWS
        ],
        "causality": payload["causality"],
        "embedding_diagnostics": payload["embedding_diagnostics"],
        "diagnostics": diagnostics,
        "feature_definition": (
            "frozen k=8 train-only item embeddings; scaled dot-product softmax over "
            "strictly prior user interactions; attention-weighted pool of prior labels"
        ),
    }
    return frames, y, users, metadata
