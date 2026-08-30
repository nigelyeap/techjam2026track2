"""Causal feature construction for YIXI Section 6b.

This module extends iter44's native DataFrames without changing any existing
experiment. All history-derived values use the same two-phase, date-grouped
semantics as iter27: a row can see prior dates only, never another row from
the same date and never a future label.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
ITER44_TRAIN = os.path.join(REPO_ROOT, "experiments", "iter44_gbm_native_features", "train.py")

USER_HALFLIVES = (1.0, 5.0, 7.0, 14.0)
TAB_HALFLIVES = (1.0, 5.0, 7.0, 14.0)
POPULARITY_HALFLIFE = 2.5
ALPHA = 0.5


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iter44 = _load_module(ITER44_TRAIN, "iter44_train_for_yixi2")
de = iter44._de


def fmt_h(halflife: float) -> str:
    return str(int(halflife)) if float(halflife).is_integer() else str(halflife)


def compute_keyed_decay_features(
    rows: list[tuple], key_index: int, halflives: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Exact lazy exponential decay keyed by any row field.

    This deliberately mirrors iter27's `compute_decay_features`, changing
    only the state key from user_id to the requested author_id/video_id.
    The read phase for a date completes before the update phase, so same-date
    rows cannot observe one another.
    """
    halflives = tuple(float(h) for h in halflives)
    n = len(rows)
    h_count = len(halflives)
    order = sorted(range(n), key=lambda i: rows[i][de.IDX["date"]])
    day_mult = np.asarray([0.5 ** (1.0 / h) for h in halflives], dtype=np.float64)

    decayed_pos = np.zeros((n, h_count), dtype=np.float64)
    decayed_total = np.zeros((n, h_count), dtype=np.float64)
    last_ord: dict[Any, int] = {}
    pos_state: dict[Any, np.ndarray] = {}
    total_state: dict[Any, np.ndarray] = {}

    i = 0
    while i < n:
        j = i + 1
        date = rows[order[i]][de.IDX["date"]]
        while j < n and rows[order[j]][de.IDX["date"]] == date:
            j += 1
        day_indices = order[i:j]
        date_ord = de._date_to_ordinal(date)

        # Read old state only.
        for idx in day_indices:
            key = rows[idx][key_index]
            previous_ord = last_ord.get(key)
            if previous_ord is not None:
                factors = day_mult ** (date_ord - previous_ord)
                decayed_pos[idx] = pos_state[key] * factors
                decayed_total[idx] = total_state[key] * factors

        # Aggregate the entire day, then update state once per key.
        day_pos: dict[Any, int] = {}
        day_total: dict[Any, int] = {}
        for idx in day_indices:
            row = rows[idx]
            key = row[key_index]
            day_total[key] = day_total.get(key, 0) + 1
            if row[de.IDX["label"]] == 1:
                day_pos[key] = day_pos.get(key, 0) + 1

        for key, total_count in day_total.items():
            previous_ord = last_ord.get(key)
            if previous_ord is None:
                new_pos = np.zeros(h_count, dtype=np.float64)
                new_total = np.zeros(h_count, dtype=np.float64)
            else:
                factors = day_mult ** (date_ord - previous_ord)
                new_pos = pos_state[key] * factors
                new_total = total_state[key] * factors
            new_pos = new_pos + day_pos.get(key, 0)
            new_total = new_total + total_count
            pos_state[key] = new_pos
            total_state[key] = new_total
            last_ord[key] = date_ord
        i = j

    return decayed_pos, decayed_total


def _split_array(values: np.ndarray, lengths: dict[str, int]) -> dict[str, np.ndarray]:
    out = {}
    start = 0
    for name in ("train", "valid", "test"):
        end = start + lengths[name]
        out[name] = values[start:end]
        start = end
    assert start == len(values)
    return out


def _brute_decay(
    rows: list[tuple],
    target_idx: int,
    key_fn: Callable[[tuple], Any],
    halflives: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Independent direct-sum recount over strictly earlier dates."""
    target = rows[target_idx]
    target_date = target[de.IDX["date"]]
    target_ord = de._date_to_ordinal(target_date)
    target_key = key_fn(target)
    pos = np.zeros(len(halflives), dtype=np.float64)
    total = np.zeros(len(halflives), dtype=np.float64)
    same_date_matches = 0
    for idx, row in enumerate(rows):
        if idx == target_idx or key_fn(row) != target_key:
            continue
        row_date = row[de.IDX["date"]]
        if row_date == target_date:
            same_date_matches += 1
        if row_date >= target_date:
            continue
        gap = target_ord - de._date_to_ordinal(row_date)
        weights = np.asarray([0.5 ** (gap / h) for h in halflives])
        total += weights
        if row[de.IDX["label"]] == 1:
            pos += weights
    return pos, total, same_date_matches


def _pick_targets(values: np.ndarray, start: int, end: int, count: int = 3) -> list[int]:
    if values.ndim == 2:
        mask = np.any(values[start:end] > 0, axis=1)
    else:
        mask = values[start:end] > 0
    candidates = np.flatnonzero(mask) + start
    if not len(candidates):
        raise AssertionError("no nonzero validation targets available for causal check")
    positions = np.linspace(0, len(candidates) - 1, min(count, len(candidates)), dtype=int)
    return [int(candidates[p]) for p in positions]


def verify_causality(
    rows: list[tuple],
    train_len: int,
    valid_len: int,
    user_pos: np.ndarray,
    user_total: np.ndarray,
    tab_pos: np.ndarray,
    author_pos: np.ndarray,
    author_total: np.ndarray,
    video_pos: np.ndarray,
    video_total: np.ndarray,
) -> dict[str, Any]:
    """Brute-force independent checks for every new history family."""
    start, end = train_len, train_len + valid_len
    report: dict[str, Any] = {"families": {}, "semantics": "strict prior dates only"}

    specs = [
        (
            "user_decay",
            lambda row: row[de.IDX["user_id"]],
            tuple(USER_HALFLIVES),
            user_pos,
            user_total,
            False,
        ),
        (
            "user_tab_decay",
            lambda row: (row[de.IDX["user_id"]], row[de.IDX["tab"]]),
            tuple(TAB_HALFLIVES),
            tab_pos,
            None,
            True,
        ),
        (
            "author_decay",
            lambda row: row[de.IDX["author_id"]],
            (POPULARITY_HALFLIFE,),
            author_pos,
            author_total,
            False,
        ),
        (
            "video_decay",
            lambda row: row[de.IDX["video_id"]],
            (POPULARITY_HALFLIFE,),
            video_pos,
            video_total,
            False,
        ),
    ]

    global_max_error = 0.0
    total_same_date_matches = 0
    for name, key_fn, halflives, computed_pos, computed_total, pos_only in specs:
        target_basis = computed_pos if pos_only else computed_total
        targets = _pick_targets(target_basis, start, end)
        checks = []
        family_max_error = 0.0
        for target_idx in targets:
            brute_pos, brute_total, same_date_matches = _brute_decay(
                rows, target_idx, key_fn, halflives
            )
            pos_error = float(np.max(np.abs(computed_pos[target_idx] - brute_pos)))
            total_error = 0.0
            if computed_total is not None:
                total_error = float(np.max(np.abs(computed_total[target_idx] - brute_total)))
            error = max(pos_error, total_error)
            family_max_error = max(family_max_error, error)
            total_same_date_matches += same_date_matches
            checks.append(
                {
                    "row_index": target_idx,
                    "date": int(rows[target_idx][de.IDX["date"]]),
                    "same_date_matching_rows_excluded": same_date_matches,
                    "max_abs_error": error,
                }
            )
        report["families"][name] = {
            "halflives": list(halflives),
            "checks": checks,
            "max_abs_error": family_max_error,
        }
        global_max_error = max(global_max_error, family_max_error)

    report["global_max_abs_error"] = global_max_error
    report["same_date_matching_rows_excluded_across_checks"] = total_same_date_matches
    report["passed"] = global_max_error < 1e-10 and total_same_date_matches > 0
    if not report["passed"]:
        raise AssertionError(f"causality verification failed: {report}")
    return report


def _numeric_diagnostic(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "rows": int(len(values)),
        "finite_fraction": float(np.mean(np.isfinite(values))),
        "nonzero_fraction": float(np.mean(np.abs(values) > 0)),
        "mean": float(np.mean(finite)) if len(finite) else float("nan"),
        "std": float(np.std(finite)) if len(finite) else float("nan"),
        "min": float(np.min(finite)) if len(finite) else float("nan"),
        "max": float(np.max(finite)) if len(finite) else float("nan"),
    }


def prepare_feature_frames(data_dir: str):
    """Return all candidate features aligned to iter44's native rows."""
    base_dfs, y, users = iter44.prepare(data_dir)
    splits = de.load_ext(data_dir)
    lengths = {name: len(splits[name]) for name in ("train", "valid", "test")}
    flat = splits["train"] + splits["valid"] + splits["test"]

    # Alignment checks make the join-by-row-index explicit and fail closed.
    for name in ("train", "valid", "test"):
        labels = np.asarray([row[de.IDX["label"]] for row in splits[name]], dtype=np.float32)
        row_users = np.asarray([row[de.IDX["user_id"]] for row in splits[name]])
        assert np.array_equal(labels, y[name]), f"{name} label alignment mismatch"
        assert np.array_equal(row_users, np.asarray(users[name])), f"{name} user alignment mismatch"

    print("  computing expanded user half-lives", flush=True)
    user_pos, user_total = de.compute_decay_features(flat, list(USER_HALFLIVES))
    print("  computing expanded user-tab half-lives", flush=True)
    tab_pos = de.compute_decay_tab_features(flat, list(TAB_HALFLIVES))
    print("  computing author popularity decay", flush=True)
    author_pos, author_total = compute_keyed_decay_features(
        flat, de.IDX["author_id"], (POPULARITY_HALFLIFE,)
    )
    print("  computing video popularity decay", flush=True)
    video_pos, video_total = compute_keyed_decay_features(
        flat, de.IDX["video_id"], (POPULARITY_HALFLIFE,)
    )

    causality_report = verify_causality(
        flat,
        lengths["train"],
        lengths["valid"],
        user_pos,
        user_total,
        tab_pos,
        author_pos,
        author_total,
        video_pos,
        video_total,
    )

    frames = {name: base_dfs[name].copy() for name in ("train", "valid", "test")}
    diagnostics: dict[str, Any] = {}

    for col_idx, halflife in enumerate(USER_HALFLIVES):
        suffix = fmt_h(halflife)
        rate_all = (user_pos[:, col_idx] + ALPHA) / (user_total[:, col_idx] + 2 * ALPHA)
        rate_splits = _split_array(rate_all, lengths)
        total_splits = _split_array(user_total[:, col_idx], lengths)
        for name in frames:
            frames[name][f"decay_rate_{suffix}"] = rate_splits[name]
            frames[name][f"decay_act_{suffix}"] = total_splits[name]
        diagnostics[f"decay_rate_{suffix}"] = _numeric_diagnostic(rate_splits["train"])
        diagnostics[f"decay_act_{suffix}"] = _numeric_diagnostic(total_splits["train"])

    for col_idx, halflife in enumerate(TAB_HALFLIVES):
        suffix = fmt_h(halflife)
        value_splits = _split_array(tab_pos[:, col_idx], lengths)
        for name in frames:
            frames[name][f"decay_tab_{suffix}"] = value_splits[name]
        diagnostics[f"decay_tab_{suffix}"] = _numeric_diagnostic(value_splits["train"])

    popularity_arrays = {
        "author_decay_rate_2.5": (author_pos[:, 0] + ALPHA) / (author_total[:, 0] + 2 * ALPHA),
        "author_decay_act_2.5": author_total[:, 0],
        "video_decay_rate_2.5": (video_pos[:, 0] + ALPHA) / (video_total[:, 0] + 2 * ALPHA),
        "video_decay_act_2.5": video_total[:, 0],
    }
    for column, values in popularity_arrays.items():
        value_splits = _split_array(values, lengths)
        for name in frames:
            frames[name][column] = value_splits[name]
        diagnostics[column] = _numeric_diagnostic(value_splits["train"])

    # Three predeclared crosses. All inputs are either static or already
    # causal. Tier edges and categorical vocabulary are fit on train only.
    for name, frame in frames.items():
        frame["decay_rate_per_duration"] = (
            frame["decay_rate_2.5"].to_numpy(dtype=np.float64)
            / (frame["duration_ms"].to_numpy(dtype=np.float64) + 1.0)
        )
        frame["decay_rate_x_log_activity"] = (
            frame["decay_rate_2.5"].to_numpy(dtype=np.float64)
            * np.log1p(frame["decay_act_2.5"].to_numpy(dtype=np.float64))
        )

    tier_edges = np.unique(
        np.quantile(frames["train"]["decay_act_2.5"].to_numpy(dtype=np.float64), [0.25, 0.5, 0.75])
    )
    train_tab_categories = frames["train"]["tab"].cat.categories
    tab_stride = len(train_tab_categories) + 1
    cross_values: dict[str, np.ndarray] = {}
    for name, frame in frames.items():
        tiers = np.searchsorted(
            tier_edges, frame["decay_act_2.5"].to_numpy(dtype=np.float64), side="right"
        )
        tab_codes = frame["tab"].cat.codes.to_numpy(dtype=np.int64)
        tab_codes = np.where(tab_codes >= 0, tab_codes, len(train_tab_categories))
        cross_values[name] = tiers.astype(np.int64) * tab_stride + tab_codes
    cross_dtype = pd.CategoricalDtype(categories=np.unique(cross_values["train"]))
    for name, frame in frames.items():
        frame["activity_tier_x_tab"] = pd.Series(
            pd.Categorical(cross_values[name], dtype=cross_dtype), index=frame.index
        )

    diagnostics["decay_rate_per_duration"] = _numeric_diagnostic(
        frames["train"]["decay_rate_per_duration"].to_numpy()
    )
    diagnostics["decay_rate_x_log_activity"] = _numeric_diagnostic(
        frames["train"]["decay_rate_x_log_activity"].to_numpy()
    )
    diagnostics["activity_tier_x_tab"] = {
        "train_categories": int(len(cross_dtype.categories)),
        "tier_edges_train_only": tier_edges.tolist(),
    }

    # Redundancy diagnostics use train only and do not influence the sweep.
    baseline_rate = frames["train"]["decay_rate_2.5"].to_numpy(dtype=np.float64)
    diagnostics["train_correlations_with_decay_rate_2.5"] = {
        f"decay_rate_{fmt_h(h)}": float(
            np.corrcoef(baseline_rate, frames["train"][f"decay_rate_{fmt_h(h)}"])[0, 1]
        )
        for h in USER_HALFLIVES
    }

    metadata = {
        "lengths": lengths,
        "base_columns": list(iter44.CAT_COLS + iter44.NUM_COLS),
        "causality": causality_report,
        "diagnostics": diagnostics,
    }
    return frames, y, users, metadata

