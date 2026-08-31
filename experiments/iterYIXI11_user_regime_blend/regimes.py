"""Predeclared label-free user-history regimes for Section 6k."""

from __future__ import annotations

import collections
import csv
import os

import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "KuaiRand-Pure", "data")
RAW_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
REGIME_NAMES = ("no_prior", "low_history", "high_history")
OFFICIAL = {
    "threshold_train": (20220408, 20220421),
    "history": (20220408, 20220421),
    "evaluation": (20220422, 20220428),
}
OFFICIAL_TEST = {
    "threshold_train": (20220408, 20220421),
    "history": (20220408, 20220428),
    "evaluation": (20220429, 20220508),
}
SHIFTED = {
    "threshold_train": (20220405, 20220418),
    "history": (20220405, 20220418),
    "evaluation": (20220419, 20220425),
}


def load_rows(data_dir: str = DATA_DIR):
    rows = []
    for filename in RAW_FILES:
        with open(os.path.join(data_dir, filename), newline="") as f:
            for row in csv.DictReader(f):
                rows.append((int(row["date"]), row["user_id"]))
    return rows


def lower_median_positive_count(rows, date_range):
    lo, hi = date_range
    counts = collections.Counter(user for date, user in rows if lo <= date <= hi)
    if not counts:
        raise AssertionError("empty threshold-training history")
    values = np.asarray(list(counts.values()), dtype=np.int64)
    threshold = int(np.floor(np.median(values)))
    return threshold, counts, values


def assign_regime(count: int, threshold: int):
    if count == 0:
        return "no_prior"
    if count <= threshold:
        return "low_history"
    return "high_history"


def build(spec, expected_users=None, data_dir: str = DATA_DIR):
    rows = load_rows(data_dir)
    threshold, _, threshold_values = lower_median_positive_count(
        rows, spec["threshold_train"]
    )
    hlo, hhi = spec["history"]
    history_counts = collections.Counter(
        user for date, user in rows if hlo <= date <= hhi
    )
    elo, ehi = spec["evaluation"]
    evaluation_users = [user for date, user in rows if elo <= date <= ehi]
    if expected_users is not None and not np.array_equal(
        np.asarray(evaluation_users), np.asarray(expected_users)
    ):
        raise AssertionError("raw/model evaluation user alignment failed")
    user_regimes = {
        user: assign_regime(history_counts.get(user, 0), threshold)
        for user in set(evaluation_users)
    }
    row_regimes = np.asarray([user_regimes[user] for user in evaluation_users])
    for user in set(evaluation_users):
        if len(set(row_regimes[np.asarray(evaluation_users) == user])) != 1:
            raise AssertionError("a user received multiple regimes")
    regime_stats = {}
    for regime in REGIME_NAMES:
        mask = row_regimes == regime
        regime_users = set(np.asarray(evaluation_users)[mask])
        counts = np.asarray(
            [history_counts.get(user, 0) for user in regime_users], dtype=np.int64
        )
        regime_stats[regime] = {
            "rows": int(np.sum(mask)),
            "users": int(len(regime_users)),
            "history_count_min": int(np.min(counts)) if len(counts) else None,
            "history_count_median": float(np.median(counts)) if len(counts) else None,
            "history_count_max": int(np.max(counts)) if len(counts) else None,
        }
    return row_regimes, {
        "policy": {
            "label_free": True,
            "unit": "user; one fixed regime for every row in the evaluation window",
            "history_variable": "number of impressions strictly before the evaluation date boundary",
            "threshold_rule": "lower integer median of positive per-user counts in the threshold-training range",
            "no_prior": "history count == 0",
            "low_history": "1 <= history count <= training median",
            "high_history": "history count > training median",
        },
        "ranges": spec,
        "threshold": threshold,
        "threshold_training_users": int(len(threshold_values)),
        "threshold_distribution": {
            str(q): float(np.quantile(threshold_values, q))
            for q in (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0)
        },
        "regimes": regime_stats,
    }

