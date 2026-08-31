"""Validation component rankings by predeclared causal history regime."""

from __future__ import annotations

import itertools
import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "phase_a_results.json")


def main():
    frozen = common.load_predictions()
    users = frozen["users"]
    labels = frozen["labels"]
    regimes = frozen["regimes"]
    regime_metrics = {}
    all_large_enough = True
    for regime in common.regimes.REGIME_NAMES:
        mask = regimes == regime
        regime_users = np.unique(users[mask])
        all_large_enough &= len(regime_users) >= common.MIN_REGIME_USERS
        model_metrics = {
            model: common.metric_dict(
                common.evaluate(users[mask], labels[mask], frozen[model][mask])
            )
            for model in common.MODELS
        }
        order = sorted(
            common.MODELS,
            key=lambda model: model_metrics[model]["primary"],
            reverse=True,
        )
        regime_metrics[regime] = {
            "rows": int(np.sum(mask)),
            "users": int(len(regime_users)),
            "models": model_metrics,
            "primary_order": order,
        }
        print(
            f"{regime}: " + " > ".join(
                f"{model} {model_metrics[model]['primary']:.8f}" for model in order
            ), flush=True
        )

    reversals = []
    for left_model, right_model in itertools.combinations(common.MODELS, 2):
        differences = {
            regime: float(
                regime_metrics[regime]["models"][left_model]["primary"]
                - regime_metrics[regime]["models"][right_model]["primary"]
            )
            for regime in common.regimes.REGIME_NAMES
        }
        for left_regime, right_regime in itertools.combinations(
            common.regimes.REGIME_NAMES, 2
        ):
            left = differences[left_regime]
            right = differences[right_regime]
            if (
                np.sign(left) != np.sign(right)
                and abs(left) >= common.MEANINGFUL_ORDER_MARGIN
                and abs(right) >= common.MEANINGFUL_ORDER_MARGIN
            ):
                reversals.append(
                    {
                        "model_pair": [left_model, right_model],
                        "regimes": [left_regime, right_regime],
                        "primary_differences_left_minus_right": {
                            left_regime: left,
                            right_regime: right,
                        },
                    }
                )

    eligible = bool(reversals and all_large_enough)
    results = {
        "experiment": "iterYIXI11_phase_a_component_regime_diagnostics",
        "selection_policy": {
            "regimes": "frozen before scores; training-count median and no labels",
            "minimum_regime_users": common.MIN_REGIME_USERS,
            "meaningful_order_reversal": (
                "same model pair reverses sign across two regimes with at least "
                f"{common.MEANINGFUL_ORDER_MARGIN:.3f} primary separation in each"
            ),
            "test_access": "none",
        },
        "regime_metadata": common.read_json(common.PREDICTIONS_METADATA_PATH)[
            "regime_metadata"
        ],
        "regime_component_metrics": regime_metrics,
        "all_regimes_large_enough": bool(all_large_enough),
        "meaningful_order_reversals": reversals,
        "adaptive_weight_sweep_eligible": eligible,
    }
    common.write_json(RESULTS_PATH, results)
    print(f"eligible={eligible} reversals={len(reversals)}", flush=True)
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
