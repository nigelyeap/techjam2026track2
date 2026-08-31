"""Post-selection validation-only rank-diversity diagnostic for 6f.

This does not select or promote anything.  It measures Pearson correlation of
within-user percentile outputs (equivalent to a pooled rank-score correlation)
before and after the independently selected feature transfers.
"""

from __future__ import annotations

import json
import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPRESENTATION_PATH = os.path.join(THIS_DIR, "representation_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "diversity_results.json")
YIXI5_BLEND_PATH = os.path.join(
    common.REPO_ROOT, "experiments", "iterYIXI5_xgboost_optimization", "blend.py"
)


def main() -> None:
    with open(REPRESENTATION_PATH, encoding="utf-8") as f:
        representation = json.load(f)
    frames, y, users, _ = common.load_frames(common.DATA_DIR)
    yixi5_blend = common.load_module(YIXI5_BLEND_PATH, "yixi5_blend_for_yixi6_diversity")

    print("=== fitting current/selected tree models for diversity diagnosis ===", flush=True)
    current_xgb_model, current_xgb_scores, current_xgb_metrics, _ = common.fit_xgb(
        frames, y, users, common.XGB_A0_COLUMNS, 0
    )
    selected_xgb_model, selected_xgb_scores, selected_xgb_metrics, _ = common.fit_xgb(
        frames, y, users, representation["selected_xgb"]["columns"], 0
    )
    current_lgb_model, current_lgb_scores, current_lgb_metrics, _ = common.fit_lgb(
        frames, y, users, common.LGB_B0_COLUMNS, 0
    )
    selected_lgb_model, selected_lgb_scores, selected_lgb_metrics, _ = common.fit_lgb(
        frames, y, users, representation["selected_lgb"]["columns"], 0
    )

    percentile = yixi5_blend.within_user_percentile
    current_xgb_rank = percentile(current_xgb_scores, users["valid"])
    selected_xgb_rank = percentile(selected_xgb_scores, users["valid"])
    current_lgb_rank = percentile(current_lgb_scores, users["valid"])
    selected_lgb_rank = percentile(selected_lgb_scores, users["valid"])
    current_tree_correlation = float(np.corrcoef(current_lgb_rank, current_xgb_rank)[0, 1])
    selected_tree_correlation = float(
        np.corrcoef(selected_lgb_rank, selected_xgb_rank)[0, 1]
    )

    results = {
        "experiment": "iterYIXI6_post_selection_diversity_diagnostic",
        "selection_effect": "none; verdict was frozen before this diagnostic",
        "split": "validation only",
        "method": "Pearson correlation of within-user average-tie percentile scores",
        "current_tree_rank_correlation": current_tree_correlation,
        "selected_tree_rank_correlation": selected_tree_correlation,
        "correlation_change": selected_tree_correlation - current_tree_correlation,
        "within_model_rank_change": {
            "xgb_current_vs_selected_correlation": float(
                np.corrcoef(current_xgb_rank, selected_xgb_rank)[0, 1]
            ),
            "lgb_current_vs_selected_correlation": float(
                np.corrcoef(current_lgb_rank, selected_lgb_rank)[0, 1]
            ),
            "xgb_mean_absolute_percentile_change": float(
                np.mean(np.abs(current_xgb_rank - selected_xgb_rank))
            ),
            "lgb_mean_absolute_percentile_change": float(
                np.mean(np.abs(current_lgb_rank - selected_lgb_rank))
            ),
        },
        "standalone_valid": {
            "current_xgb": common.metric_dict(current_xgb_metrics),
            "selected_xgb": common.metric_dict(selected_xgb_metrics),
            "current_lgb": common.metric_dict(current_lgb_metrics),
            "selected_lgb": common.metric_dict(selected_lgb_metrics),
        },
        "test_accessed": False,
    }
    common.write_json(RESULTS_PATH, results)
    print(
        f"tree rank correlation: current={current_tree_correlation:.8f} "
        f"selected={selected_tree_correlation:.8f} "
        f"change={results['correlation_change']:+.8f}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)

    del current_xgb_model, selected_xgb_model, current_lgb_model, selected_lgb_model


if __name__ == "__main__":
    main()
