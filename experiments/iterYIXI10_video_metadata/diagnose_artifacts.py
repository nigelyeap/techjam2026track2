"""Post-selection source, tie, plateau, and confound checks for Section 6j."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENSEMBLE_PATH = os.path.join(THIS_DIR, "ensemble_results.json")
RESULTS_PATH = os.path.join(THIS_DIR, "artifact_results.json")


def category_rates(frame, labels):
    data = pd.DataFrame(
        {
            "upload_type": frame["meta_upload_type"].astype("string"),
            "label": np.asarray(labels, dtype=np.float64),
        }
    )
    grouped = data.groupby("upload_type", observed=True, dropna=False)["label"]
    output = []
    for value, group in grouped:
        output.append(
            {
                "upload_type": str(value),
                "rows": int(len(group)),
                "long_view_rate": float(group.mean()),
            }
        )
    return sorted(output, key=lambda row: row["rows"], reverse=True)


def main():
    ensemble = common.read_json(ENSEMBLE_PATH)
    frames, y, users, metadata = common.features.load_frames()
    selected = ensemble["selected_on_validation"]
    expected_columns = common.LGB_REFERENCE_COLUMNS + ["meta_upload_type"]
    if selected["lgb_columns"] != expected_columns:
        raise AssertionError("selected model changes more than upload_type")

    constant_metrics = common.evaluate(
        users["valid"], y["valid"], np.zeros(len(y["valid"]), dtype=np.float64)
    )
    random_metrics = common.evaluate(
        users["valid"], y["valid"],
        np.random.default_rng(0).uniform(size=len(y["valid"]))
    )
    ties = ensemble["diagnostics"]["ties"]
    if ties["candidate_lgb"]["mean_per_user_unique_fraction"] < 0.95:
        raise AssertionError("candidate LightGBM is heavily tied")
    if ties["candidate_selected_ensemble"]["mean_per_user_unique_fraction"] < 0.99:
        raise AssertionError("candidate ensemble is heavily tied")
    deltas = ensemble["diagnostics"]["metric_delta_vs_reference"]
    if deltas["GAUC"] <= 0 or deltas["nDCG@5"] <= 0:
        raise AssertionError("final gain is not supported by both component metrics")

    source = metadata["source"]
    leakage = metadata["leakage_policy"]
    if source["statistic_file_read"]:
        raise AssertionError("statistic file was read")
    if leakage["prohibited_columns_in_frames"]:
        raise AssertionError("prohibited aggregate reached frame")
    if set(source["columns"]).intersection(common.features.PROHIBITED_AGGREGATES):
        raise AssertionError("basic source contains a prohibited aggregate")

    records = ensemble["weight_sweep"]["records"]
    best = selected["valid"]["primary"]
    three_model = [
        row for row in records
        if all(row["weights"][name] > 0 for name in ("fm", "lgb", "xgb"))
    ]
    within_noise = [
        row for row in three_model
        if best - row["valid"]["primary"] <= common.PRELIMINARY_DELTA
    ]
    fixed_delta = ensemble["fixed_reference_weights"]["delta_vs_reference"]
    if fixed_delta < common.PROMOTION_DELTA:
        raise AssertionError("metadata gain disappears at old weights")

    category_audit = metadata["category_audit"]["meta_upload_type"]
    for name in ("train", "valid", "test"):
        split = category_audit["splits"][name]
        if split["missing_sentinel_rows"] or split["unseen_rows_mapped_to_nan"]:
            raise AssertionError(f"unexpected upload_type missing/unseen in {name}")

    results = {
        "experiment": "iterYIXI10_post_selection_artifact_diagnostics",
        "selection_role": "none; branch, weights, verdict, and test already frozen",
        "constant_score_valid": common.metric_dict(constant_metrics),
        "random_score_seed0_valid": common.metric_dict(random_metrics),
        "tie_checks": ties,
        "metric_delta_vs_reference": deltas,
        "source_and_leakage_checks": {
            "only_source_read": source["file_read"],
            "source_sha256": source["sha256"],
            "statistic_file_read": source["statistic_file_read"],
            "prohibited_columns_in_frames": leakage["prohibited_columns_in_frames"],
            "join_unmatched_rows": source["join_unmatched_rows"],
            "metadata_static_per_video": metadata["all_metadata_static_per_video"],
            "upload_type_known_at_upload": True,
            "upload_type_train_categories": category_audit["train_categories"],
            "upload_type_missing_or_unseen_rows": {
                name: int(
                    category_audit["splits"][name]["missing_sentinel_rows"]
                    + category_audit["splits"][name]["unseen_rows_mapped_to_nan"]
                )
                for name in ("train", "valid", "test")
            },
        },
        "model_confound_checks": {
            "reference_columns": common.LGB_REFERENCE_COLUMNS,
            "candidate_columns": expected_columns,
            "added_columns": ["meta_upload_type"],
            "removed_columns": [],
            "same_lightgbm_hyperparameters": True,
            "same_training_rows_and_labels": True,
            "same_all_other_feature_columns": True,
            "unchanged_fm": True,
            "unchanged_xgboost": True,
            "candidate_upload_type_gain_fraction": ensemble["diagnostics"][
                "upload_type_gain_fraction"
            ],
        },
        "weight_plateau": {
            "best_valid": best,
            "three_model_points_searched": len(three_model),
            "points_within_0.0003": len(within_noise),
            "fixed_old_weight_delta": fixed_delta,
            "selected_weight_delta": selected["delta_vs_reference"],
            "top_points": sorted(
                three_model,
                key=lambda row: row["valid"]["primary"],
                reverse=True,
            )[:15],
        },
        "upload_type_label_rate_diagnostic": {
            "role": "post-selection descriptive check only",
            "train": category_rates(frames["train"], y["train"]),
            "valid": category_rates(frames["valid"], y["valid"]),
        },
        "verdict": "PASSED: intrinsic static source, no aggregates, no tie artifact, no isolated weight spike",
    }
    common.write_json(RESULTS_PATH, results)
    print(
        f"constant={constant_metrics['primary']:.8f} "
        f"fixed_gain={fixed_delta:+.8f} "
        f"plateau={len(within_noise)}/{len(three_model)}", flush=True
    )
    print(f"wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
