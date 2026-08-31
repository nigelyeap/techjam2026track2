"""Independent metadata tests and validation-only composition for LightGBM."""

from __future__ import annotations

import gc
import os

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "phase_lgb_results.json")


def columns_for(groups):
    return common.LGB_REFERENCE_COLUMNS + [
        column for group in groups for column in common.FEATURE_GROUPS[group]
    ]


def main():
    frames, y, users, metadata = common.features.load_frames()
    print("=== LightGBM reference fidelity ===", flush=True)
    ref_model, ref_scores, ref_metrics, ref_gain = common.fit_lgb(
        frames, y, users, common.LGB_REFERENCE_COLUMNS, 0
    )
    if abs(ref_metrics["primary"] - common.LGB_REFERENCE_VALID) > 1e-8:
        raise AssertionError("LightGBM reference drift")
    reference = common.record(
        "L0_reference", [], common.LGB_REFERENCE_COLUMNS, "lgb",
        ref_model, ref_metrics, ref_gain
    )
    reference["ties"] = common.unique_stats(ref_scores, users["valid"])

    print("=== independent intrinsic metadata groups ===", flush=True)
    candidates = []
    seed0 = {}
    for index, group in enumerate(common.FEATURE_GROUPS, start=1):
        columns = columns_for([group])
        model, scores, metrics, gain = common.fit_lgb(
            frames, y, users, columns, 0
        )
        row = common.record(
            f"L{index}_{group}", [group], columns, "lgb", model, metrics, gain
        )
        row["delta_vs_reference"] = float(
            metrics["primary"] - ref_metrics["primary"]
        )
        row["metric_delta_vs_reference"] = {
            name: float(metrics[name] - ref_metrics[name])
            for name in ("GAUC", "nDCG@5", "primary")
        }
        row["group_gain_fraction"] = float(
            sum(gain[column] for column in common.FEATURE_GROUPS[group])
        )
        row["ties"] = common.unique_stats(scores, users["valid"])
        row["clears_preliminary"] = bool(
            row["delta_vs_reference"] >= common.PRELIMINARY_DELTA
        )
        candidates.append(row)
        seed0[group] = (model, scores, metrics, gain)
        print(
            f"  {group}: valid={metrics['primary']:.8f} "
            f"delta={row['delta_vs_reference']:+.8f}", flush=True
        )

    survivors = [
        row["groups"][0] for row in candidates if row["clears_preliminary"]
    ]
    composition = []
    selected_groups = []
    selected_metrics = ref_metrics
    selected_columns = common.LGB_REFERENCE_COLUMNS
    selected_model = ref_model
    selected_scores = ref_scores
    selected_gain = ref_gain
    if survivors:
        best_independent = max(
            (row for row in candidates if row["groups"][0] in survivors),
            key=lambda row: row["valid"]["primary"],
        )
        selected_groups = list(best_independent["groups"])
        selected_columns = list(best_independent["columns"])
        selected_model, selected_scores, selected_metrics, selected_gain = seed0[selected_groups[0]]
        remaining = [group for group in survivors if group not in selected_groups]
        while remaining:
            proposals = []
            for group in remaining:
                groups = selected_groups + [group]
                columns = columns_for(groups)
                model, scores, metrics, gain = common.fit_lgb(
                    frames, y, users, columns, 0
                )
                row = common.record(
                    f"LC_{'_'.join(groups)}", groups, columns, "lgb",
                    model, metrics, gain
                )
                row["delta_vs_reference"] = float(
                    metrics["primary"] - ref_metrics["primary"]
                )
                row["incremental_delta"] = float(
                    metrics["primary"] - selected_metrics["primary"]
                )
                row["all_groups_independently_preliminary"] = True
                proposals.append((row, model, scores, metrics, gain))
                print(
                    f"  composition {groups}: valid={metrics['primary']:.8f} "
                    f"increment={row['incremental_delta']:+.8f}", flush=True
                )
            best = max(proposals, key=lambda item: item[0]["valid"]["primary"])
            for proposal in proposals:
                composition.append(proposal[0])
            if best[0]["incremental_delta"] < common.PRELIMINARY_DELTA:
                break
            selected_groups = list(best[0]["groups"])
            selected_columns = list(best[0]["columns"])
            selected_model, selected_scores, selected_metrics, selected_gain = best[1:]
            remaining = [group for group in remaining if group not in selected_groups]

    selected_delta = float(selected_metrics["primary"] - ref_metrics["primary"])
    confirmation = {
        "performed": False,
        "confirmed": False,
        "reason": "selected seed-0 delta below +0.001 confirmation threshold",
    }
    if selected_delta >= common.PROMOTION_DELTA:
        print(f"=== confirming selected LightGBM groups {selected_groups} ===", flush=True)
        confirmation = common.confirm_representation(
            "lgb", frames, y, users, common.LGB_REFERENCE_COLUMNS,
            selected_columns, ref_metrics, selected_metrics
        )

    selected_record = common.record(
        "L_selected", selected_groups, selected_columns, "lgb",
        selected_model, selected_metrics, selected_gain,
    )
    selected_record["delta_vs_reference"] = selected_delta
    selected_record["ties"] = common.unique_stats(selected_scores, users["valid"])
    results = {
        "experiment": "iterYIXI10_lightgbm_intrinsic_metadata",
        "environment": common.environment(),
        "feature_metadata": metadata,
        "fixed_config": common.LGB_CONFIG,
        "selection_policy": {
            "selector": "official validation primary only",
            "independent_preliminary_delta": common.PRELIMINARY_DELTA,
            "confirmation_delta": common.PROMOTION_DELTA,
            "composition": "forward add only independently preliminary-positive groups; carry only >=0.0003 incremental gain",
            "music_id": "independent high-cardinality group",
            "test_access": "none",
        },
        "reference": reference,
        "independent_candidates": candidates,
        "preliminary_survivors": survivors,
        "composition_candidates": composition,
        "selected_on_validation": selected_record,
        "confirmation": confirmation,
        "confirmed_branch": bool(confirmation["confirmed"]),
    }
    common.write_json(RESULTS_PATH, results)
    print(
        f"selected={selected_groups} delta={selected_delta:+.8f} "
        f"confirmed={results['confirmed_branch']}", flush=True
    )
    print(f"wrote {RESULTS_PATH}", flush=True)
    del ref_model, ref_scores
    gc.collect()


if __name__ == "__main__":
    main()
