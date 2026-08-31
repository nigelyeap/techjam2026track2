"""Phase A: sequential LightGBM ranking-objective-only validation sweep."""

from __future__ import annotations

import gc
import os

import numpy as np

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "phase_a_results.json")
SUPPORT_PATH = os.path.join(THIS_DIR, "support_results.json")
REFERENCE_CONFIG = {"objective": "lambdarank"}


def assess(frames, y, users, name, config):
    model, scores, metrics, gain = common.fit_lgb(frames, y, users, config, 0)
    row = common.record(name, dict(config), model, metrics, gain, "lgb")
    row["ties"] = common.tie_stats(scores, users["valid"])
    print(
        f"  {name}: primary={metrics['primary']:.8f} "
        f"GAUC={metrics['GAUC']:.8f} nDCG@5={metrics['nDCG@5']:.8f} "
        f"best_iter={model.best_iteration_}",
        flush=True,
    )
    return row, model, scores


def sequential_axis(frames, y, users, axis, carried, candidates):
    rows = []
    for label, updates in candidates:
        config = dict(carried["rank_config"])
        config.update(updates)
        row, model, scores = assess(frames, y, users, label, config)
        rows.append(row)
        del model, scores
        gc.collect()
    winner = max(rows, key=lambda row: row["valid"]["primary"])
    gain = float(winner["valid"]["primary"] - carried["valid"]["primary"])
    accepted = gain >= common.PRELIMINARY_DELTA
    print(
        f"  {axis} winner={winner['name']} gain_vs_carried={gain:+.8f} "
        f"accepted={accepted}",
        flush=True,
    )
    return {
        "axis": axis,
        "starting_point": carried,
        "candidates": rows,
        "winner": winner,
        "gain_vs_carried": gain,
        "carried": winner if accepted else carried,
        "accepted": accepted,
    }


def main():
    support = common.read_json(SUPPORT_PATH)
    if not all(
        row["supported"] for row in support["lightgbm"].values()
    ):
        raise RuntimeError("LightGBM support inspection did not pass")

    frames, y, users, feature_metadata = common.load_frames()
    results = {
        "experiment": "iterYIXI7_phase_a_lightgbm_ranking_objective",
        "environment": common.environment(),
        "support_source": os.path.basename(SUPPORT_PATH),
        "features": list(common.LGB_TUNING_COLUMNS),
        "feature_metadata": feature_metadata,
        "fixed_tree_config": common.LGB_TREE_CONFIG,
        "selection_policy": {
            "selector": "official validation primary only",
            "preliminary_delta": common.PRELIMINARY_DELTA,
            "promotion_delta": common.PROMOTION_DELTA,
            "method": "sequential lambdarank axes plus one independent rank_xendcg branch; no Cartesian product",
            "test_access": "none in this runner",
        },
    }

    print("=== post-6f LightGBM harness fidelity ===", flush=True)
    reference, reference_model, reference_scores = assess(
        frames, y, users, "post6f_lambdarank_reference", REFERENCE_CONFIG
    )
    if abs(reference["valid"]["primary"] - common.LGB_POST6F_VALID) > 1e-8:
        raise AssertionError(
            f"post-6f LGB drift: {reference['valid']['primary']} "
            f"vs {common.LGB_POST6F_VALID}"
        )
    results["reference"] = reference
    results["harness_fidelity_passed"] = True
    carried = reference

    print("\n=== truncation level (implicit reference default=30) ===", flush=True)
    truncation = sequential_axis(
        frames,
        y,
        users,
        "lambdarank_truncation_level",
        carried,
        [
            ("truncation_5", {"lambdarank_truncation_level": 5}),
            ("truncation_10", {"lambdarank_truncation_level": 10}),
            ("truncation_20", {"lambdarank_truncation_level": 20}),
            ("truncation_50", {"lambdarank_truncation_level": 50}),
        ],
    )
    carried = truncation["carried"]

    print("\n=== sigmoid from carried lambdarank configuration ===", flush=True)
    sigmoid = sequential_axis(
        frames,
        y,
        users,
        "sigmoid",
        carried,
        [
            ("sigmoid_0.5", {"sigmoid": 0.5}),
            ("sigmoid_1.0", {"sigmoid": 1.0}),
            ("sigmoid_2.0", {"sigmoid": 2.0}),
        ],
    )
    carried = sigmoid["carried"]

    print("\n=== lambdarank normalization from carried configuration ===", flush=True)
    normalization = sequential_axis(
        frames,
        y,
        users,
        "lambdarank_norm",
        carried,
        [
            ("lambdarank_norm_true", {"lambdarank_norm": True}),
            ("lambdarank_norm_false", {"lambdarank_norm": False}),
        ],
    )
    carried = normalization["carried"]

    print("\n=== independent alternative objective branch ===", flush=True)
    xendcg, xendcg_model, xendcg_scores = assess(
        frames, y, users, "rank_xendcg", {"objective": "rank_xendcg"}
    )
    xendcg_gain_vs_carried = float(
        xendcg["valid"]["primary"] - carried["valid"]["primary"]
    )
    xendcg_accepted = xendcg_gain_vs_carried >= common.PRELIMINARY_DELTA
    selected = xendcg if xendcg_accepted else carried
    selected_delta = float(
        selected["valid"]["primary"] - reference["valid"]["primary"]
    )

    results["sequential_sweeps"] = {
        "truncation": truncation,
        "sigmoid": sigmoid,
        "normalization": normalization,
    }
    results["rank_xendcg_branch"] = {
        "candidate": xendcg,
        "gain_vs_carried_lambdarank": xendcg_gain_vs_carried,
        "accepted": xendcg_accepted,
    }
    results["selected_on_validation"] = selected
    results["selected_on_validation"]["delta_vs_reference"] = selected_delta
    results["metric_delta_vs_reference"] = {
        metric: float(selected["valid"][metric] - reference["valid"][metric])
        for metric in ("GAUC", "nDCG@5", "primary")
    }

    confirmation_rows = []
    if selected_delta >= common.PROMOTION_DELTA:
        print("\n=== five-seed paired LightGBM confirmation ===", flush=True)
        confirmation_rows.append(
            {
                "seed": 0,
                "reference_valid": reference["valid"],
                "candidate_valid": selected["valid"],
                "delta": selected_delta,
            }
        )
        for seed in common.SEEDS[1:]:
            ref_model, ref_scores, ref_metrics, _ = common.fit_lgb(
                frames, y, users, REFERENCE_CONFIG, seed
            )
            cand_model, cand_scores, cand_metrics, _ = common.fit_lgb(
                frames, y, users, selected["rank_config"], seed
            )
            delta = float(cand_metrics["primary"] - ref_metrics["primary"])
            confirmation_rows.append(
                {
                    "seed": seed,
                    "reference_valid": common.metric_dict(ref_metrics),
                    "candidate_valid": common.metric_dict(cand_metrics),
                    "delta": delta,
                }
            )
            print(
                f"  seed={seed} ref={ref_metrics['primary']:.8f} "
                f"candidate={cand_metrics['primary']:.8f} delta={delta:+.8f}",
                flush=True,
            )
            del ref_model, ref_scores, cand_model, cand_scores
            gc.collect()
    results["five_seed_confirmation"] = common.confirmation_summary(confirmation_rows)
    results["confirmed_branch"] = bool(
        results["five_seed_confirmation"]["confirmed"]
    )
    results["standalone_verdict"] = (
        "CONFIRMED" if results["confirmed_branch"] else "NOT_CONFIRMED"
    )
    common.write_json(RESULTS_PATH, results)
    print(
        f"selected={selected['name']} valid={selected['valid']['primary']:.8f} "
        f"delta={selected_delta:+.8f} verdict={results['standalone_verdict']}",
        flush=True,
    )
    print(f"wrote {RESULTS_PATH}", flush=True)
    del reference_model, reference_scores, xendcg_model, xendcg_scores
    gc.collect()


if __name__ == "__main__":
    main()
