"""Phase B: sequential XGBoost ranking-pair-only validation sweep."""

from __future__ import annotations

import gc
import os

import common


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(THIS_DIR, "phase_b_results.json")
SUPPORT_PATH = os.path.join(THIS_DIR, "support_results.json")
REFERENCE_CONFIG = {}


def assess(frames, y, users, name, config):
    model, scores, metrics, gain = common.fit_xgb(frames, y, users, config, 0)
    row = common.record(name, dict(config), model, metrics, gain, "xgb")
    row["effective_objective_config"] = common.xgb_effective_objective(model)
    row["ties"] = common.tie_stats(scores, users["valid"])
    print(
        f"  {name}: primary={metrics['primary']:.8f} "
        f"GAUC={metrics['GAUC']:.8f} nDCG@5={metrics['nDCG@5']:.8f} "
        f"best_iter={model.best_iteration}",
        flush=True,
    )
    return row, model, scores


def gated_axis(frames, y, users, axis, carried, candidates):
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
    if not all(row["supported"] for row in support["xgboost"].values()):
        raise RuntimeError("XGBoost support inspection did not pass")

    frames, y, users, feature_metadata = common.load_frames()
    results = {
        "experiment": "iterYIXI7_phase_b_xgboost_pair_generation",
        "environment": common.environment(),
        "support_source": os.path.basename(SUPPORT_PATH),
        "features": list(common.XGB_TUNING_COLUMNS),
        "feature_metadata": feature_metadata,
        "fixed_tree_config": common.xgb_tree_config(),
        "selection_policy": {
            "selector": "official validation primary only",
            "preliminary_delta": common.PRELIMINARY_DELTA,
            "promotion_delta": common.PROMOTION_DELTA,
            "method": "sequential pair method, pairs per sample, then normalization; no ordinary tree changes",
            "test_access": "none in this runner",
        },
    }

    print("=== post-6f XGBoost harness fidelity ===", flush=True)
    reference, reference_model, reference_scores = assess(
        frames, y, users, "post6f_implicit_pair_reference", REFERENCE_CONFIG
    )
    if abs(reference["valid"]["primary"] - common.XGB_POST6F_VALID) > 1e-8:
        raise AssertionError(
            f"post-6f XGB drift: {reference['valid']['primary']} "
            f"vs {common.XGB_POST6F_VALID}"
        )
    results["reference"] = reference
    results["harness_fidelity_passed"] = True
    carried = reference

    # Installed 3.4.1 defaults to topk with UINT_MAX pairs.  Change only the
    # method here; the following phase then restricts pair counts to 5--20.
    print("\n=== pair method from installed implicit reference ===", flush=True)
    method = gated_axis(
        frames,
        y,
        users,
        "lambdarank_pair_method",
        carried,
        [("pair_method_mean", {"lambdarank_pair_method": "mean"})],
    )
    carried = method["carried"]

    print("\n=== pair count from carried method ===", flush=True)
    pair_count = gated_axis(
        frames,
        y,
        users,
        "lambdarank_num_pair_per_sample",
        carried,
        [
            ("pairs_5", {"lambdarank_num_pair_per_sample": 5}),
            ("pairs_10", {"lambdarank_num_pair_per_sample": 10}),
            ("pairs_20", {"lambdarank_num_pair_per_sample": 20}),
        ],
    )
    carried = pair_count["carried"]

    print("\n=== ranking normalization from carried configuration ===", flush=True)
    normalization = gated_axis(
        frames,
        y,
        users,
        "lambdarank_normalization",
        carried,
        [
            ("normalization_true", {"lambdarank_normalization": True}),
            ("normalization_false", {"lambdarank_normalization": False}),
        ],
    )
    carried = normalization["carried"]

    selected = carried
    selected_delta = float(
        selected["valid"]["primary"] - reference["valid"]["primary"]
    )
    results["sequential_sweeps"] = {
        "pair_method": method,
        "pair_count": pair_count,
        "normalization": normalization,
    }
    results["selected_on_validation"] = selected
    results["selected_on_validation"]["delta_vs_reference"] = selected_delta
    results["metric_delta_vs_reference"] = {
        metric: float(selected["valid"][metric] - reference["valid"][metric])
        for metric in ("GAUC", "nDCG@5", "primary")
    }

    confirmation_rows = []
    if selected_delta >= common.PROMOTION_DELTA:
        print("\n=== five-seed paired XGBoost confirmation ===", flush=True)
        confirmation_rows.append(
            {
                "seed": 0,
                "reference_valid": reference["valid"],
                "candidate_valid": selected["valid"],
                "delta": selected_delta,
            }
        )
        for seed in common.SEEDS[1:]:
            ref_model, ref_scores, ref_metrics, _ = common.fit_xgb(
                frames, y, users, REFERENCE_CONFIG, seed
            )
            cand_model, cand_scores, cand_metrics, _ = common.fit_xgb(
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
    del reference_model, reference_scores
    gc.collect()


if __name__ == "__main__":
    main()
