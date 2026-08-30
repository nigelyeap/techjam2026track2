"""Sweep driver for iter15. Loads/joins data ONCE (load_ext is feature-set
independent), then loops seeds x configs, writing incremental results to
results.json after every run so partial progress survives interruption.
Run as an ordinary foreground command (per operational notes: no background
long-running processes for this run).
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, USER_FIELDS, VIDEO_FIELDS
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

CAUSAL = ('activity', 'tab', 'rate')
CONFIGS = {
    'base_causal':        CAUSAL,
    'causal_plus_user':   CAUSAL + tuple(USER_FIELDS),
    'causal_plus_video':  CAUSAL + tuple(VIDEO_FIELDS),
    'causal_plus_both':   CAUSAL + tuple(USER_FIELDS) + tuple(VIDEO_FIELDS),
}

SEEDS_PHASE1 = [0, 1, 2]


def _clean(d):
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(res):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(res, f, indent=2)


def run_sweep(seeds=SEEDS_PHASE1, configs=CONFIGS, tag_prefix=''):
    print("loading + joining data once (shared across all configs/seeds)...")
    t0 = time.time()
    splits = load_ext(DATA_DIR)
    print(f"  done in {time.time()-t0:.1f}s")

    results = load_results()
    for cfg_name, feature_set in configs.items():
        for seed in seeds:
            key = f"{tag_prefix}{cfg_name}__seed{seed}"
            if key in results:
                print(f"skip {key} (already have result)")
                continue
            print(f"\n=== {key} : features={feature_set} ===")
            t0 = time.time()
            res = run_bpr_ext(DATA_DIR, feature_set=feature_set, seed=seed,
                               verbose=False, _cached_splits=splits)
            dt = time.time() - t0
            entry = {
                'config': cfg_name, 'features': list(feature_set), 'n_fields': 5 + len(feature_set),
                'seed': seed, 'valid': _clean(res['valid']), 'test': _clean(res['test']),
                'time_s': dt,
            }
            results[key] = entry
            save_results(results)
            print(f"  valid primary={entry['valid']['primary']:.5f}  "
                  f"test primary={entry['test']['primary']:.5f}  ({dt:.1f}s)")
    return results


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['sweep', 'confirm'], default='sweep')
    ap.add_argument('--confirm_config', default=None,
                     help="config name to run extra seeds {3,4} for (phase=confirm)")
    a = ap.parse_args()
    if a.phase == 'sweep':
        run_sweep(seeds=SEEDS_PHASE1, configs=CONFIGS)
    else:
        assert a.confirm_config in CONFIGS, f"unknown config {a.confirm_config}"
        run_sweep(seeds=[3, 4], configs={a.confirm_config: CONFIGS[a.confirm_config]})
    print("\nAll done. See results.json")
