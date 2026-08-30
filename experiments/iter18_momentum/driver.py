"""Sweep driver for iter18. Runs each (config, seed) as a plain foreground
call in-process, writing incremental results to results.json after every
run so partial progress survives interruption. Reuses one cached load_ext()
call across the whole sweep (feature computation is shared; only encode_ext
differs per feature_set, which is cheap).
"""
import json, os, sys, time
import numpy as np
from train import run_bpr_ext


def _clean(d):
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

CONFIGS = [
    ('base',                       ('activity', 'tab', 'rate')),
    ('+last1',                     ('activity', 'tab', 'rate', 'last1')),
    ('+lastk_rate',                ('activity', 'tab', 'rate', 'lastk_rate')),
    ('+last1+lastk_rate',          ('activity', 'tab', 'rate', 'last1', 'lastk_rate')),
    ('+last1+lastk_rate+gap',      ('activity', 'tab', 'rate', 'last1', 'lastk_rate', 'gap')),
    ('+gap',                       ('activity', 'tab', 'rate', 'gap')),
    ('+last1+gap',                 ('activity', 'tab', 'rate', 'last1', 'gap')),
    ('+lastk_rate+gap',            ('activity', 'tab', 'rate', 'lastk_rate', 'gap')),
]
SEEDS_PHASE1 = [0, 1, 2]


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(res):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(res, f, indent=2)


def key(name, seed):
    return f"{name}__seed{seed}"


def main():
    results = load_results()
    cache = {}
    for name, feature_set in CONFIGS:
        for seed in SEEDS_PHASE1:
            k = key(name, seed)
            if k in results:
                print(f"skip {k} (already done): valid primary = {results[k]['valid']['primary']:.5f}")
                continue
            print(f"\n=== running {name} seed={seed} features={feature_set} ===")
            t0 = time.time()
            res = run_bpr_ext(DATA_DIR, feature_set=feature_set, seed=seed, verbose=True,
                               _cache=cache)
            dt = time.time() - t0
            results[k] = {'name': name, 'features': list(feature_set), 'seed': seed,
                          'valid': _clean(res['valid']), 'test': _clean(res['test']),
                          'time_s': dt}
            save_results(results)
            print(f"=== done {k} in {dt:.1f}s | valid primary={res['valid']['primary']:.5f} "
                  f"test primary={res['test']['primary']:.5f} ===")
    print("\nPhase 1 (3-seed sweep) complete.")


if __name__ == '__main__':
    main()
