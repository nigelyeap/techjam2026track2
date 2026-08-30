"""iter29: train/valid-date-shifted robustness check for iter25's
n_buckets=20 finding.

Runs n_buckets=10 (control) vs n_buckets=20 (the finding under test) on the
SHIFTED split (this dir's data_ext.py -- train 20220405..18 / valid
20220419..25 / test 20220426..05-05, see data_ext.py's module docstring for
the exact derivation), 5 seeds each, at alpha=1.0 (default) and k=16
(default), on iter19's EXACT feature set (decay_rate_3, decay_act_3, tab,
last1, lastk_rate, gap) -- no combination with alpha retuning or any other
iteration's refinements, so this is a clean, isolated re-test of iter25's
Axis B n_buckets finding only, mirroring iter25's driver_axisB.py structure
and incremental-save pattern.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, HALFLIVES, ALPHA
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
FEATURE_SET = ('decay_rate_3', 'decay_act_3', 'tab', 'last1', 'lastk_rate', 'gap')
DEFAULT_ALPHA = ALPHA  # 1.0, iter19's default -- not retuned here
K_DEFAULT = 16

NBUCKET_VALUES = [10, 20]
SEEDS = [0, 1, 2, 3, 4]


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as fh:
            return json.load(fh)
    return []


def save_results(res):
    with open(RESULTS_PATH, 'w') as fh:
        json.dump(res, fh, indent=2)


def already_done(res, tag, seed):
    return any(r['tag'] == tag and r['seed'] == seed for r in res)


def run_config(tag, n_buckets, seeds, results, alpha=DEFAULT_ALPHA, k=K_DEFAULT, epochs=40):
    for seed in seeds:
        if already_done(results, tag, seed):
            print(f"[skip] {tag} seed={seed} already done", flush=True)
            continue
        t0 = time.time()
        out = run_bpr_ext(DATA_DIR, feature_set=FEATURE_SET, halflives=HALFLIVES,
                           seed=seed, verbose=False, epochs=epochs, splits_cache=SPLITS_CACHE,
                           alpha=alpha, n_buckets=n_buckets, k=k)
        dt = time.time() - t0
        rec = {'tag': tag, 'alpha': alpha, 'k': k, 'n_buckets': n_buckets, 'seed': seed,
               'valid_primary': float(out['valid']['primary']), 'valid_gauc': float(out['valid']['GAUC']),
               'valid_ndcg5': float(out['valid']['nDCG@5']),
               'test_primary': float(out['test']['primary']), 'test_gauc': float(out['test']['GAUC']),
               'test_ndcg5': float(out['test']['nDCG@5']), 'seconds': dt}
        results.append(rec)
        save_results(results)
        print(f"[done] {tag:20s} seed={seed} valid={out['valid']['primary']:.5f} "
              f"test={out['test']['primary']:.5f}  ({dt:.1f}s)", flush=True)


if __name__ == '__main__':
    print("loading extended dataset (shifted split, cached after first run)...", flush=True)
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s  sizes={{k: len(v) for k, v in SPLITS_CACHE.items()}}",
          flush=True)
    print({k: len(v) for k, v in SPLITS_CACHE.items()}, flush=True)

    results = load_results()

    for nb in NBUCKET_VALUES:
        run_config(f'nbuckets_{nb}', nb, SEEDS, results)

    print("\niter29 shifted-split n_buckets sweep complete.", flush=True)
