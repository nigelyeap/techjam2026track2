"""iter21 Axis B: embedding capacity (k) and bucket-count (n_buckets) resweep
on iter16's exact winning feature set (decay_rate_3 + decay_act_3 + tab),
3 seeds each, holding alpha at BEST_ALPHA (set below, from Axis A's result).

Loads the (cached) extended dataset ONCE, then loops over configs x seeds,
writing incremental results to results_axisB.json after every run so partial
progress survives interruption. Foreground script, no backgrounding needed.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, HALFLIVES
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_axisB.json')
FEATURE_SET = ('decay_rate_3', 'decay_act_3', 'tab')

# Set from Axis A's outcome before running. Axis A (alpha in {0.1,0.25,0.5,1,2,5,10},
# 3 seeds) found alpha=0.5 best on valid (0.62153) with a consistent, same-direction
# test improvement too (0.62014 vs alpha=1.0's 0.61727) -- a real, generalizing signal,
# not a valid-only artifact. Using alpha=0.5 here per the dispatch instructions.
BEST_ALPHA = 0.5

K_VALUES = [16, 24, 32]
NBUCKET_VALUES = [5, 10, 20]


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


def run_config(tag, k, n_buckets, seeds, results, alpha=BEST_ALPHA, epochs=40):
    for seed in seeds:
        if already_done(results, tag, seed):
            print(f"[skip] {tag} seed={seed} already done")
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
              f"test={out['test']['primary']:.5f}  ({dt:.1f}s)")


if __name__ == '__main__':
    print(f"loading extended dataset (cached after first run)... BEST_ALPHA={BEST_ALPHA}")
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s")

    results = load_results()
    seeds3 = [0, 1, 2]

    # ---- k sweep (n_buckets=10 fixed) ----
    for k in K_VALUES:
        run_config(f'k_{k}', k, 10, seeds3, results)

    # ---- n_buckets sweep (k=16 fixed) ----
    for nb in NBUCKET_VALUES:
        run_config(f'nbuckets_{nb}', 16, nb, seeds3, results)

    print("\nAxis B complete.")
