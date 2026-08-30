"""iter21 Axis A extension: alpha=0.5 was best on the original {0.5,1,2,5,10}
grid, monotonically, at the grid boundary. Extend downward to {0.1, 0.25} to
check whether the trend continues (true optimum lower) or reverses/plateaus
(0.5 near the true optimum) before committing to a 5-seed confirmation.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, HALFLIVES
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_axisA.json')
FEATURE_SET = ('decay_rate_3', 'decay_act_3', 'tab')
ALPHAS = [0.1, 0.25]


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


def run_config(tag, alpha, seeds, results, k=16, n_buckets=10, epochs=40):
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
    print("loading extended dataset (cached after first run)...")
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s")

    results = load_results()
    seeds3 = [0, 1, 2]

    for alpha in ALPHAS:
        run_config(f'alpha_{alpha}', alpha, seeds3, results)

    print("\nAxis A extension complete.")
