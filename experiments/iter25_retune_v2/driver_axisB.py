"""iter25 Axis B: embedding capacity (k) and bucket-count (n_buckets) resweep
on iter19's exact winning feature set (decay_rate_3 + decay_act_3 + tab +
last1 + lastk_rate + gap), 3 seeds each, at the DEFAULT alpha=1.0 (iter19's
value) -- this axis is independent of Axis A's alpha finding and checks
whether the README's "capacity isn't the bottleneck" finding (established on
the original pointwise FM baseline, k=8/16/32 -> 0.5895/0.5902/0.5887) still
holds now that the model has BPR + a much richer 6-field feature set.

n_buckets controls the quantile bucket count used for ALL bucketed
continuous features (dur_bucket, decay_rate, decay_act, lastk_rate, gap) --
see data_ext.py's encode_ext. This was never run in iter21 (Axis B was
never started there at all).

Loads the (cached) extended dataset ONCE, then loops over configs x seeds,
writing incremental results to results_axisB.json after every run so partial
progress survives interruption. Foreground script, no backgrounding needed
(each run ~90-100s; whole axis ~18 runs ~30 min).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, HALFLIVES, ALPHA
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_axisB.json')
FEATURE_SET = ('decay_rate_3', 'decay_act_3', 'tab', 'last1', 'lastk_rate', 'gap')
DEFAULT_ALPHA = ALPHA  # 1.0, iter19's default -- Axis B is orthogonal to Axis A

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


def run_config(tag, k, n_buckets, seeds, results, alpha=DEFAULT_ALPHA, epochs=40):
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
    print(f"loading extended dataset (cached after first run)... DEFAULT_ALPHA={DEFAULT_ALPHA}", flush=True)
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    results = load_results()
    seeds3 = [0, 1, 2]

    # ---- k sweep (n_buckets=10 fixed); k=16 duplicates the iter19-parity
    # point but is cheap and keeps this axis self-contained/comparable ----
    for k in K_VALUES:
        run_config(f'k_{k}', k, 10, seeds3, results)

    # ---- n_buckets sweep (k=16 fixed); n_buckets=10 duplicates k_16 above,
    # already_done() will skip it automatically ----
    for nb in NBUCKET_VALUES:
        run_config(f'nbuckets_{nb}', 16, nb, seeds3, results)

    print("\nAxis B complete.", flush=True)
