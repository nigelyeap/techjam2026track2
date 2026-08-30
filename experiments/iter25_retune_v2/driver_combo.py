"""iter25 combo: alpha=0.5 (Axis A winner) + n_buckets=20 (Axis B winner),
combined on top of iter19's exact feature set (decay_rate_3, decay_act_3,
tab, last1, lastk_rate, gap), k=16 (Axis B found k=24/32 do NOT help --
capacity confirmed still not the lever; n_buckets is the lever).

Axis A alone (alpha=0.5, n_buckets=10, k=16, 3 seeds): valid 0.63013 / test 0.62696
Axis B alone (alpha=1.0, n_buckets=20, k=16, 3 seeds): valid 0.62996 / test 0.62994
This combo (alpha=0.5, n_buckets=20, k=16):            beats both individually.

Run: `python3 driver_combo.py` (5 seeds, ~10 min). Writes results_combo.json
incrementally, same schema as driver_axisA.py/driver_axisB.py.

(Historical note: the actual iter25 confirmation run that produced this
iteration's published results_combo.json was executed via direct
`train.py --alpha 0.5 --n_buckets 20 --seed {0..4}` CLI calls rather than
through this driver script -- this driver is provided for a clean,
one-command reproduction / rerun path with the same incremental-save
safety as the axis drivers.)
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, HALFLIVES
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_combo.json')
FEATURE_SET = ('decay_rate_3', 'decay_act_3', 'tab', 'last1', 'lastk_rate', 'gap')
ALPHA = 0.5
N_BUCKETS = 20
K = 16


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


if __name__ == '__main__':
    print("loading extended dataset (cached after first run)...", flush=True)
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    results = load_results()
    tag = 'combo_alpha0.5_nb20'
    for seed in [0, 1, 2, 3, 4]:
        if already_done(results, tag, seed):
            print(f"[skip] seed={seed} already done", flush=True)
            continue
        t0 = time.time()
        out = run_bpr_ext(DATA_DIR, feature_set=FEATURE_SET, halflives=HALFLIVES,
                           seed=seed, verbose=False, epochs=40, splits_cache=SPLITS_CACHE,
                           alpha=ALPHA, n_buckets=N_BUCKETS, k=K)
        dt = time.time() - t0
        rec = {'tag': tag, 'alpha': ALPHA, 'k': K, 'n_buckets': N_BUCKETS, 'seed': seed,
               'valid_primary': float(out['valid']['primary']), 'valid_gauc': float(out['valid']['GAUC']),
               'valid_ndcg5': float(out['valid']['nDCG@5']),
               'test_primary': float(out['test']['primary']), 'test_gauc': float(out['test']['GAUC']),
               'test_ndcg5': float(out['test']['nDCG@5']), 'seconds': dt}
        results.append(rec)
        save_results(results)
        print(f"[done] seed={seed} valid={out['valid']['primary']:.5f} "
              f"test={out['test']['primary']:.5f}  ({dt:.1f}s)", flush=True)

    print("\nCombo 5-seed confirmation complete.", flush=True)
