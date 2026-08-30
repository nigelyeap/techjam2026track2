"""Sweep driver for iter16. Loads the (cached) extended dataset ONCE, then
loops over configs x seeds, writing incremental results to results.json after
every run so partial progress survives interruption. Foreground script, no
backgrounding needed — each run is ~15-60s.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, HALFLIVES
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')


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


def run_config(tag, feature_set, seeds, results, epochs=40):
    for seed in seeds:
        if already_done(results, tag, seed):
            print(f"[skip] {tag} seed={seed} already done")
            continue
        t0 = time.time()
        out = run_bpr_ext(DATA_DIR, feature_set=feature_set, halflives=HALFLIVES,
                           seed=seed, verbose=False, epochs=epochs, splits_cache=SPLITS_CACHE)
        dt = time.time() - t0
        rec = {'tag': tag, 'features': list(feature_set), 'seed': seed,
               'valid_primary': float(out['valid']['primary']), 'valid_gauc': float(out['valid']['GAUC']),
               'valid_ndcg5': float(out['valid']['nDCG@5']),
               'test_primary': float(out['test']['primary']), 'test_gauc': float(out['test']['GAUC']),
               'test_ndcg5': float(out['test']['nDCG@5']), 'seconds': dt}
        results.append(rec)
        save_results(results)
        print(f"[done] {tag:35s} seed={seed} valid={out['valid']['primary']:.5f} "
              f"test={out['test']['primary']:.5f}  ({dt:.1f}s)")


if __name__ == '__main__':
    print("loading extended dataset (cached after first run)...")
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s")

    results = load_results()
    seeds3 = [0, 1, 2]

    # ---- Phase 0: sanity baseline — flat rate alone (should ~match iter11's rate-alone) ----
    run_config('flat_rate', ('rate',), seeds3, results)

    # ---- Phase 1: halflife sweep, decay_rate alone ----
    for h in HALFLIVES:
        run_config(f'decay_rate_{h}', (f'decay_rate_{h}',), seeds3, results)

    print("\nPhase 0+1 complete. Proceeding to Phase 2 combos (best halflife=3, runner-up=7).")

    # ---- Phase 2: feature combos around the winning halflife (3d) + runner-up (7d) ----
    for h in (3, 7):
        run_config(f'decay_rate_{h}+tab', (f'decay_rate_{h}', 'tab'), seeds3, results)
        run_config(f'decay_rate_{h}+decay_act_{h}+tab',
                   (f'decay_rate_{h}', f'decay_act_{h}', 'tab'), seeds3, results)
        run_config(f'decay_rate_{h}+flat_rate', (f'decay_rate_{h}', 'rate'), seeds3, results)

    print("\nPhase 2 complete. Proceeding to Phase 3: 5-seed confirmation of top 2 combos.")

    # ---- Phase 3: 5-seed confirmation of the winner + close runner-up ----
    seeds5 = [0, 1, 2, 3, 4]
    run_config('decay_rate_3+decay_act_3+tab', ('decay_rate_3', 'decay_act_3', 'tab'), seeds5, results)
    run_config('decay_rate_3+tab', ('decay_rate_3', 'tab'), seeds5, results)

    print("\nPhase 3 complete.")
