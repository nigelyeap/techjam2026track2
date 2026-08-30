"""Sweep driver for iter19. Loads the (cached) fused dataset ONCE, then loops
over configs x seeds, writing incremental results to results.json after every
run so partial progress survives interruption. Foreground script.
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
        print(f"[done] {tag:40s} seed={seed} valid={out['valid']['primary']:.5f} "
              f"test={out['test']['primary']:.5f}  ({dt:.1f}s)", flush=True)


if __name__ == '__main__':
    print("loading fused dataset (cached after first run)...")
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s")

    results = load_results()
    seeds3 = [0, 1, 2]

    FEATS_IT16 = ('decay_rate_3', 'decay_act_3', 'tab')
    FEATS_IT18 = ('activity', 'tab', 'rate', 'last1', 'lastk_rate', 'gap')
    FEATS_COMBO = ('decay_rate_3', 'decay_act_3', 'tab', 'last1', 'lastk_rate', 'gap')

    # ---- Phase 1: parity/sanity checks ----
    run_config('iter16_alone', FEATS_IT16, seeds3, results)
    run_config('iter18_alone', FEATS_IT18, seeds3, results)

    # ---- Phase 2: full fusion ----
    run_config('combo_full', FEATS_COMBO, seeds3, results)

    print("\nPhase 1+2 complete.")

    # ---- Phase 3: ablate which momentum feature is additive on top of decay ----
    run_config('combo_minus_last1', ('decay_rate_3', 'decay_act_3', 'tab', 'lastk_rate', 'gap'),
               seeds3, results)
    run_config('combo_minus_lastk_rate', ('decay_rate_3', 'decay_act_3', 'tab', 'last1', 'gap'),
               seeds3, results)
    run_config('combo_minus_gap', ('decay_rate_3', 'decay_act_3', 'tab', 'last1', 'lastk_rate'),
               seeds3, results)

    print("\nPhase 3 complete.")

    # ---- Phase 4: 5-seed confirmation of the full combo (huge valid margin over iter16) ----
    seeds5 = [0, 1, 2, 3, 4]
    run_config('combo_full', FEATS_COMBO, seeds5, results)

    print("\nPhase 4 complete.")
