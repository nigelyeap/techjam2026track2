"""Sweep driver for iter22. Loads the (cached) extended dataset ONCE, then
loops over sampling_mode x alpha x seed, writing incremental results to
results.json after every run so partial progress survives interruption.
Foreground script (each run ~40-50s).

Feature set is FIXED throughout to iter16's exact winning combo
(decay_rate_3, decay_act_3, tab) -- this iteration only changes the BPR
user-sampling WEIGHT (flat pos_len**alpha vs decayed_pos_total**alpha), never
the model's input features.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, HALFLIVES
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
FEATURE_SET = ('decay_rate_3', 'decay_act_3', 'tab')
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


def run_config(tag, sampling_mode, alpha, seeds, results, epochs=40):
    for seed in seeds:
        if already_done(results, tag, seed):
            print(f"[skip] {tag} seed={seed} already done")
            continue
        t0 = time.time()
        out = run_bpr_ext(DATA_DIR, feature_set=FEATURE_SET, halflives=HALFLIVES,
                           seed=seed, verbose=False, epochs=epochs, splits_cache=SPLITS_CACHE,
                           sampling_mode=sampling_mode, alpha=alpha, decay_halflife=3)
        dt = time.time() - t0
        rec = {'tag': tag, 'sampling_mode': sampling_mode, 'alpha': alpha, 'seed': seed,
               'valid_primary': float(out['valid']['primary']), 'valid_gauc': float(out['valid']['GAUC']),
               'valid_ndcg5': float(out['valid']['nDCG@5']),
               'test_primary': float(out['test']['primary']), 'test_gauc': float(out['test']['GAUC']),
               'test_ndcg5': float(out['test']['nDCG@5']), 'seconds': dt}
        results.append(rec)
        save_results(results)
        print(f"[done] {tag:30s} seed={seed} valid={out['valid']['primary']:.5f} "
              f"test={out['test']['primary']:.5f}  ({dt:.1f}s)")


if __name__ == '__main__':
    print("loading extended dataset (cached after first run)...")
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s")

    results = load_results()
    seeds3 = [0, 1, 2]
    seeds5 = [0, 1, 2, 3, 4]

    # ---- Phase 1: decayed-sampling-weight alpha sweep (3 seeds each) ----
    for alpha in (0.5, 1.0, 1.5, 2.0):
        run_config(f'decay_sampling_alpha{alpha}', 'decay', alpha, seeds3, results)

    print("\nPhase 1 complete.")
