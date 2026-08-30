"""Sweep driver for iter23. Loads the (cached) fused dataset ONCE, then loops
over configs x seeds, writing incremental results to results.json after
every run so partial progress survives interruption. Foreground script.

Feature set is FIXED to iter19's exact winning fused combo throughout
(decay_rate_3, decay_act_3, tab, last1, lastk_rate, gap). Only the BPR
user-sampling mode/alpha varies:
  - combo_full_flat: sampling_mode=flat, alpha=1.0 (iter19-style, unchanged) --
    in-harness reference, should reproduce iter19's own 3-seed numbers.
  - combo_full_decay_alphaX: sampling_mode=decay, alpha in {0.25,0.5,0.75,1.0},
    halflife=3d (matching decay_act_3/decay_rate_3).

Harness-fidelity check (run manually before this driver, see RESULT.md) already
confirmed bit-exact reproduction of iter22's own numbers when run with
feature_set restricted to iter16-only (no momentum fields):
  - flat/alpha=1.0 seed=0: valid 0.6194308996200562 / test 0.6176092624664307
  - decay/alpha=0.5 seeds 0-2: bit-exact match to iter22 results.json
This driver only runs the FULL fused feature set (with momentum fields).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, HALFLIVES
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
FEATURE_SET = ('decay_rate_3', 'decay_act_3', 'tab', 'last1', 'lastk_rate', 'gap')
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
              f"test={out['test']['primary']:.5f}  ({dt:.1f}s)", flush=True)


if __name__ == '__main__':
    print("loading fused dataset (cached after first run)...")
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s")

    results = load_results()
    seeds3 = [0, 1, 2]
    seeds5 = [0, 1, 2, 3, 4]

    # ---- Phase 1: control -- fused features + FLAT iter19-style sampling ----
    run_config('combo_full_flat', 'flat', 1.0, seeds3, results)

    # ---- Phase 2: decay-aware sampling alpha sweep on fused feature set ----
    for alpha in (0.25, 0.5, 0.75, 1.0):
        run_config(f'combo_full_decay_alpha{alpha}', 'decay', alpha, seeds3, results)

    print("\nPhase 1+2 complete.")
