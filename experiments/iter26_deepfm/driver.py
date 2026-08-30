"""Sweep driver for iter26. Loads the (cached) iter19 fused dataset ONCE,
then loops over configs x seeds, writing incremental results to results.json
after every run so partial progress survives interruption. Foreground script,
meant to be run with output redirected to a log file and polled.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import run_deepfm_bpr, load_ext, HALFLIVES, DEFAULT_FEATURES

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


def run_config(tag, results, seeds, **kwargs):
    for seed in seeds:
        if already_done(results, tag, seed):
            print(f"[skip] {tag} seed={seed} already done", flush=True)
            continue
        t0 = time.time()
        out = run_deepfm_bpr(DATA_DIR, seed=seed, verbose=False, splits_cache=SPLITS_CACHE, **kwargs)
        dt = time.time() - t0
        rec = {'tag': tag, 'seed': seed, 'hidden': list(kwargs.get('hidden', ())),
               'use_deep': kwargs.get('use_deep', True), 'mlp_lr': kwargs.get('mlp_lr'),
               'diverged': bool(out.get('diverged', False)),
               'valid_primary': float(out['valid']['primary']), 'valid_gauc': float(out['valid']['GAUC']),
               'valid_ndcg5': float(out['valid']['nDCG@5']),
               'test_primary': float(out['test']['primary']), 'test_gauc': float(out['test']['GAUC']),
               'test_ndcg5': float(out['test']['nDCG@5']), 'seconds': dt}
        results.append(rec)
        save_results(results)
        print(f"[done] {tag:40s} seed={seed} valid={out['valid']['primary']:.5f} "
              f"test={out['test']['primary']:.5f} diverged={out.get('diverged')}  ({dt:.1f}s)", flush=True)


if __name__ == '__main__':
    print("loading fused dataset (cached after first run)...", flush=True)
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    results = load_results()
    seeds3 = [0, 1, 2]

    # ---- Phase 0: harness-fidelity check (use_deep=False must reproduce iter19) ----
    print("\n=== Phase 0: harness-fidelity check (MLP disabled == iter19 FM-only BPR) ===", flush=True)
    run_config('fm_only_parity', results, seeds3, feature_set=DEFAULT_FEATURES,
                use_deep=False, hidden=())

    # ---- Phase 1: width x depth sweep, 3 seeds, default lr for MLP (=FM lr) ----
    print("\n=== Phase 1: width x depth sweep (3 seeds, mlp_lr=lr=0.001) ===", flush=True)
    for width in (16, 32, 64):
        run_config(f'deep_h{width}', results, seeds3, feature_set=DEFAULT_FEATURES,
                    use_deep=True, hidden=(width,))
        run_config(f'deep_h{width}x{width}', results, seeds3, feature_set=DEFAULT_FEATURES,
                    use_deep=True, hidden=(width, width))

    print("\nPhase 0+1 complete.", flush=True)

    # ---- Phase 2: 5-seed confirmation of the top candidates from Phase 1 ----
    # deep_h64x64 had the highest 3-seed valid mean (0.63161); deep_h32 was the
    # best single-hidden-layer config and tied deep_h32x32 for 2nd-highest mean
    # (0.63088) with lower variance. Both clearly beat fm_only_parity's 3-seed
    # mean (0.62933) by >=0.0016. Extend both to seeds 3,4 (already have 0,1,2).
    print("\n=== Phase 2: 5-seed confirmation of top candidates ===", flush=True)
    seeds45 = [3, 4]
    run_config('deep_h64x64', results, seeds45, feature_set=DEFAULT_FEATURES,
                use_deep=True, hidden=(64, 64))
    run_config('deep_h32', results, seeds45, feature_set=DEFAULT_FEATURES,
                use_deep=True, hidden=(32,))
    # Also extend the harness-fidelity parity baseline itself to 5 seeds so the
    # final comparison is apples-to-apples 5-seed vs 5-seed (not 5 vs 3).
    run_config('fm_only_parity', results, seeds45, feature_set=DEFAULT_FEATURES,
                use_deep=False, hidden=())

    print("\nPhase 2 complete.", flush=True)
