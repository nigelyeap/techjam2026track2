"""Driver for iter30's variance-reduction study on iter26's `deep_h32`.

Loads the (cached) iter19 fused dataset ONCE, then loops over configs x
seeds, writing incremental results to results.json after every run so
partial progress survives interruption. Meant to be run in the background
with output redirected to a log file.

Phases:
  0. Harness-fidelity reference: reproduce iter26's `deep_h32` at 3 seeds
     under THIS dir's harness (DeepFMVR with init_scale_mult=1.0,
     mlp_seed=None -- byte-identical config to iter26's DeepFM).
  1. Lever 1 -- lower mlp_lr in {0.0005, 0.0002}, 3 seeds each.
  2. Lever 2 -- higher l2_mlp in {1e-4, 1e-3}, 3 seeds each.
  3. Lever 3 -- smaller init_scale_mult in {0.5, 0.25}, 3 seeds each.
  4. 5-seed extension of whichever Phase 1-3 configs look promising
     (decided after Phase 1-3 results are in -- see EXTEND_5SEED below,
     edited by hand once the 3-seed numbers are known).
  5. Lever 4 -- ensembling n_members=3 independent deep-part inits
     (same data/FM seed, different mlp_seed per member), 3 outer seeds,
     default hyperparams (mlp_lr=None, l2_mlp=1e-5, init_scale_mult=1.0)
     matching iter26's own deep_h32 config exactly except for the
     ensembling itself.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import run_deepfm_bpr, run_deepfm_ensemble, load_ext, HALFLIVES, DEFAULT_FEATURES

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


def run_config(tag, results, seeds, ensemble=0, **kwargs):
    for seed in seeds:
        if already_done(results, tag, seed):
            print(f"[skip] {tag} seed={seed} already done", flush=True)
            continue
        t0 = time.time()
        if ensemble > 0:
            out = run_deepfm_ensemble(DATA_DIR, seed=seed, verbose=False, splits_cache=SPLITS_CACHE,
                                       n_members=ensemble, **kwargs)
        else:
            out = run_deepfm_bpr(DATA_DIR, seed=seed, verbose=False, splits_cache=SPLITS_CACHE, **kwargs)
        dt = time.time() - t0
        rec = {'tag': tag, 'seed': seed, 'ensemble': ensemble,
               'mlp_lr': kwargs.get('mlp_lr'), 'l2_mlp': kwargs.get('l2_mlp'),
               'init_scale_mult': kwargs.get('init_scale_mult'),
               'diverged': bool(out.get('diverged', False)),
               'n_members_used': out.get('n_members_used'),
               'valid_primary': float(out['valid']['primary']), 'valid_gauc': float(out['valid']['GAUC']),
               'valid_ndcg5': float(out['valid']['nDCG@5']),
               'test_primary': float(out['test']['primary']), 'test_gauc': float(out['test']['GAUC']),
               'test_ndcg5': float(out['test']['nDCG@5']), 'seconds': dt}
        results.append(rec)
        save_results(results)
        print(f"[done] {tag:28s} seed={seed} valid={out['valid']['primary']:.5f} "
              f"test={out['test']['primary']:.5f} diverged={out.get('diverged')}  ({dt:.1f}s)", flush=True)


if __name__ == '__main__':
    print("loading fused dataset (cached after first run)...", flush=True)
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    results = load_results()
    seeds3 = [0, 1, 2]
    seeds45 = [3, 4]

    # ---- Phase 0: reference -- reproduce iter26's deep_h32 (3 seeds) under this harness ----
    print("\n=== Phase 0: reference (deep_h32, default hyperparams, this harness) ===", flush=True)
    run_config('ref_deep_h32', results, seeds3, hidden=(32,), use_deep=True,
                mlp_lr=None, l2_mlp=1e-5, init_scale_mult=1.0)

    # ---- Phase 1: lever 1 -- lower mlp_lr ----
    print("\n=== Phase 1: lever 1 (lower mlp_lr) ===", flush=True)
    for mlp_lr in (0.0005, 0.0002):
        run_config(f'mlp_lr_{mlp_lr}', results, seeds3, hidden=(32,), use_deep=True,
                    mlp_lr=mlp_lr, l2_mlp=1e-5, init_scale_mult=1.0)

    # ---- Phase 2: lever 2 -- higher l2_mlp ----
    print("\n=== Phase 2: lever 2 (higher l2_mlp) ===", flush=True)
    for l2_mlp in (1e-4, 1e-3):
        run_config(f'l2_mlp_{l2_mlp}', results, seeds3, hidden=(32,), use_deep=True,
                    mlp_lr=None, l2_mlp=l2_mlp, init_scale_mult=1.0)

    # ---- Phase 3: lever 3 -- smaller init scale ----
    print("\n=== Phase 3: lever 3 (smaller init_scale_mult) ===", flush=True)
    for ism in (0.5, 0.25):
        run_config(f'init_scale_{ism}', results, seeds3, hidden=(32,), use_deep=True,
                    mlp_lr=None, l2_mlp=1e-5, init_scale_mult=ism)

    print("\nPhase 0-3 complete.", flush=True)

    # ---- Phase 5: lever 4 -- ensembling (3 outer seeds, n_members=3, default hyperparams) ----
    print("\n=== Phase 5: lever 4 (ensemble of 3 deep-part inits, default hyperparams) ===", flush=True)
    run_config('ensemble3_default', results, seeds3, ensemble=3, hidden=(32,),
                mlp_lr=None, l2_mlp=1e-5, init_scale_mult=1.0)

    print("\nPhase 5 complete. All Phase 0-5 (3-seed) runs done.", flush=True)
