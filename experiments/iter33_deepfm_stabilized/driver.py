"""Driver for iter33: does iter30's `init_scale_mult=0.5` variance-reduction
lever, applied to iter28's exact setup (DeepFM deep_h32 on iter24's refined
feature set), nudge iter28's noisy near-zero valid delta into a real positive
one?

Loads the (cached, symlinked from iter28_deepfm_refined_features/) iter24-
refined-feature dataset ONCE, then loops over configs x seeds, writing
incremental results to results.json after EVERY single run (append+save,
not buffered) so partial progress survives a platform session-limit kill --
this exact pair of source experiments (iter28, iter30) were both killed
mid-run by that failure mode last round.

Phases:
  0. Harness-fidelity check: reproduce iter28's `deep_h32` (hidden=(32,),
     init_scale_mult=1.0 -- exact no-op) at seeds 0,1,2. Must match iter28's
     own published per-seed numbers (from iter28/results.json,
     tag='deep_h32', seeds 0-2) closely (bit-exact expected, since
     init_scale_mult=1.0 multiplies the init scale by exactly 1.0).
  1. Stabilized config: same everything, init_scale_mult=0.5 (iter30's
     recommended lever), seeds 0,1,2.
  2. Decision: compare Phase 0 vs Phase 1 3-seed valid means.
       - If Phase 1's valid mean is >= Phase 0's valid mean (real,
         consistent-direction improvement, even a modest one -- the goal
         here is variance reduction possibly unlocking a small mean gain,
         not a huge jump), extend BOTH configs to 5 seeds (seeds 3,4) for a
         fair matched comparison, then compare the 5-seed stabilized mean
         against iter24's own 5-seed valid reference (0.63251).
       - If Phase 1 is worse (or flat-negative) on the 3-seed valid mean,
         stop here -- report the honest "does not help" result without
         spending the extra 2 seeds x 2 configs of compute.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import run_deepfm_bpr, load_ext, HALFLIVES, TAB_HALFLIVES, DEFAULT_FEATURES

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

ITER24_BEST_VALID_5SEED = 0.63251  # iter24 5-seed valid reference (plain FM, no deep part)
ITER28_DEEP_H32_5SEED_VALID = 0.63244  # iter28's own 5-seed deep_h32 (init_scale_mult=1.0 implicitly)
ITER28_DEEP_H32_5SEED_TEST = 0.62996
# iter28's own per-seed deep_h32 numbers (seeds 0-2), for the bit-exactness check:
ITER28_DEEP_H32_SEEDS_VALID = {0: 0.6319554448127747, 1: 0.6348360776901245, 2: 0.6312517523765564}
ITER28_DEEP_H32_SEEDS_TEST = {0: 0.6279833316802979, 1: 0.6313208341598511, 2: 0.6297892332077026}


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
               'init_scale_mult': kwargs.get('init_scale_mult', 1.0),
               'use_deep': kwargs.get('use_deep', True),
               'diverged': bool(out.get('diverged', False)),
               'valid_primary': float(out['valid']['primary']), 'valid_gauc': float(out['valid']['GAUC']),
               'valid_ndcg5': float(out['valid']['nDCG@5']),
               'test_primary': float(out['test']['primary']), 'test_gauc': float(out['test']['GAUC']),
               'test_ndcg5': float(out['test']['nDCG@5']), 'seconds': dt}
        # write immediately after EVERY run, not buffered -- survives a mid-sweep kill
        results.append(rec)
        save_results(results)
        print(f"[done] {tag:24s} seed={seed} valid={out['valid']['primary']:.5f} "
              f"test={out['test']['primary']:.5f} diverged={out.get('diverged')}  ({dt:.1f}s)", flush=True)


def mean_std(results, tag, key):
    import statistics
    vals = [r[key] for r in results if r['tag'] == tag]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


if __name__ == '__main__':
    print(f"loading iter24-refined-feature dataset (cached, symlinked from iter28)... "
          f"features={DEFAULT_FEATURES}", flush=True)
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    results = load_results()
    seeds3 = [0, 1, 2]
    seeds45 = [3, 4]

    # ---- Phase 0: harness-fidelity check (deep_h32, init_scale_mult=1.0, must match iter28) ----
    print("\n=== Phase 0: harness-fidelity check (deep_h32, init_scale_mult=1.0) ===", flush=True)
    run_config('ref_deep_h32', results, seeds3, feature_set=DEFAULT_FEATURES,
                use_deep=True, hidden=(32,), init_scale_mult=1.0)
    m0, s0 = mean_std(results, 'ref_deep_h32', 'valid_primary')
    mt0, st0 = mean_std(results, 'ref_deep_h32', 'test_primary')
    print(f"ref_deep_h32 3-seed: valid mean={m0:.5f} std={s0:.5f} | test mean={mt0:.5f} std={st0:.5f}", flush=True)
    for r in results:
        if r['tag'] == 'ref_deep_h32' and r['seed'] in ITER28_DEEP_H32_SEEDS_VALID:
            dv = r['valid_primary'] - ITER28_DEEP_H32_SEEDS_VALID[r['seed']]
            dt_ = r['test_primary'] - ITER28_DEEP_H32_SEEDS_TEST[r['seed']]
            print(f"  seed={r['seed']} vs iter28 published: Δvalid={dv:+.7f} Δtest={dt_:+.7f}", flush=True)

    # ---- Phase 1: stabilized config (init_scale_mult=0.5), 3 seeds ----
    print("\n=== Phase 1: init_scale_mult=0.5 (iter30's recommended lever), 3 seeds ===", flush=True)
    run_config('stab_0.5', results, seeds3, feature_set=DEFAULT_FEATURES,
                use_deep=True, hidden=(32,), init_scale_mult=0.5)
    m1, s1 = mean_std(results, 'stab_0.5', 'valid_primary')
    mt1, st1 = mean_std(results, 'stab_0.5', 'test_primary')
    print(f"stab_0.5 3-seed: valid mean={m1:.5f} std={s1:.5f} | test mean={mt1:.5f} std={st1:.5f}", flush=True)

    delta3 = m1 - m0
    print(f"\n3-seed valid delta (stab_0.5 - ref_deep_h32): {delta3:+.5f}", flush=True)

    # ---- Decision: extend to 5 seeds only if the 3-seed direction is non-negative ----
    if delta3 >= 0:
        print("\nPositive/flat direction at 3 seeds -- extending BOTH configs to 5 seeds (2,3,4->3,4).", flush=True)
        run_config('ref_deep_h32', results, seeds45, feature_set=DEFAULT_FEATURES,
                    use_deep=True, hidden=(32,), init_scale_mult=1.0)
        run_config('stab_0.5', results, seeds45, feature_set=DEFAULT_FEATURES,
                    use_deep=True, hidden=(32,), init_scale_mult=0.5)
        m0_5, s0_5 = mean_std(results, 'ref_deep_h32', 'valid_primary')
        mt0_5, st0_5 = mean_std(results, 'ref_deep_h32', 'test_primary')
        m1_5, s1_5 = mean_std(results, 'stab_0.5', 'valid_primary')
        mt1_5, st1_5 = mean_std(results, 'stab_0.5', 'test_primary')
        print(f"\nref_deep_h32 5-seed: valid mean={m0_5:.5f} std={s0_5:.5f} | test mean={mt0_5:.5f} std={st0_5:.5f}", flush=True)
        print(f"stab_0.5     5-seed: valid mean={m1_5:.5f} std={s1_5:.5f} | test mean={mt1_5:.5f} std={st1_5:.5f}", flush=True)
        print(f"\n5-seed valid delta (stab_0.5 - ref_deep_h32): {m1_5-m0_5:+.5f}", flush=True)
        print(f"5-seed valid delta (stab_0.5 - iter24 reference {ITER24_BEST_VALID_5SEED}): "
              f"{m1_5-ITER24_BEST_VALID_5SEED:+.5f}", flush=True)
    else:
        print("\nNegative direction at 3 seeds -- NOT extending to 5 seeds. "
              "init_scale_mult=0.5 does not show even a modest improvement here.", flush=True)

    print("\nDriver complete.", flush=True)
