"""Sweep driver for iter28. Loads the (cached) iter24-refined-feature dataset
ONCE, then loops over configs x seeds, writing incremental results to
results.json after every run so partial progress survives interruption.
Foreground script, meant to be run with output redirected to a log file and
polled (background + pid-bound wait, per process instructions).

Phase 0: harness-fidelity check. With use_deep=False, this run must
bit-exact-reproduce iter24's own published 5-seed table (same means, same
stds, same per-seed values) -- confirms this dir's copy of iter24's
data_ext.py + the FM-part code path in deepfm_bpr_step introduce zero drift
before trusting anything else.

Phase 1: confirmatory width sweep on iter24's feature set, single hidden
layer only, {16,32,64}, 3 seeds each (0,1,2). iter26 already ran an 18-config
width x depth sweep on the ORIGINAL feature set and found deep_h32 the most
robust single-layer choice; this is a narrower confirmatory check (not a full
re-sweep) since the richer iter24 feature set may shift the optimal width.

Phase 2 (conditional): whichever width wins on 3-seed valid mean gets
extended to a full 5-seed run (seeds 0-4) IF it clears iter24's 5-seed valid
reference (0.63251) by roughly the 0.001-0.002 confirmation margin.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import run_deepfm_bpr, load_ext, HALFLIVES, TAB_HALFLIVES, DEFAULT_FEATURES

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

ITER24_BEST_VALID_5SEED = 0.63251  # reference: decay_rate_2.5+decay_act_2.5+decay_tab_3+last1+lastk_rate+gap, 5-seed, plain FM (no deep part)
ITER26_DEEP_H32_VALID_5SEED = 0.63079  # reference: iter26's deep_h32 on iter19's ORIGINAL feature set, 5-seed


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


def mean_std(results, tag, key):
    import statistics
    vals = [r[key] for r in results if r['tag'] == tag]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


if __name__ == '__main__':
    print(f"loading iter24-refined-feature dataset (cached after first run)... features={DEFAULT_FEATURES}", flush=True)
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    results = load_results()
    seeds5 = [0, 1, 2, 3, 4]
    seeds3 = [0, 1, 2]

    # ---- Phase 0: harness-fidelity check (use_deep=False must reproduce iter24 exactly) ----
    print("\n=== Phase 0: harness-fidelity check (MLP disabled == iter24 plain-FM BPR) ===", flush=True)
    run_config('fm_only_parity', results, seeds5, feature_set=DEFAULT_FEATURES,
                use_deep=False, hidden=())
    m, s = mean_std(results, 'fm_only_parity', 'valid_primary')
    mt, st = mean_std(results, 'fm_only_parity', 'test_primary')
    print(f"fm_only_parity 5-seed: valid mean={m:.5f} std={s:.5f} | test mean={mt:.5f} std={st:.5f}", flush=True)
    print(f"iter24 published 5-seed: valid mean=0.63251 std=0.00050 | test mean=0.62843 std=0.00086", flush=True)

    # ---- Phase 1: single-hidden-layer width confirmatory sweep, 3 seeds ----
    print("\n=== Phase 1: width sweep {16,32,64}, single hidden layer, 3 seeds ===", flush=True)
    for width in (16, 32, 64):
        run_config(f'deep_h{width}', results, seeds3, feature_set=DEFAULT_FEATURES,
                    use_deep=True, hidden=(width,))

    widths = (16, 32, 64)
    means3 = {w: mean_std(results, f'deep_h{w}', 'valid_primary')[0] for w in widths}
    print("\nPhase 1 3-seed valid means:", {w: round(v, 5) for w, v in means3.items()}, flush=True)
    best_w = max(means3, key=lambda w: means3[w])
    print(f"Phase 1 best width: {best_w} (3-seed valid mean {means3[best_w]:.5f})", flush=True)

    # ---- Phase 2 (conditional): 5-seed confirmation of the winning width ----
    margin = means3[best_w] - ITER24_BEST_VALID_5SEED
    print(f"\nMargin of deep_h{best_w} (3-seed) vs iter24 5-seed valid reference ({ITER24_BEST_VALID_5SEED}): {margin:+.5f}", flush=True)
    if margin > 0.001:
        print(f"\n=== Phase 2: 5-seed confirmation of deep_h{best_w} (seeds 3,4 added) ===", flush=True)
        run_config(f'deep_h{best_w}', results, [3, 4], feature_set=DEFAULT_FEATURES,
                    use_deep=True, hidden=(best_w,))
        m5, s5 = mean_std(results, f'deep_h{best_w}', 'valid_primary')
        mt5, st5 = mean_std(results, f'deep_h{best_w}', 'test_primary')
        print(f"deep_h{best_w} 5-seed: valid mean={m5:.5f} std={s5:.5f} | test mean={mt5:.5f} std={st5:.5f}", flush=True)
    else:
        print("\nMargin below confirmation threshold (~0.001-0.002) -- but extending to 5 seeds anyway per "
              "task instruction (default deep_h32 must still be reported at 5 seeds).", flush=True)
        # Per the dispatch prompt, deep_h32 is the designated starting config and must be
        # reported with a 5-seed number regardless, for direct comparability with iter24/iter26.
        run_config('deep_h32', results, [3, 4], feature_set=DEFAULT_FEATURES,
                    use_deep=True, hidden=(32,))
        m5, s5 = mean_std(results, 'deep_h32', 'valid_primary')
        mt5, st5 = mean_std(results, 'deep_h32', 'test_primary')
        print(f"deep_h32 5-seed: valid mean={m5:.5f} std={s5:.5f} | test mean={mt5:.5f} std={st5:.5f}", flush=True)

    print("\nDriver complete.", flush=True)
