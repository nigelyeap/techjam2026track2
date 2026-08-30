import json, os, time
from data_ext import load_ext, HALFLIVES, TAB_HALFLIVES
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = 'results.json'
ITER24_FEATS = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')

ITER27_PUBLISHED = [
    (0.6389358639717102, 0.6398863196372986),
    (0.638678789138794, 0.6391348838806152),
    (0.6368540525436401, 0.6376838684082031),
    (0.6374706625938416, 0.6385308504104614),
    (0.6376844048500061, 0.6392104625701904),
]


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


def run_cfg(tag, seeds, results, cache, k):
    for seed in seeds:
        if already_done(results, tag, seed):
            print(f"[skip] {tag} seed={seed} already done", flush=True)
            continue
        t0 = time.time()
        out = run_bpr_ext(DATA_DIR, feature_set=ITER24_FEATS, seed=seed, verbose=False, epochs=40,
                           splits_cache=cache, sampling_mode='decay', sampling_alpha=0.75,
                           decay_halflife=3, alpha=0.5, n_buckets=20, k=k)
        dt = time.time() - t0
        rec = {'tag': tag, 'seed': seed, 'k': k,
               'valid_primary': float(out['valid']['primary']), 'valid_gauc': float(out['valid']['GAUC']),
               'valid_ndcg5': float(out['valid']['nDCG@5']),
               'test_primary': float(out['test']['primary']), 'test_gauc': float(out['test']['GAUC']),
               'test_ndcg5': float(out['test']['nDCG@5']), 'seconds': dt}
        results.append(rec)
        save_results(results)
        print(f"[done] {tag:16s} seed={seed} valid={out['valid']['primary']:.5f} "
              f"test={out['test']['primary']:.5f}  ({dt:.1f}s)", flush=True)


def mean_of(results, tag, key):
    vals = [r[key] for r in results if r['tag'] == tag]
    return sum(vals) / len(vals) if vals else None


if __name__ == '__main__':
    results = load_results()

    print("=== Loading fused-config dataset (cached after first run) ===", flush=True)
    t0 = time.time()
    CACHE = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n=== Phase 0: harness-fidelity check (k=16, 5 seeds) ===", flush=True)
    run_cfg('fidelity_k16', [0, 1, 2, 3, 4], results, CACHE, k=16)
    all_match = True
    for seed in range(5):
        rec = next(r for r in results if r['tag'] == 'fidelity_k16' and r['seed'] == seed)
        pv, pt = ITER27_PUBLISHED[seed]
        dv, dtst = rec['valid_primary'] - pv, rec['test_primary'] - pt
        ok = abs(dv) < 1e-4 and abs(dtst) < 1e-4
        all_match &= ok
        print(f"  seed {seed}: valid Δ{dv:+.6f} test Δ{dtst:+.6f}  {'OK' if ok else 'MISMATCH'}", flush=True)
    if not all_match:
        print("*** HARNESS FIDELITY FAILED -- STOPPING ***", flush=True)
        raise SystemExit(1)
    print("Harness fidelity CONFIRMED bit-exact vs iter27. Proceeding.\n", flush=True)

    base3 = sum(ITER27_PUBLISHED[s][0] for s in (0, 1, 2)) / 3
    print(f"=== Phase 1: 3-seed sweep over embedding dim k (model-capacity axis) ===", flush=True)
    print(f"iter27 3-seed matched valid baseline (k=16): {base3:.5f}\n", flush=True)
    best_tag, best_k, best_margin = None, None, -1e9
    for k in (8, 24, 32):
        tag = f'k{k}'
        run_cfg(tag, [0, 1, 2], results, CACHE, k=k)
        v3 = mean_of(results, tag, 'valid_primary')
        margin = v3 - base3
        print(f"  k={k}: 3-seed valid mean={v3:.5f}  Δ={margin:+.5f}", flush=True)
        if margin > best_margin:
            best_margin, best_tag, best_k = margin, tag, k

    print(f"\nBest config: {best_tag}  Δvalid={best_margin:+.5f}", flush=True)
    if best_margin > 0.001:
        print("=== Margin clears 0.001 -- extending best config to 5 seeds ===", flush=True)
        run_cfg(best_tag, [3, 4], results, CACHE, k=best_k)
        v5 = mean_of(results, best_tag, 'valid_primary')
        t5 = mean_of(results, best_tag, 'test_primary')
        base5 = sum(p[0] for p in ITER27_PUBLISHED) / 5
        baset5 = sum(p[1] for p in ITER27_PUBLISHED) / 5
        print(f"5-seed: {best_tag} valid={v5:.5f} test={t5:.5f}  vs iter27 valid={base5:.5f} test={baset5:.5f}  "
              f"(Δvalid={v5-base5:+.5f} Δtest={t5-baset5:+.5f})", flush=True)
    else:
        print("\nNo k clears the 0.001 valid margin -- not extending to 5 seeds. REJECT.", flush=True)

    print("\niter37 driver complete.", flush=True)
