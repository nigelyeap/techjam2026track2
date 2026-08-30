"""iter11 sweep driver. Loads data_ext.load_ext() ONCE (raw causal-feature
computation is the expensive shared step, independent of feature_set/alpha),
then runs run_bpr_ext for every (config, seed) via train.py, writing results
incrementally to sweep_results.json so partial progress survives interruption.

Phase 1 (feature ablation, alpha=1.0 default, 3 seeds each):
  {'rate'}, {'tab','rate'}, {'activity','rate'}
  -- iter9 already tested {'activity'}, {'activity','tab'}, {'activity','tab','rate'}
     (see LEDGER.md / iter9 RESULT.md); not rerun here.

Phase 2 (Laplace-smoothing sweep, at the best feature combo from phase 1,
alpha in {0.5, 2.0, 5.0}, 3 seeds each). alpha=1.0 at that combo is reused
from phase 1's results (no rerun needed).
"""
import json, os, time
from data_ext import load_ext
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
OUT = 'sweep_results.json'
SEEDS = (0, 1, 2)


def load_results():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return []


def save_results(results):
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)


def run_config(splits, results, name, feature_set, alpha, seed):
    key = (name, alpha, seed)
    done = {(r['name'], r['alpha'], r['seed']) for r in results}
    if key in done:
        print(f"skip {name} alpha={alpha} seed={seed} (already done)")
        return
    t0 = time.time()
    res = run_bpr_ext(DATA_DIR, feature_set=feature_set, seed=seed, verbose=False,
                       alpha=alpha, splits=splits)
    dt = time.time() - t0
    row = {'name': name, 'features': list(feature_set), 'alpha': alpha, 'seed': seed,
           'valid': float(res['valid']['primary']), 'test': float(res['test']['primary']),
           'valid_GAUC': float(res['valid']['GAUC']), 'valid_nDCG5': float(res['valid']['nDCG@5']),
           'test_GAUC': float(res['test']['GAUC']), 'test_nDCG5': float(res['test']['nDCG@5']),
           'time_s': dt}
    results.append(row)
    print(f"{name:20s} alpha={alpha} seed={seed} valid={row['valid']:.5f} test={row['test']:.5f} ({dt:.1f}s)")
    save_results(results)


def main():
    print(f"loading {DATA_DIR} (causal features computed once, shared across all configs) ...")
    t0 = time.time()
    splits = load_ext(DATA_DIR)
    print(f"loaded in {time.time()-t0:.1f}s")

    results = load_results()

    # ---- Phase 1: feature ablation at alpha=1.0 ----
    phase1 = [
        ('rate', ('rate',)),
        ('tab+rate', ('tab', 'rate')),
        ('activity+rate', ('activity', 'rate')),
    ]
    for name, fset in phase1:
        for seed in SEEDS:
            run_config(splits, results, name, fset, 1.0, seed)

    # pick best combo by mean valid so far (phase 1 configs only)
    def mean_valid(name, alpha):
        vals = [r['valid'] for r in results if r['name'] == name and r['alpha'] == alpha]
        return sum(vals) / len(vals) if vals else -1

    phase1_names = [n for n, _ in phase1]
    best_name = max(phase1_names, key=lambda n: mean_valid(n, 1.0))
    best_fset = dict(phase1)[best_name]
    print(f"\nphase 1 done. best combo so far: {best_name} "
          f"(valid mean {mean_valid(best_name, 1.0):.5f})")

    # ---- Phase 2: alpha sweep at best combo ----
    for alpha in (0.5, 2.0, 5.0):
        for seed in SEEDS:
            run_config(splits, results, best_name, best_fset, alpha, seed)

    print("\nsweep complete.")


if __name__ == '__main__':
    main()
