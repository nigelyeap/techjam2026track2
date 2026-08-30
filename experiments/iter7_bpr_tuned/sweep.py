"""Driver for the iter7 sweep: loads data once, runs all (alpha,k,lr,seed) configs
via run_bpr from train.py, writes results to sweep_results.json incrementally."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from data import load
from train import run_bpr

DATA_DIR = '../../KuaiRand-Pure/data'
OUT = 'sweep_results.json'

def main():
    print(f"loading {DATA_DIR} ...")
    splits = load(DATA_DIR)

    # config, seed grid for the sweep phase (3 seeds each)
    sweep_configs = []
    for alpha in (0.5, 0.75, 1.0, 1.5):
        sweep_configs.append({'name': f'alpha={alpha}', 'alpha': alpha, 'k': 16, 'lr': 0.001})
    # phase 2: best alpha from phase 1 = 1.5 (3-seed valid mean 0.60260, highest).
    # sweep k and lr at that alpha.
    BEST_ALPHA = 1.5
    for k in (24, 32):
        sweep_configs.append({'name': f'k={k}', 'alpha': BEST_ALPHA, 'k': k, 'lr': 0.001})
    for lr in (0.0005, 0.002):
        sweep_configs.append({'name': f'lr={lr}', 'alpha': BEST_ALPHA, 'k': 16, 'lr': lr})

    results = []
    if os.path.exists(OUT):
        with open(OUT) as f:
            results = json.load(f)

    done = {(r['name'], r['seed']) for r in results}
    for cfg in sweep_configs:
        for seed in (0, 1, 2):
            key = (cfg['name'], seed)
            if key in done:
                print(f"skip {cfg['name']} seed={seed} (already done)")
                continue
            t0 = time.time()
            res = run_bpr(splits, k=cfg['k'], lr=cfg['lr'], alpha=cfg['alpha'],
                          seed=seed, verbose=False)
            dt = time.time() - t0
            row = {'name': cfg['name'], 'alpha': cfg['alpha'], 'k': cfg['k'], 'lr': cfg['lr'],
                   'seed': seed, 'valid': float(res['valid']['primary']), 'test': float(res['test']['primary']),
                   'valid_GAUC': float(res['valid']['GAUC']), 'valid_nDCG5': float(res['valid']['nDCG@5']),
                   'test_GAUC': float(res['test']['GAUC']), 'test_nDCG5': float(res['test']['nDCG@5']),
                   'time_s': dt}
            results.append(row)
            print(f"{cfg['name']:12s} seed={seed} valid={row['valid']:.4f} test={row['test']:.4f} ({dt:.1f}s)")
            with open(OUT, 'w') as f:
                json.dump(results, f, indent=2)

    print("\nsweep phase 1 (alpha) done.")

if __name__ == '__main__':
    main()
