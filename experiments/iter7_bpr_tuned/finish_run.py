"""Finish iter7: run alpha=1.5, k=16, lr=0.001 for seeds 3 and 4 (seeds 0-2 already
in sweep_results.json from phase 1). Appends new rows to sweep_results.json."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from data import load
from train import run_bpr

DATA_DIR = '../../KuaiRand-Pure/data'
OUT = 'sweep_results.json'

def main():
    print(f"loading {DATA_DIR} ...")
    splits = load(DATA_DIR)

    with open(OUT) as f:
        results = json.load(f)
    done = {(r['name'], r['seed']) for r in results}

    cfg = {'name': 'alpha=1.5', 'alpha': 1.5, 'k': 16, 'lr': 0.001}
    for seed in (3, 4):
        key = (cfg['name'], seed)
        if key in done:
            print(f"skip {cfg['name']} seed={seed} (already done)")
            continue
        t0 = time.time()
        res = run_bpr(splits, k=cfg['k'], lr=cfg['lr'], alpha=cfg['alpha'], seed=seed, verbose=False)
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

    print("\nfinish_run done.")

if __name__ == '__main__':
    main()
