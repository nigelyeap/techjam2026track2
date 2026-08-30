"""Extend the winning config (+last1+lastk_rate+gap) to seeds 3,4 for a full
5-seed confirmation, matching iter9's protocol. Appends into results.json."""
import json, os, time
from train import run_bpr_ext
from driver import _clean, RESULTS_PATH, DATA_DIR, key

WINNER = ('activity', 'tab', 'rate', 'last1', 'lastk_rate', 'gap')
NAME = '+last1+lastk_rate+gap'

def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)

def save_results(res):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(res, f, indent=2)

def main():
    results = load_results()
    cache = {}
    for seed in (3, 4):
        k = key(NAME, seed)
        if k in results:
            print(f"skip {k} (already done)")
            continue
        print(f"\n=== running {NAME} seed={seed} ===")
        t0 = time.time()
        res = run_bpr_ext(DATA_DIR, feature_set=WINNER, seed=seed, verbose=True, _cache=cache)
        dt = time.time() - t0
        results[k] = {'name': NAME, 'features': list(WINNER), 'seed': seed,
                      'valid': _clean(res['valid']), 'test': _clean(res['test']), 'time_s': dt}
        save_results(results)
        print(f"=== done {k} in {dt:.1f}s | valid primary={res['valid']['primary']:.5f} "
              f"test primary={res['test']['primary']:.5f} ===")

if __name__ == '__main__':
    main()
