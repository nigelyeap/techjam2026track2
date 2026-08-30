"""iter43b: CatBoost sweep -- the default run (depth=6, 1000 iters) badly
underperformed (valid 0.6094 vs FM's 0.6389), worse than LightGBM's own
default-config run. LightGBM's sweep (iter41/sweep.py) showed *smaller*
trees do better on this feature set (all larger num_leaves/n_estimators
configs underperformed num_leaves=31/n_estimators=500) -- this data is
low-signal-density (bucketed categorical features, quantile edges already
smooth the input), so big tree ensembles overfit. Test the same hypothesis
for CatBoost: shallower depth, fewer iterations, and the loss function
axis (YetiRank vs QueryRMSE vs PairLogitPairwise).
"""
import os, sys, json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import train as cb_train

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')

GRID = [
    dict(depth=3, iterations=300, learning_rate=0.05, l2_leaf_reg=3.0, loss_function='YetiRank'),
    dict(depth=4, iterations=300, learning_rate=0.05, l2_leaf_reg=3.0, loss_function='YetiRank'),
    dict(depth=4, iterations=500, learning_rate=0.03, l2_leaf_reg=10.0, loss_function='YetiRank'),
    dict(depth=3, iterations=300, learning_rate=0.05, l2_leaf_reg=3.0, loss_function='QueryRMSE'),
    dict(depth=4, iterations=300, learning_rate=0.05, l2_leaf_reg=3.0, loss_function='PairLogitPairwise'),
]


def main():
    data, enc = cb_train.prepare(DATA_DIR)
    results = []
    for i, cfg in enumerate(GRID):
        model, va, te, _ = cb_train.run(DATA_DIR, seed=0, verbose=False, _cache=(data, enc), **cfg)
        print(f"[{i}] {cfg} -> valid={va['primary']:.5f} test={te['primary']:.5f} "
              f"best_iter={model.get_best_iteration()}", flush=True)
        results.append({'cfg': cfg, 'valid': float(va['primary']), 'test': float(te['primary']),
                         'best_iteration': int(model.get_best_iteration())})
    results.sort(key=lambda r: -r['valid'])
    print("\n=== ranked by valid primary ===")
    for r in results:
        print(f"  valid={r['valid']:.5f} test={r['test']:.5f}  {r['cfg']}")
    with open(os.path.join(_THIS_DIR, 'sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nwrote sweep_results.json")


if __name__ == '__main__':
    main()
