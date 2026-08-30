"""iter44d: sweep.py's winner (num_leaves=7) beat FM's ensemble outright
(valid 0.64632 vs FM 0.63988, test 0.64412 vs FM 0.64187) -- and the trend
across sweep.py's results is monotonically better as num_leaves shrinks
(255 -> ... -> 15 -> 7). Push further: is 7 the true optimum, or does it
keep improving down to even smaller trees (which would be a strong signal
of severe overfitting at any larger capacity on this feature set)?
"""
import os, sys, json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import train as gbm_train

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')

GRID = [
    dict(num_leaves=2, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=1.0),
    dict(num_leaves=3, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=1.0),
    dict(num_leaves=4, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=1.0),
    dict(num_leaves=5, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=1.0),
    dict(num_leaves=6, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=1.0),
    dict(num_leaves=7, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=1.0),  # repeat of sweep.py winner, sanity check
    dict(num_leaves=8, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=1.0),
    dict(num_leaves=9, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=1.0),
    dict(num_leaves=10, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=1.0),
    dict(num_leaves=7, learning_rate=0.03, n_estimators=800, min_child_samples=200, reg_lambda=1.0),
    dict(num_leaves=7, learning_rate=0.05, n_estimators=500, min_child_samples=100, reg_lambda=1.0),
    dict(num_leaves=7, learning_rate=0.05, n_estimators=500, min_child_samples=400, reg_lambda=1.0),
    dict(num_leaves=7, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=2.0),
    dict(num_leaves=7, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=0.5),
]


def main():
    data = gbm_train.prepare(DATA_DIR)
    results = []
    for i, cfg in enumerate(GRID):
        model, va, te, _ = gbm_train.run(DATA_DIR, seed=0, verbose=False, _cache=data, **cfg)
        print(f"[{i}] {cfg} -> valid={va['primary']:.5f} test={te['primary']:.5f} "
              f"best_iter={model.best_iteration_}", flush=True)
        results.append({'cfg': cfg, 'valid': float(va['primary']), 'test': float(te['primary']),
                         'best_iteration': int(model.best_iteration_)})
    results.sort(key=lambda r: -r['valid'])
    print("\n=== ranked by valid primary ===")
    for r in results:
        print(f"  valid={r['valid']:.5f} test={r['test']:.5f}  {r['cfg']}")
    with open(os.path.join(_THIS_DIR, 'sweep2_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nwrote sweep2_results.json")


if __name__ == '__main__':
    main()
