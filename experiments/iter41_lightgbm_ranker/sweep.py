"""iter41c: LightGBM hyperparameter sweep on valid only. The standalone
run in RESULT.md used untuned defaults (num_leaves=31, lr=0.05,
n_estimators=500) -- before concluding LightGBM can't compete with FM,
give it a real chance with a small grid search, since default GBM
hyperparameters are rarely optimal on a new dataset/feature set.
"""
import os, sys, json, itertools
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402
import lightgbm as lgb
import train as lgb_train

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')

GRID = [
    dict(num_leaves=31, learning_rate=0.05, n_estimators=500, min_child_samples=50, reg_lambda=0.0),   # baseline (already run)
    dict(num_leaves=63, learning_rate=0.05, n_estimators=800, min_child_samples=50, reg_lambda=0.0),
    dict(num_leaves=127, learning_rate=0.03, n_estimators=1200, min_child_samples=30, reg_lambda=0.0),
    dict(num_leaves=255, learning_rate=0.02, n_estimators=2000, min_child_samples=20, reg_lambda=0.0),
    dict(num_leaves=63, learning_rate=0.05, n_estimators=800, min_child_samples=100, reg_lambda=1.0),
    dict(num_leaves=31, learning_rate=0.1, n_estimators=400, min_child_samples=50, reg_lambda=0.0),
    dict(num_leaves=15, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=1.0),
]


def main():
    data, enc = lgb_train.prepare(DATA_DIR)
    results = []
    for i, cfg in enumerate(GRID):
        model, va, te, _, _ = lgb_train.run(DATA_DIR, seed=0, verbose=False, _cache=(data, enc), **cfg)
        print(f"[{i}] {cfg} -> valid={va['primary']:.5f} test={te['primary']:.5f} best_iter={model.best_iteration_}", flush=True)
        results.append({'cfg': cfg, 'valid': float(va['primary']), 'test': float(te['primary']),
                         'best_iteration': int(model.best_iteration_)})
    results.sort(key=lambda r: -r['valid'])
    print("\n=== ranked by valid primary ===")
    for r in results:
        print(f"  valid={r['valid']:.5f} test={r['test']:.5f}  {r['cfg']}")
    with open(os.path.join(_THIS_DIR, 'sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nwrote sweep_results.json")


if __name__ == '__main__':
    main()
