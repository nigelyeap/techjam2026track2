"""iter44b: hyperparameter sweep on the native-feature LightGBM ranker.

Default config (num_leaves=15, min_child_samples=200, reg_lambda=1.0 --
carried over from iter41's bucketed-feature sweep winner) already closed
almost the entire gap to FM (valid 0.63935 vs FM's 0.63988). Test whether
the native (un-bucketed) representation wants a *different* capacity/
regularization regime than the bucketed one did -- it carries strictly
more information per split, so it may tolerate (or want) less aggressive
regularization than iter41's winner needed.
"""
import os, sys, json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import train as gbm_train

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')

GRID = [
    dict(num_leaves=15, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=1.0),  # current default/baseline
    dict(num_leaves=31, learning_rate=0.05, n_estimators=500, min_child_samples=100, reg_lambda=0.5),
    dict(num_leaves=31, learning_rate=0.05, n_estimators=500, min_child_samples=50, reg_lambda=0.0),
    dict(num_leaves=63, learning_rate=0.03, n_estimators=800, min_child_samples=50, reg_lambda=0.5),
    dict(num_leaves=15, learning_rate=0.03, n_estimators=1000, min_child_samples=200, reg_lambda=1.0),
    dict(num_leaves=15, learning_rate=0.05, n_estimators=500, min_child_samples=300, reg_lambda=2.0),
    dict(num_leaves=7, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=1.0),
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
    with open(os.path.join(_THIS_DIR, 'sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nwrote sweep_results.json")


if __name__ == '__main__':
    main()
