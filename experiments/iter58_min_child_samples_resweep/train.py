"""iter58: min_child_samples resweep under linear_tree=True + learning_rate=0.10.

min_child_samples=200 was tuned in iter44/46 against the OLD constant-leaf
tree at learning_rate=0.05. iter53 checked linear_lambda and iter57 checked
reg_lambda under the new linear_tree=True + learning_rate=0.10 config, both
REJECT (flat plateaus). min_child_samples controls how much data each leaf's
linear fit gets to train on under linear_tree=True, which is a more direct
interaction with the structural change than reg_lambda was -- never
re-checked after both structural changes. Sweeps min_child_samples directly
on top of iter55's exact winning config.
"""
import os, sys, importlib.util
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_iter51 = _load_module(
    os.path.join(_THIS_DIR, '..', 'iter51_linear_tree', 'train.py'), 'iter58_iter51_train')
run = _iter51.run


if __name__ == '__main__':
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')

    _, _, _, cache = run(DATA_DIR, linear_tree=True, num_leaves=2, learning_rate=0.10, seed=0, verbose=False)

    MCS_GRID = [20, 50, 100, 150, 200, 300, 400, 600, 800, 1200]
    results = []
    for mcs in MCS_GRID:
        _, va, te, _ = run(DATA_DIR, linear_tree=True, num_leaves=2, learning_rate=0.10,
                            min_child_samples=mcs, seed=0, _cache=cache, verbose=False)
        print(f"min_child_samples={mcs:5d}  valid={va['primary']:.5f}  test={te['primary']:.5f}", flush=True)
        results.append((mcs, va['primary'], te['primary']))

    best = max(results, key=lambda r: r[1])
    print(f"\nbest: min_child_samples={best[0]}  valid={best[1]:.5f}  test={best[2]:.5f}")
    print(f"iter55 baseline (min_child_samples=200): valid=0.67052 test=0.65277")
