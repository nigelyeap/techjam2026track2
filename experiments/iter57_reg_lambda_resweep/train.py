"""iter57: reg_lambda resweep under linear_tree=True + learning_rate=0.10.

reg_lambda=1.0 was tuned in iter44/46 against the OLD constant-leaf tree at
learning_rate=0.05. iter53 already checked linear_lambda (the leaf-linear-
model's own regularizer) under linear_tree=True and found the default
optimal, but reg_lambda (the tree-structure objective's regularizer) was
never re-checked after BOTH linear_tree=True (iter51) AND
learning_rate=0.10 (iter55) changed the training dynamics. Sweeps
reg_lambda directly on top of iter55's exact winning config.
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
    os.path.join(_THIS_DIR, '..', 'iter51_linear_tree', 'train.py'), 'iter57_iter51_train')
run = _iter51.run


if __name__ == '__main__':
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')

    _, _, _, cache = run(DATA_DIR, linear_tree=True, num_leaves=2, learning_rate=0.10, seed=0, verbose=False)

    LAMBDA_GRID = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    results = []
    for rl in LAMBDA_GRID:
        _, va, te, _ = run(DATA_DIR, linear_tree=True, num_leaves=2, learning_rate=0.10,
                            reg_lambda=rl, seed=0, _cache=cache, verbose=False)
        print(f"reg_lambda={rl:.2f}  valid={va['primary']:.5f}  test={te['primary']:.5f}", flush=True)
        results.append((rl, va['primary'], te['primary']))

    best = max(results, key=lambda r: r[1])
    print(f"\nbest: reg_lambda={best[0]:.2f}  valid={best[1]:.5f}  test={best[2]:.5f}")
    print(f"iter55 baseline (reg_lambda=1.0): valid=0.67052 test=0.65277")
