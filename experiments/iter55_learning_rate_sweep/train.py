"""iter55: learning_rate sweep under linear_tree=True.

Unlike iter52/53/54 (which retested OLD, already-rejected knobs against the
new linear_tree baseline), this is a genuinely new hypothesis: linear_tree
changes what each boosting round buys the model (a per-leaf linear fit
instead of a flat constant), so the optimal learning_rate -- tuned back in
iter44 against the old constant-leaf tree -- was never re-validated against
this structural change. Sweeps learning_rate with n_estimators=500 fixed
(early stopping picks best_iteration), num_leaves=2, linear_tree=True,
everything else at iter51's exact winning hyperparameters.
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
    os.path.join(_THIS_DIR, '..', 'iter51_linear_tree', 'train.py'), 'iter55_iter51_train')
run = _iter51.run


if __name__ == '__main__':
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')

    _, _, _, cache = run(DATA_DIR, linear_tree=True, num_leaves=2, seed=0, verbose=False)

    LR_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    results = []
    for lr in LR_GRID:
        _, va, te, _ = run(DATA_DIR, linear_tree=True, num_leaves=2,
                            learning_rate=lr, seed=0, _cache=cache, verbose=False)
        print(f"learning_rate={lr:.2f}  valid={va['primary']:.5f}  test={te['primary']:.5f}", flush=True)
        results.append((lr, va['primary'], te['primary']))

    best = max(results, key=lambda r: r[1])
    print(f"\nbest: learning_rate={best[0]:.2f}  valid={best[1]:.5f}  test={best[2]:.5f}")
    print(f"iter51 baseline (lr=0.05): valid=0.66932 test=0.65146")
