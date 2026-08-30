"""iter52: capacity (num_leaves) resweep under linear_tree=True.

iter46's capacity sweep (num_leaves down to the floor of 2) was run against
the OLD constant-leaf GBM, where every extra leaf only buys another flat
constant. iter51 found that linear_tree=True turns each leaf into a linear
model instead -- a structural change to what capacity even means, so the
old "shrink to the floor" conclusion is not guaranteed to transfer. This
resweeps num_leaves with linear_tree=True fixed on, everything else at
iter51's exact winning hyperparameters (lr=0.05, n_estimators=500,
min_child_samples=200, reg_lambda=1.0), seed=0 first pass, then 5-seed
confirms whichever value wins if it beats iter51's num_leaves=2 baseline
(valid=0.66932) by more than the 0.0003 look-threshold.
"""
import os, sys, importlib.util
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_iter51 = _load_module(
    os.path.join(_THIS_DIR, '..', 'iter51_linear_tree', 'train.py'), 'iter52_iter51_train')

run = _iter51.run  # identical signature: run(data_dir, linear_tree, num_leaves, ..., seed, _cache)

if __name__ == '__main__':
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
    LEAVES_GRID = [2, 3, 4, 5, 7, 10, 15, 20, 31]

    _, _, _, cache = run(DATA_DIR, linear_tree=True, num_leaves=2, seed=0, verbose=False)

    print(f"baseline (iter51, num_leaves=2): valid=0.66932 test=0.65146\n")
    results = []
    for nl in LEAVES_GRID:
        _, va, te, _ = run(DATA_DIR, linear_tree=True, num_leaves=nl, seed=0, _cache=cache, verbose=False)
        print(f"  num_leaves={nl:3d}  valid={va['primary']:.5f}  test={te['primary']:.5f}", flush=True)
        results.append((nl, va['primary'], te['primary']))

    best_nl, best_va, best_te = max(results, key=lambda r: r[1])
    print(f"\nbest: num_leaves={best_nl}  valid={best_va:.5f}  test={best_te:.5f}")
    gain = best_va - 0.66932
    print(f"gain over iter51 baseline: {gain:.5f}")

    if gain > 0.0003 and best_nl != 2:
        print(f"=== clears 0.0003 look-threshold, running 4 more seeds at num_leaves={best_nl} ===")
        vas, tes = [best_va], [best_te]
        for s in (1, 2, 3, 4):
            _, va_s, te_s, _ = run(DATA_DIR, linear_tree=True, num_leaves=best_nl, seed=s, _cache=cache)
            print(f"  seed={s} valid={va_s['primary']:.5f} test={te_s['primary']:.5f}", flush=True)
            vas.append(va_s['primary']); tes.append(te_s['primary'])
        print(f"\n5-seed valid: mean={np.mean(vas):.5f} min={np.min(vas):.5f} max={np.max(vas):.5f}")
        print(f"5-seed test:  mean={np.mean(tes):.5f} min={np.min(tes):.5f} max={np.max(tes):.5f}")
    else:
        print("(no improvement clearing the look-threshold at a different num_leaves -- iter51's num_leaves=2 stands)")
