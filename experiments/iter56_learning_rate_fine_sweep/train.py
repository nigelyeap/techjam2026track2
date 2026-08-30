"""iter56: fine-grained learning_rate sweep around iter55's winner (0.10).

iter55's coarse sweep {0.01..0.20} was highly non-monotonic (0.07 collapsed
to 0.639 between two much better points at 0.05 and 0.10) -- evidence of a
narrow, jagged landscape at this ultra-low-capacity regime rather than a
smooth one. This checks whether an even better point exists near 0.10 that
the coarse grid skipped over.
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
    os.path.join(_THIS_DIR, '..', 'iter51_linear_tree', 'train.py'), 'iter56_iter51_train')
run = _iter51.run


if __name__ == '__main__':
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')

    _, _, _, cache = run(DATA_DIR, linear_tree=True, num_leaves=2, seed=0, verbose=False)

    LR_GRID = [0.08, 0.085, 0.09, 0.095, 0.10, 0.105, 0.11, 0.115, 0.12, 0.13]
    results = []
    for lr in LR_GRID:
        _, va, te, _ = run(DATA_DIR, linear_tree=True, num_leaves=2,
                            learning_rate=lr, seed=0, _cache=cache, verbose=False)
        print(f"learning_rate={lr:.3f}  valid={va['primary']:.5f}  test={te['primary']:.5f}", flush=True)
        results.append((lr, va['primary'], te['primary']))

    best = max(results, key=lambda r: r[1])
    print(f"\nbest: learning_rate={best[0]:.3f}  valid={best[1]:.5f}  test={best[2]:.5f}")
    print(f"iter55 baseline (lr=0.10): valid=0.67052 test=0.65277")
