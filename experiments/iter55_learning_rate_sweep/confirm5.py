import os, sys, importlib.util
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_iter51 = _load_module(os.path.join(_THIS_DIR, '..', 'iter51_linear_tree', 'train.py'), 'iter55_confirm_iter51_train')
run = _iter51.run

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
_, _, _, cache = run(DATA_DIR, linear_tree=True, num_leaves=2, learning_rate=0.10, seed=0, verbose=False)

vas, tes = [], []
for s in range(5):
    _, va, te, _ = run(DATA_DIR, linear_tree=True, num_leaves=2, learning_rate=0.10, seed=s, _cache=cache, verbose=False)
    print(f"seed={s} valid={va['primary']:.5f} test={te['primary']:.5f}", flush=True)
    vas.append(va['primary']); tes.append(te['primary'])

print(f"\n5-seed valid: mean={np.mean(vas):.5f} min={np.min(vas):.5f} max={np.max(vas):.5f} std={np.std(vas):.5f}")
print(f"5-seed test:  mean={np.mean(tes):.5f} min={np.min(tes):.5f} max={np.max(tes):.5f} std={np.std(tes):.5f}")
print(f"\ncompare iter51 baseline (lr=0.05) 5-seed: mean valid=0.66926 mean test=0.65140")
