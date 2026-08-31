"""5-seed robustness check on the surprising -0.029 valid regression from
adding v_tag_primary alone (isolated via debug.py). Large enough magnitude
that it's worth checking it's not a seed=0 fluke (early-stopping tie-break
sensitivity, etc.) before writing it up as a confirmed REJECT."""
import os, sys, importlib.util
sys.path.insert(0, '.')

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

t63 = _load_module('experiments/iter63_decay_tab_rate/train.py', 't63confirm5')
r68 = _load_module('experiments/iter68_side_info_native/run.py', 'r68confirm5')
DATA_DIR = 'KuaiRand-Pure/data'

dfs, y, u = t63.prepare(DATA_DIR, 'rate_only')
dfs_aug, new_cats, new_nums = r68.augment(dfs, {'video_basic'})
tag_cols = t63.CAT_COLS + ['v_tag_primary']
base_cols = t63.CAT_COLS
num_cols = t63.VARIANT_NUM_COLS['rate_only']

print(f"{'seed':<6}{'base_valid':>12}{'tag_valid':>12}{'Δvalid':>10}{'base_test':>12}{'tag_test':>12}{'Δtest':>10}")
deltas_va, deltas_te = [], []
for seed in range(5):
    vb, tb = r68.train_eval(dfs, y, u, base_cols, num_cols, seed=seed)
    vt, tt = r68.train_eval(dfs_aug, y, u, tag_cols, num_cols, seed=seed)
    dv = vt['primary'] - vb['primary']
    dt = tt['primary'] - tb['primary']
    deltas_va.append(dv); deltas_te.append(dt)
    print(f"{seed:<6}{vb['primary']:>12.5f}{vt['primary']:>12.5f}{dv:>+10.5f}{tb['primary']:>12.5f}{tt['primary']:>12.5f}{dt:>+10.5f}")

import numpy as np
print(f"\nmean Δvalid={np.mean(deltas_va):+.5f}  wins={sum(1 for d in deltas_va if d>0)}/5")
print(f"mean Δtest ={np.mean(deltas_te):+.5f}  wins={sum(1 for d in deltas_te if d>0)}/5")
