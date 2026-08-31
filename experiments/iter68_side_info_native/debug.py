import os, sys, importlib.util
import numpy as np
import pandas as pd
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _REPO_ROOT)
from evaluate import evaluate

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t63 = _load_module(os.path.join(_REPO_ROOT, 'experiments', 'iter63_decay_tab_rate', 'train.py'), 'iter63_train_for_68dbg')
r68 = _load_module(os.path.join(_THIS_DIR, 'run.py'), 'iter68_run_for_dbg')

dfs, y, u = t63.prepare(DATA_DIR, 'rate_only')

# check the NaN source
lut = r68._load_video_basic_lut()
vid_raw = dfs['valid']['video_id'].astype(str)
missing = [k for k in vid_raw.unique() if k not in lut]
print('video_ids in valid not in lut:', len(missing), missing[:5])

for sub in (['v_type'], ['v_upload_type'], ['v_tag_primary']):
    dfs_aug, new_cats, new_nums = r68.augment(dfs, {'video_basic'})
    cols = [c for c in new_cats if c in sub]
    va, te = r68.train_eval(dfs_aug, y, u, t63.CAT_COLS + cols, t63.VARIANT_NUM_COLS['rate_only'],
                             seed=0, verbose=True, tag=f'only {sub}')
