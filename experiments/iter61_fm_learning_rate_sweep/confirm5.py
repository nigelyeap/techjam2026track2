"""iter61 (part 3): 5-seed confirmation of FM learning_rate=0.0005."""
import os, sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'iter27_triple_fusion'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'iter47_stacking_meta'))
from evaluate import evaluate  # noqa: E402
from data_ext import load_ext, encode_ext, HALFLIVES, TAB_HALFLIVES  # noqa: E402
import stack as iter47  # noqa: E402

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')
SEEDS = (0, 1, 2, 3, 4)

if __name__ == '__main__':
    splits = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    enc, dim = encode_ext(splits, feature_set=FEATURES, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                           alpha=0.5, n_buckets=20)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    va_list, te_list = [], []
    for seed in SEEDS:
        m_new = iter47.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits['train'], dim, seed=seed, lr=0.0005)
        va_new = evaluate(uva, yva, iter47.sigmoid(m_new.predict(Xva)))
        te_new = evaluate(ute, yte, iter47.sigmoid(m_new.predict(Xte)))

        m_old = iter47.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits['train'], dim, seed=seed, lr=0.001)
        va_old = evaluate(uva, yva, iter47.sigmoid(m_old.predict(Xva)))
        te_old = evaluate(ute, yte, iter47.sigmoid(m_old.predict(Xte)))

        print(f"seed={seed}  lr=0.0005: valid={va_new['primary']:.5f} test={te_new['primary']:.5f}  |  "
              f"lr=0.001: valid={va_old['primary']:.5f} test={te_old['primary']:.5f}  |  "
              f"delta_valid={va_new['primary']-va_old['primary']:+.5f}", flush=True)
        va_list.append((va_new['primary'], va_old['primary']))
        te_list.append((te_new['primary'], te_old['primary']))

    va_new_arr = np.array([x[0] for x in va_list]); va_old_arr = np.array([x[1] for x in va_list])
    te_new_arr = np.array([x[0] for x in te_list]); te_old_arr = np.array([x[1] for x in te_list])
    print(f"\n5-seed valid: lr=0.0005 mean={va_new_arr.mean():.5f} std={va_new_arr.std():.5f}  |  "
          f"lr=0.001 mean={va_old_arr.mean():.5f} std={va_old_arr.std():.5f}  |  "
          f"mean_delta={va_new_arr.mean()-va_old_arr.mean():+.5f}  wins={int((va_new_arr>va_old_arr).sum())}/5")
    print(f"5-seed test:  lr=0.0005 mean={te_new_arr.mean():.5f} std={te_new_arr.std():.5f}  |  "
          f"lr=0.001 mean={te_old_arr.mean():.5f} std={te_old_arr.std():.5f}  |  "
          f"mean_delta={te_new_arr.mean()-te_old_arr.mean():+.5f}")
