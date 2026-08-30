"""iter45: retest CatBoost on iter44's GBM-native (un-bucketed) encoding.

iter43 rejected CatBoost, but only on FM's bucketed encoding (same
bottleneck iter44 diagnosed and fixed for LightGBM). Never retested on
the native encoding -- flagged as a cheap follow-up in iter44's RESULT.md
and SUBMISSION.md. Reuses iter44's prepare()/CAT_COLS/NUM_COLS unchanged
so the feature set is bit-for-bit identical to the LightGBM run; only the
model changes.
"""
import os, sys, importlib.util
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402

_ITER44_DIR = os.path.join(_THIS_DIR, '..', 'iter44_gbm_native_features')


def _load_iter44_train():
    path = os.path.join(_ITER44_DIR, 'train.py')
    spec = importlib.util.spec_from_file_location('iter44_train_v2', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_t44 = _load_iter44_train()
CAT_COLS, NUM_COLS = _t44.CAT_COLS, _t44.NUM_COLS


def _to_catboost_frame(df):
    df = df.copy()
    for c in CAT_COLS:
        df[c] = df[c].astype(object).where(df[c].notna(), 'UNK').astype(str)
    for c in NUM_COLS:
        df[c] = df[c].astype(np.float32)
    return df[CAT_COLS + NUM_COLS]


def run(data_dir, depth=6, iterations=500, learning_rate=0.05, l2_leaf_reg=3.0,
        loss_function='YetiRank', seed=0, verbose=False, _cache=None):
    from catboost import CatBoostRanker, Pool

    dfs, y, u = _t44.prepare(data_dir) if _cache is None else _cache
    Xtr, ytr, utr = _t44._sort_by_user(dfs['train'], y['train'], u['train'])
    Xva, yva, uva = _t44._sort_by_user(dfs['valid'], y['valid'], u['valid'])

    Xtr_cb, Xva_cb = _to_catboost_frame(Xtr), _to_catboost_frame(Xva)
    Xte_cb = _to_catboost_frame(dfs['test'])
    cat_idx = list(range(len(CAT_COLS)))

    train_pool = Pool(Xtr_cb, label=ytr, group_id=utr, cat_features=cat_idx)
    valid_pool = Pool(Xva_cb, label=yva, group_id=uva, cat_features=cat_idx)

    model = CatBoostRanker(
        loss_function=loss_function, depth=depth, iterations=iterations,
        learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg,
        random_seed=seed, verbose=False, early_stopping_rounds=30,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=valid_pool)

    va_scores = model.predict(Xva_cb)
    te_scores = model.predict(Xte_cb)
    va = evaluate(uva, yva, va_scores)  # uva/yva are user-sorted, matching Xva_cb's row order
    te = evaluate(u['test'], y['test'], te_scores)
    if verbose:
        print(f"best_iteration={model.get_best_iteration()}  valid={va}  test={te}")
    return model, va, te, (dfs, y, u)


if __name__ == '__main__':
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
    model, va, te, _cache = run(DATA_DIR, verbose=True)
    print("valid:", va)
    print("test:", te)
