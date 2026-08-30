"""iter43: CatBoost ranker (YetiRank / QueryRMSE) -- another genuinely
different open-source library from FM (bilinear factorization) and from
LightGBM (leaf-wise GBDT with a naive integer-categorical treatment).
CatBoost's headline difference: ordered boosting (reduces target leakage
from greedy statistics) and native categorical handling via ordered
target statistics, rather than LightGBM's raw-integer categorical split
search -- worth trying given LightGBM's own warning about "sparse
[categorical] values" on our globally-offset encoding (iter41).

Reuses iter27's `load_ext`/`encode_ext` verbatim, same feature set as the
FM line and iter41's LightGBM ranker, for a fair three-way comparison.
"""
import os, sys, importlib.util
import numpy as np
from catboost import CatBoostRanker, Pool

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402

_ITER27_DIR = os.path.join(_THIS_DIR, '..', 'iter27_triple_fusion')


def _load_iter27_data_ext():
    path = os.path.join(_ITER27_DIR, 'data_ext.py')
    spec = importlib.util.spec_from_file_location('iter27_data_ext_v2', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_de = _load_iter27_data_ext()
FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')


def _sort_by_user(X, y, u):
    u = np.asarray(u)
    order = np.argsort(u, kind='stable')
    Xs, ys, us = X[order], y[order], u[order]
    return Xs, ys, us


def prepare(data_dir, feature_set=FEATURES, use_cache=True):
    splits = _de.load_ext(data_dir, use_cache=use_cache)
    enc, dim = _de.encode_ext(splits, feature_set=feature_set)
    out = {}
    for name in ('train', 'valid', 'test'):
        X, y, u = enc[name]
        out[name] = _sort_by_user(X, y, u)
    return out, enc


def run(data_dir, iterations=1000, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
        loss_function='YetiRank', seed=0, verbose=False, _cache=None):
    data, enc = prepare(data_dir) if _cache is None else _cache
    Xtr, ytr, utr = data['train']
    Xva, yva, uva = data['valid']
    Xte, yte, ute = data['test']

    cat_idx = list(range(Xtr.shape[1]))
    train_pool = Pool(Xtr, label=ytr, group_id=utr, cat_features=cat_idx)
    valid_pool = Pool(Xva, label=yva, group_id=uva, cat_features=cat_idx)

    model = CatBoostRanker(
        iterations=iterations, depth=depth, learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg, loss_function=loss_function,
        random_seed=seed, verbose=False, early_stopping_rounds=50,
        eval_metric='NDCG:top=5',
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    va_scores = model.predict(Xva)
    te_scores = model.predict(Xte)
    va = evaluate(uva, yva, va_scores)
    te = evaluate(ute, yte, te_scores)
    if verbose:
        print(f"best_iteration={model.get_best_iteration()}  valid={va}  test={te}")
    return model, va, te, (data, enc)


if __name__ == '__main__':
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
    model, va, te, _cache = run(DATA_DIR, verbose=True)
    print("valid:", va)
    print("test:", te)
