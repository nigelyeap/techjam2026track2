"""iter41: LightGBM LambdaRank ranker on iter27's exact proven feature set.

Different model family from every prior iteration (FM = bilinear low-rank
factorization; this is gradient-boosted trees) and a different objective
(lambdarank directly optimizes NDCG via pairwise-swap gradients weighted by
the resulting |delta NDCG|, rather than BPR's plain pairwise concordance).
Both are named explicitly as in-scope in the handover doc's resource policy.

Reuses iter27's `load_ext`/`encode_ext` verbatim (via importlib, same
same-named-module-collision avoidance as data_prep.py) for the categorical
field encoding -- the only new code here is the LightGBM group/ranking
plumbing.
"""
import os, sys, importlib.util
import numpy as np
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402

_ITER27_DIR = os.path.join(_THIS_DIR, '..', 'iter27_triple_fusion')


def _load_iter27_data_ext():
    path = os.path.join(_ITER27_DIR, 'data_ext.py')
    spec = importlib.util.spec_from_file_location('iter27_data_ext', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_de = _load_iter27_data_ext()
FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')


def _sort_by_user(X, y, u):
    u = np.asarray(u)
    order = np.argsort(u, kind='stable')
    Xs, ys, us = X[order], y[order], u[order]
    _, group = np.unique(us, return_counts=True)
    # np.unique sorts users too, but since `us` is already sorted (stable
    # sort by u above), group counts come out in row order automatically.
    return Xs, ys, us, group


def prepare(data_dir, feature_set=FEATURES, use_cache=True):
    splits = _de.load_ext(data_dir, use_cache=use_cache)
    enc, dim = _de.encode_ext(splits, feature_set=feature_set)
    out = {}
    for name in ('train', 'valid', 'test'):
        X, y, u = enc[name]
        out[name] = _sort_by_user(X, y, u)
    return out, enc


def run(data_dir, num_leaves=31, learning_rate=0.05, n_estimators=500,
        min_child_samples=50, reg_lambda=0.0, seed=0, verbose=False, _cache=None):
    data, enc = prepare(data_dir) if _cache is None else _cache
    Xtr, ytr, utr, gtr = data['train']
    Xva, yva, uva, gva = data['valid']
    Xte, yte, ute, gte = data['test']

    model = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', eval_at=[5],
        num_leaves=num_leaves, learning_rate=learning_rate,
        n_estimators=n_estimators, min_child_samples=min_child_samples,
        reg_lambda=reg_lambda, random_state=seed, verbosity=-1,
        n_jobs=-1,
    )
    cat_idx = list(range(Xtr.shape[1]))
    model.fit(
        Xtr, ytr, group=gtr,
        eval_set=[(Xva, yva)], eval_group=[gva],
        categorical_feature=cat_idx,
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    va_scores = model.predict(Xva)
    te_scores = model.predict(Xte)
    va = evaluate(uva, yva, va_scores)
    te = evaluate(ute, yte, te_scores)
    if verbose:
        print(f"best_iteration={model.best_iteration_}  valid={va}  test={te}")

    # also predict in ORIGINAL (unsorted) row order, for score-level blending
    # with other models (e.g. the FM ensemble) that use encode_ext's own order.
    Xva_orig, yva_orig, uva_orig = enc['valid']
    Xte_orig, yte_orig, ute_orig = enc['test']
    va_scores_orig = model.predict(Xva_orig)
    te_scores_orig = model.predict(Xte_orig)

    return model, va, te, (data, enc), (va_scores_orig, te_scores_orig)


if __name__ == '__main__':
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
    model, va, te, _cache, _orig_scores = run(DATA_DIR, verbose=True)
    print("valid:", va)
    print("test:", te)
