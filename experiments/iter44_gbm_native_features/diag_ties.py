"""Diagnostic: is the num_leaves=2..7 "improvement" real, or a metric
artifact from score ties?

evaluate.py's nDCG@5 sorts each user's rows by score using Python's stable
sort (`lst.sort(key=lambda x: -x[0])`). Ties preserve original row order.
GAUC's AUC calc does proper rank-averaging for ties (order-invariant), but
nDCG@5 does NOT -- a heavily-tied model's nDCG@5 silently inherits whatever
ranking quality already exists in the raw row order (e.g. chronological
order in the source log), for free, with no real prediction happening.

Small-num_leaves LightGBM models produce few distinct additive leaf-value
combinations -> lots of exact score ties -> exactly the failure mode this
checks for.
"""
import os, sys, importlib.util
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402
import train as gbm_train

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')


def tie_stats(scores, uids):
    uids = np.asarray(uids)
    scores = np.asarray(scores)
    n_unique_overall = len(np.unique(scores))
    frac_in_ties = 0
    total = 0
    per_user_unique_frac = []
    for u in np.unique(uids):
        s = scores[uids == u]
        nu = len(np.unique(s))
        per_user_unique_frac.append(nu / len(s))
        total += len(s)
    return n_unique_overall, len(scores), np.mean(per_user_unique_frac)


def main():
    data = gbm_train.prepare(DATA_DIR)
    dfs, y, u = data
    uva, yva = u['valid'], y['valid']

    print("=== trivial baseline: constant score for everyone (no model at all) ===")
    const_scores = np.zeros(len(yva))
    r_const = evaluate(uva, yva, const_scores)
    print(f"  ALL-CONSTANT SCORE (pure original row order): {r_const}")

    print("\n=== trivial baseline: uniform random score ===")
    rng = np.random.default_rng(0)
    rand_scores = rng.uniform(size=len(yva))
    r_rand = evaluate(uva, yva, rand_scores)
    print(f"  RANDOM SCORE: {r_rand}")

    for nl in [2, 3, 5, 7, 15, 31]:
        model, va, te, _ = gbm_train.run(DATA_DIR, seed=0, verbose=False, _cache=data,
                                          num_leaves=nl, learning_rate=0.05, n_estimators=500,
                                          min_child_samples=200, reg_lambda=1.0)
        va_scores = model.predict(dfs['valid'])
        n_unique, n_total, mean_frac_unique = tie_stats(va_scores, uva)
        print(f"\nnum_leaves={nl}: valid={va}")
        print(f"  unique score values overall: {n_unique}/{n_total}")
        print(f"  mean fraction of unique scores WITHIN a user's own row set: {mean_frac_unique:.4f}"
              f"  (1.0 = no ties at all, low = heavy ties)")


if __name__ == '__main__':
    main()
