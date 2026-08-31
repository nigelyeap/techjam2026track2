"""iter76: can num_leaves=3's extra capacity be tamed by regularization it
was never given a chance to use?

iter74 found num_leaves=3 regresses vs. num_leaves=2 (Δvalid -0.00233) with
best_iteration collapsing 48->24, i.e. rapid overfitting. But iter57
(reg_lambda resweep) and iter58 (min_child_samples resweep) that previously
found those two knobs flat/moot were run AT num_leaves=2, where iter57's own
diagnosis was that "nearly all model flexibility lives in the per-leaf linear
fit" at 2 leaves, leaving the tree-structure regularizer nothing to do.
num_leaves=3 introduces an actual split decision (which of 2 candidate splits
to keep) that DOES have structure for reg_lambda to regularize, and 3 leaves
means less data per leaf than 2 leaves' ~50/50 split -- both min_child_samples
and linear_lambda (leaf-linear L2, tuned at num_leaves=2 in iter53 where
"each leaf gets ~half the training set, plenty of data") could plausibly bind
differently at num_leaves=3. None of these three regularizers have been swept
at any num_leaves other than 2. This is the same "combine methods that already
work" family as iter76's genesis: reuse iter63's proven feature set, iter74's
capacity finding, and Round 17's regularization-sweep methodology together.

Staged coordinate-descent (matches Round 17's own single-axis-sweep style):
  Stage 1: num_leaves=3, sweep reg_lambda, everything else at iter63 defaults.
  Stage 2: num_leaves=3, best reg_lambda from stage 1, sweep min_child_samples.
  Stage 3: num_leaves=3, best from stages 1-2, sweep linear_lambda.
Goal: does ANY point in this space beat iter63's num_leaves=2 baseline
(valid=0.67168)? If not, num_leaves=3's capacity is fundamentally
mismatched to this data/feature combination, not just under-regularized
with default settings, and the GBM hyperparameter space is truly exhausted.
"""
import os, sys, importlib.util
import numpy as np
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
_ITER63_DIR = os.path.join(_REPO_ROOT, 'experiments', 'iter63_decay_tab_rate')
sys.path.insert(0, _REPO_ROOT)
from evaluate import evaluate  # noqa: E402


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t63 = _load_module(os.path.join(_ITER63_DIR, 'train.py'), 'iter76_t63')

BASELINE_VALID = 0.67168
BASELINE_TEST = 0.65353


def run_config(dfs, y, u, num_leaves, reg_lambda, min_child_samples, linear_lambda, seed=0, tag=''):
    Xtr, ytr, utr = t63._sort_by_user(dfs['train'], y['train'], u['train'])
    Xva, yva, uva = t63._sort_by_user(dfs['valid'], y['valid'], u['valid'])
    gtr = np.unique(utr, return_counts=True)[1]
    gva = np.unique(uva, return_counts=True)[1]
    model = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', eval_at=[5],
        num_leaves=num_leaves, learning_rate=0.10, n_estimators=500,
        min_child_samples=min_child_samples, reg_lambda=reg_lambda,
        linear_lambda=linear_lambda, random_state=seed, verbosity=-1,
        n_jobs=-1, linear_tree=True,
    )
    model.fit(Xtr, ytr, group=gtr, eval_set=[(Xva, yva)], eval_group=[gva],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    va_scores = model.predict(dfs['valid'])
    te_scores = model.predict(dfs['test'])
    va = evaluate(u['valid'], y['valid'], va_scores)
    te = evaluate(u['test'], y['test'], te_scores)
    print(f"[{tag}] best_iter={model.best_iteration_} valid={va['primary']:.5f} test={te['primary']:.5f}", flush=True)
    return va['primary'], te['primary']


if __name__ == '__main__':
    DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')

    print("=== preparing features (rate_only, shared) ===", flush=True)
    dfs, y, u = t63.prepare(DATA_DIR, 'rate_only')

    print("\n=== harness-fidelity check (num_leaves=2, defaults) ===", flush=True)
    va0, te0 = run_config(dfs, y, u, num_leaves=2, reg_lambda=1.0, min_child_samples=200,
                           linear_lambda=0.0, tag='baseline (fidelity check)')
    print(f"  expect valid={BASELINE_VALID} test={BASELINE_TEST}", flush=True)
    assert abs(va0 - BASELINE_VALID) < 1e-4 and abs(te0 - BASELINE_TEST) < 1e-4, "harness fidelity check FAILED"
    print("  PASS", flush=True)

    print("\n=== Stage 1: num_leaves=3, sweep reg_lambda ===", flush=True)
    stage1_grid = [1.0, 3.0, 10.0, 30.0, 100.0]
    stage1_results = {}
    for rl in stage1_grid:
        v, t = run_config(dfs, y, u, num_leaves=3, reg_lambda=rl, min_child_samples=200,
                           linear_lambda=0.0, tag=f'nl3_rl{rl}')
        stage1_results[rl] = (v, t)
    best_rl = max(stage1_results, key=lambda k: stage1_results[k][0])
    print(f"  best reg_lambda={best_rl} -> valid={stage1_results[best_rl][0]:.5f}", flush=True)

    print("\n=== Stage 2: num_leaves=3, reg_lambda={} fixed, sweep min_child_samples ===".format(best_rl), flush=True)
    stage2_grid = [200, 500, 1000, 2000, 4000]
    stage2_results = {}
    for mcs in stage2_grid:
        v, t = run_config(dfs, y, u, num_leaves=3, reg_lambda=best_rl, min_child_samples=mcs,
                           linear_lambda=0.0, tag=f'nl3_mcs{mcs}')
        stage2_results[mcs] = (v, t)
    best_mcs = max(stage2_results, key=lambda k: stage2_results[k][0])
    print(f"  best min_child_samples={best_mcs} -> valid={stage2_results[best_mcs][0]:.5f}", flush=True)

    print("\n=== Stage 3: num_leaves=3, reg_lambda={}, min_child_samples={} fixed, sweep linear_lambda ===".format(best_rl, best_mcs), flush=True)
    stage3_grid = [0.0, 0.1, 0.5, 1.0, 3.0]
    stage3_results = {}
    for ll in stage3_grid:
        v, t = run_config(dfs, y, u, num_leaves=3, reg_lambda=best_rl, min_child_samples=best_mcs,
                           linear_lambda=ll, tag=f'nl3_ll{ll}')
        stage3_results[ll] = (v, t)
    best_ll = max(stage3_results, key=lambda k: stage3_results[k][0])
    print(f"  best linear_lambda={best_ll} -> valid={stage3_results[best_ll][0]:.5f}", flush=True)

    best_overall_valid, best_overall_test = stage3_results[best_ll]
    print("\n=== FINAL: best num_leaves=3 config vs num_leaves=2 baseline ===", flush=True)
    print(f"  num_leaves=3, reg_lambda={best_rl}, min_child_samples={best_mcs}, linear_lambda={best_ll}", flush=True)
    print(f"  valid={best_overall_valid:.5f} (baseline {BASELINE_VALID:.5f}, delta {best_overall_valid-BASELINE_VALID:+.5f})", flush=True)
    print(f"  test={best_overall_test:.5f} (baseline {BASELINE_TEST:.5f}, delta {best_overall_test-BASELINE_TEST:+.5f})", flush=True)
