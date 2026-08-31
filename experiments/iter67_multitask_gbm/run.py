"""iter67: independent re-verification of Xuxia's GBM-side multi-task
stacking finding (see experiments/LEDGER.md). 4 auxiliary LightGBM
classifiers (is_like/is_follow/is_comment/is_forward) trained on the same
native feature set as the main GBM, via 5-fold OOF on train (no leakage)
and full-train-fit for valid/test, fed back into the main `long_view` GBM
as 4 new numeric columns. Single-seed check against iter63's rate_only
baseline (valid=0.67168); a >0.001 valid gain would need a 5-seed confirm
per protocol, otherwise this is a clean single-seed REJECT/CONFIRM.
"""
import os, sys, importlib.util
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _REPO_ROOT)
from evaluate import evaluate  # noqa: E402
from aux_labels import load_aux_labels  # noqa: E402

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
AUX_TARGETS = ('is_like', 'is_follow', 'is_comment', 'is_forward')
N_FOLDS = 5


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sort_by_user(df, y, u):
    u = np.asarray(u)
    order = np.argsort(u, kind='stable')
    return df.iloc[order].reset_index(drop=True), y[order], u[order]


if __name__ == '__main__':
    t = _load_module(os.path.join(_REPO_ROOT, 'experiments', 'iter63_decay_tab_rate', 'train.py'),
                      'iter63_train_for_67')

    print("=== preparing iter63 rate_only features (cached) ===", flush=True)
    dfs, y, u = t.prepare(DATA_DIR, 'rate_only')
    splits = t._de.load_ext(DATA_DIR)
    origidx = {name: np.array([x[t._de.IDX['orig_idx']] for x in splits[name]]) for name in ('train', 'valid', 'test')}

    print("=== re-parsing raw CSVs for aux engagement labels ===", flush=True)
    aux_full = load_aux_labels(DATA_DIR)

    # alignment verification: reconstructed long_view via orig_idx must exactly match
    # the already-trusted y[name] from data_ext.py, for every row in every split
    for name in ('train', 'valid', 'test'):
        recon = aux_full['long_view'][origidx[name]]
        trusted = y[name].astype(np.int8)
        n_mismatch = int((recon != trusted).sum())
        print(f"  alignment check [{name}]: {len(recon)} rows, mismatches={n_mismatch}", flush=True)
        assert n_mismatch == 0, f"orig_idx alignment broken for {name}"
    print("  PASS: orig_idx alignment exact on all splits (0 mismatches)", flush=True)

    aux = {name: {tgt: aux_full[tgt][origidx[name]] for tgt in AUX_TARGETS} for name in ('train', 'valid', 'test')}
    for tgt in AUX_TARGETS:
        rate = aux['train'][tgt].mean()
        print(f"  train prevalence {tgt}: {rate:.4%}", flush=True)

    print(f"\n=== baseline (rate_only, no aux columns) seed=0 ===", flush=True)
    _, va_base, te_base, _ = t.run(DATA_DIR, 'rate_only', seed=0, _cache=(dfs, y, u), verbose=True)

    print(f"\n=== training 4 auxiliary classifiers ({N_FOLDS}-fold OOF on train) ===", flush=True)
    Xtr_full, Xva_full, Xte_full = dfs['train'], dfs['valid'], dfs['test']
    oof_cols = {tgt: np.zeros(len(Xtr_full), dtype=np.float64) for tgt in AUX_TARGETS}
    va_cols = {tgt: np.zeros(len(Xva_full), dtype=np.float64) for tgt in AUX_TARGETS}
    te_cols = {tgt: np.zeros(len(Xte_full), dtype=np.float64) for tgt in AUX_TARGETS}

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    for tgt in AUX_TARGETS:
        ytgt = aux['train'][tgt].astype(np.float32)
        for fold, (tr_idx, ho_idx) in enumerate(kf.split(Xtr_full)):
            clf = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.1,
                                      random_state=0, verbosity=-1, n_jobs=-1)
            clf.fit(Xtr_full.iloc[tr_idx], ytgt[tr_idx])
            oof_cols[tgt][ho_idx] = clf.predict_proba(Xtr_full.iloc[ho_idx])[:, 1]
        clf_full = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.1,
                                       random_state=0, verbosity=-1, n_jobs=-1)
        clf_full.fit(Xtr_full, ytgt)
        va_cols[tgt] = clf_full.predict_proba(Xva_full)[:, 1]
        te_cols[tgt] = clf_full.predict_proba(Xte_full)[:, 1]
        print(f"  {tgt}: OOF train AP-ish mean_pred={oof_cols[tgt].mean():.4f} "
              f"(actual rate={ytgt.mean():.4f})", flush=True)

    dfs_aug = {}
    for name, base_df in (('train', Xtr_full), ('valid', Xva_full), ('test', Xte_full)):
        src = oof_cols if name == 'train' else (va_cols if name == 'valid' else te_cols)
        df2 = base_df.copy()
        for tgt in AUX_TARGETS:
            df2[f'aux_{tgt}'] = src[tgt]
        dfs_aug[name] = df2

    print(f"\n=== training main GBM WITH 4 aux columns (seed=0) ===", flush=True)
    Xtr, ytr, utr = sort_by_user(dfs_aug['train'], y['train'], u['train'])
    Xva, yva, uva = sort_by_user(dfs_aug['valid'], y['valid'], u['valid'])
    gtr = np.unique(utr, return_counts=True)[1]
    gva = np.unique(uva, return_counts=True)[1]
    model = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', eval_at=[5],
        num_leaves=2, learning_rate=0.10, n_estimators=500, min_child_samples=200,
        reg_lambda=1.0, random_state=0, verbosity=-1, n_jobs=-1, linear_tree=True,
    )
    model.fit(Xtr, ytr, group=gtr, eval_set=[(Xva, yva)], eval_group=[gva],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    va_scores = model.predict(dfs_aug['valid'])
    te_scores = model.predict(dfs_aug['test'])
    va_aug = evaluate(u['valid'], y['valid'], va_scores)
    te_aug = evaluate(u['test'], y['test'], te_scores)

    print(f"\nbaseline (no aux):  valid={va_base['primary']:.5f} test={te_base['primary']:.5f}")
    print(f"with aux columns:   valid={va_aug['primary']:.5f} test={te_aug['primary']:.5f}")
    print(f"delta:              valid={va_aug['primary']-va_base['primary']:+.5f} "
          f"test={te_aug['primary']-te_base['primary']:+.5f}")

    importances = dict(zip(model.feature_name_, model.feature_importances_))
    print("\naux column importances (split count):")
    for tgt in AUX_TARGETS:
        print(f"  aux_{tgt}: {importances.get(f'aux_{tgt}', 'MISSING')}")
    print(f"  total trees actually built: {model.best_iteration_ or model.n_estimators}")
