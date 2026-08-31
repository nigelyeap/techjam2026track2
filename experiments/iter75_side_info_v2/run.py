"""iter75: retest the REMAINING side-info columns from video_features_basic_pure.csv
that iter68 never touched -- server_width/server_height (video orientation) and
music_type. iter68 tested video_type/upload_type/tag(primary) from this same file
and found only v_tag_primary (44-level unordered categorical) regressed the
num_leaves=2 model; v_type/v_upload_type were no-ops. This is a structurally
different lever from Round 22 (decayed-rate generalization, closed 4/4 REJECT)
and Round 23 (num_leaves resweep, REJECT): a raw side-info categorical retest,
same family as iter68 but on the two fields iter68 left untested.

music_id itself (7202 unique values, near-video-granularity) is deliberately
excluded -- same high-cardinality-destabilization risk already demonstrated by
v_tag_primary (44 levels) in iter68; not worth the risk for a field expected to
be redundant with video_id.

New fields:
  - v_orientation: derived from server_width vs server_height (not the raw
    pixel dimensions, which have 156/120 unique values each and would repeat
    v_tag_primary's cardinality problem). 3 levels: 'portrait' (height>width),
    'landscape' (width>height), 'square' (equal). A genuinely new content
    property (aspect ratio / how the video was shot or cropped) not
    represented by any existing feature.
  - v_music_type: 5 non-null levels + UNK for the 203 rows with missing
    music_type. Low cardinality, genuinely untested.

Ablation (single seed=0, iter55's winning LightGBM config unchanged):
baseline (iter63 rate_only) vs. +orientation vs. +music_type vs. +both.
"""
import os, sys, csv, importlib.util
import numpy as np
import pandas as pd
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _REPO_ROOT)
from evaluate import evaluate  # noqa: E402

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
_ITER63_DIR = os.path.join(_REPO_ROOT, 'experiments', 'iter63_decay_tab_rate')


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t63 = _load_module(os.path.join(_ITER63_DIR, 'train.py'), 'iter63_train_for_75')


def _orientation(w, h):
    if w > h:
        return 'landscape'
    if h > w:
        return 'portrait'
    return 'square'


def _load_video_basic_lut():
    lut = {}
    with open(os.path.join(DATA_DIR, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            w, h = float(r['server_width']), float(r['server_height'])
            mt = r['music_type']
            lut[r['video_id']] = (_orientation(w, h), mt if mt else 'UNK')
    return lut


def augment(dfs, which):
    """which: subset of {'orientation', 'music_type'}"""
    new_cats = []
    dfs = {name: df.copy() for name, df in dfs.items()}
    vid_raw = {name: dfs[name]['video_id'].astype(str) for name in dfs}

    lut = _load_video_basic_lut()
    default = ('UNK', 'UNK')
    fields = []
    if 'orientation' in which:
        fields.append(('v_orientation', 0))
    if 'music_type' in which:
        fields.append(('v_music_type', 1))

    for f, i in fields:
        for name in dfs:
            dfs[name][f] = vid_raw[name].map(lambda k: lut.get(k, default)[i])
        new_cats.append(f)

    cats = {c: pd.CategoricalDtype(categories=dfs['train'][c].unique()) for c in new_cats}
    for name in dfs:
        for c in new_cats:
            dfs[name][c] = dfs[name][c].astype(cats[c])
    return dfs, new_cats


def train_eval(dfs, y, u, cat_cols, num_cols, seed=0, verbose=False, tag=''):
    Xtr, ytr, utr = t63._sort_by_user(dfs['train'][cat_cols + num_cols], y['train'], u['train'])
    Xva, yva, uva = t63._sort_by_user(dfs['valid'][cat_cols + num_cols], y['valid'], u['valid'])
    gtr = np.unique(utr, return_counts=True)[1]
    gva = np.unique(uva, return_counts=True)[1]
    model = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', eval_at=[5],
        num_leaves=2, learning_rate=0.10, n_estimators=500, min_child_samples=200,
        reg_lambda=1.0, random_state=seed, verbosity=-1, n_jobs=-1, linear_tree=True,
    )
    model.fit(Xtr, ytr, group=gtr, eval_set=[(Xva, yva)], eval_group=[gva],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    va_scores = model.predict(dfs['valid'][cat_cols + num_cols])
    te_scores = model.predict(dfs['test'][cat_cols + num_cols])
    va = evaluate(u['valid'], y['valid'], va_scores)
    te = evaluate(u['test'], y['test'], te_scores)
    if verbose:
        print(f"[{tag}] seed={seed} valid={va['primary']:.5f} test={te['primary']:.5f}", flush=True)
    return va, te


if __name__ == '__main__':
    print("=== data sanity: orientation / music_type distributions ===", flush=True)
    lut = _load_video_basic_lut()
    from collections import Counter
    print("orientation:", Counter(v[0] for v in lut.values()), flush=True)
    print("music_type:", Counter(v[1] for v in lut.values()), flush=True)

    print("\n=== preparing iter63 rate_only base features (cached) ===", flush=True)
    dfs0, y, u = t63.prepare(DATA_DIR, 'rate_only')
    base_cat, base_num = t63.CAT_COLS, t63.VARIANT_NUM_COLS['rate_only']

    print("\n=== harness-fidelity check: reproduce iter63 baseline exactly ===", flush=True)
    va0, te0 = train_eval(dfs0, y, u, base_cat, base_num, seed=0, verbose=True, tag='baseline (fidelity check)')
    print(f"  expect valid=0.67168 test=0.65353", flush=True)
    assert abs(va0['primary'] - 0.67168) < 1e-4 and abs(te0['primary'] - 0.65353) < 1e-4, "harness fidelity check FAILED"
    print("  PASS", flush=True)

    configs = [
        ('orientation', {'orientation'}),
        ('music_type', {'music_type'}),
        ('both', {'orientation', 'music_type'}),
    ]

    results = {'baseline': (va0['primary'], te0['primary'])}
    for tag, which in configs:
        print(f"\n=== variant: +{tag} ===", flush=True)
        dfs_aug, new_cats = augment(dfs0, which)
        va, te = train_eval(dfs_aug, y, u, base_cat + new_cats, base_num, seed=0, verbose=True, tag=tag)
        results[tag] = (va['primary'], te['primary'])

    print("\n=== summary (seed 0) ===")
    print(f"{'variant':<14} {'valid':>9} {'test':>9} {'Δvalid':>9} {'Δtest':>9}")
    bva, bte = results['baseline']
    for tag in ['baseline'] + [c[0] for c in configs]:
        v, tt = results[tag]
        print(f"{tag:<14} {v:>9.5f} {tt:>9.5f} {v-bva:>+9.5f} {tt-bte:>+9.5f}")
