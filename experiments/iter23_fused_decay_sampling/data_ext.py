"""iter23: fusion of iter19's INPUT-FEATURE fusion (iter16 decay + iter18
momentum) with iter22's TRAINING-TIME decay-aware BPR user-sampling weight.

This module is iter19's data_ext.py (fused feature computation, unmodified
below this docstring) PLUS one addition: `compute_final_decayed_pos`,
copied verbatim from iter22/data_ext.py -- the non-causal, once-per-user
scalar (recency-decayed count of a user's TRAIN positive rows, decayed to
the end of the train period) that iter22 uses in place of the flat
`pos_len[user] ** alpha` BPR sampling weight. It operates on the SAME
(date, user_id, ..., label, ...) row-tuple prefix (indices 0, 1, 6) that
every other function in this module already relies on, so it works
unmodified against this module's extended row tuples (or the raw
`splits['train']` list) with no adaptation. This is a TRAINING-TIME
sampling-frequency choice, never a per-row model feature, so there is no
leakage risk by construction -- see compute_final_decayed_pos's own
docstring and the PART D causality spot-check in __main__ below for the
required verification anyway.

Original iter19 docstring follows (fused feature computation, unmodified):
---
iter19: fusion of iter16's recency-decayed (exponential half-life) causal
history features with iter18's timestamp-level session-momentum features,
on top of iter9's original flat date-grouped causal features.

Motivation (from LEDGER.md Round 5 summary / iter18's own writeup): iter16
(decay_rate_3/decay_act_3/tab, valid 0.62030) and iter18
(activity/tab/rate/last1/lastk_rate/gap, valid 0.61417) each independently
beat iter9 by finding real signal in HOW history is time-weighted, but were
never combined. They target different time horizons:
  - iter16: multi-day exponential decay of the user's ENTIRE prior history
    (date-level granularity, `<` traversal over `date`).
  - iter18: single-immediately-preceding-interaction / last-5-interaction
    momentum (time_ms-level granularity, `<` traversal over `(time_ms,
    orig_idx)`).
This module computes BOTH families of features over the SAME underlying
per-row data, as two independent causal traversals whose outputs are simply
joined onto the same rows by row index (they don't need to share a
traversal -- see dispatch prompt). Both traversals are reused UNMODIFIED via
importlib from iter16/iter18's own already-verified implementations, rather
than reimplemented from scratch:
  - iter16_recency_decay/data_ext.py: compute_decay_features(rows, halflives)
    -- needs only rows[i] = (date, user_id, ..., label, ...) at indices
    (0, 1, ..., 6); operates correctly on ANY row tuple whose first 7 columns
    match that (date, user_id, video_id, author_id, tab, duration_ms, label)
    layout, regardless of what's appended after index 6.
  - iter9_history_dense/data_ext.py: compute_causal_features(rows) -- same
    prefix-compatibility (needs indices 0, 1, 4, 6).
  - iter18_momentum/data_ext.py: compute_momentum_features(rows, K) -- needs
    user_id at index 1, label at index 6, time_ms at index 8, orig_idx at
    index 9.

Because this module's own row tuple (built by `_load_raw_time`, copied from
iter18's loader so `time_ms`/`orig_idx` are present) has EXACTLY iter18's
10-column prefix (date, user_id, video_id, author_id, tab, duration_ms,
label, hourmin, time_ms, orig_idx), all three imported functions can be
called directly against it with no adaptation -- this is verified explicitly
in this file's __main__ causality-check block (not just assumed).

Extended row tuple layout (see IDX for the authoritative index map):
  0 date, 1 user_id, 2 video_id, 3 author_id, 4 tab, 5 duration_ms, 6 label,
  7 hourmin, 8 time_ms, 9 orig_idx,
  10 activity, 11 tab_pos, 12 prior_pos, 13 prior_total,        <- iter9 flat
  14 last1, 15 lastk_sum, 16 lastk_cnt, 17 gap_ms,               <- iter18 momentum
  18.. decay_pos_h0, decay_total_h0, decay_pos_h1, decay_total_h1, ...  <- iter16 decay
  (h0, h1, ... follow the order of HALFLIVES)

Does NOT modify data.py, iter9's data_ext.py, iter16's data_ext.py, or
iter18's data_ext.py.
"""
import os, sys, csv, collections, importlib.util, pickle
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from data import SPLITS  # noqa: E402  (date ranges only, kept in sync w/ data.py)


def _load_module(name, rel_path):
    path = os.path.join(_THIS_DIR, *rel_path.split('/'))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load iter9/iter16/iter18's data_ext.py modules under distinct module names
# (they're all literally named "data_ext.py" -- a plain `import data_ext`
# would collide/shadow, so use importlib with explicit unique names, same
# pattern iter17/iter18 already established).
_iter9_de = _load_module('iter9_data_ext', '../iter9_history_dense/data_ext.py')
_iter16_de = _load_module('iter16_data_ext', '../iter16_recency_decay/data_ext.py')
_iter18_de = _load_module('iter18_data_ext', '../iter18_momentum/data_ext.py')

compute_causal_features = _iter9_de.compute_causal_features    # date-grouped flat features
compute_decay_features = _iter16_de.compute_decay_features      # date-grouped exponential decay
compute_momentum_features = _iter18_de.compute_momentum_features  # time_ms-level momentum
_date_to_ordinal = _iter16_de._date_to_ordinal

LABEL = 'long_view'
BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
ALPHA = 1.0
K_DEFAULT = 5
HALFLIVES = [3]  # iter16's winning halflife only -- this iteration isn't re-sweeping halflife

IDX = dict(date=0, user_id=1, video_id=2, author_id=3, tab=4, duration_ms=5, label=6,
           hourmin=7, time_ms=8, orig_idx=9,
           activity=10, tab_pos=11, prior_pos=12, prior_total=13,
           last1=14, lastk_sum=15, lastk_cnt=16, gap_ms=17)
DECAY_BASE = 18  # decay_pos_h0 starts here; decay_pos_h{i} at DECAY_BASE+2i, decay_total at +2i+1


def _halflife_col(h, halflives=HALFLIVES):
    pos = halflives.index(h)
    return DECAY_BASE + 2 * pos, DECAY_BASE + 2 * pos + 1


def _load_raw_time(data_dir):
    """Verbatim copy of iter18's _load_raw_time (same files, same row order,
    same vid2author join, same date-range filtering, same orig_idx
    assignment) -- duplicated rather than imported only because iter18's
    version is a module-local closure-free function with no side effects to
    share; behavior is identical by inspection."""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0,
                             int(r['hourmin']), int(r['time_ms']), len(rows)))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


_CACHE_VERSION = 1


def _cache_path(halflives):
    key = '-'.join(str(h) for h in halflives)
    return os.path.join(_THIS_DIR, f'.cache_v{_CACHE_VERSION}_{key}.pkl')


def load_ext(data_dir, halflives=HALFLIVES, K=K_DEFAULT, use_cache=True):
    """Returns dict split -> list of extended rows (see IDX / DECAY_BASE for
    layout). Runs THREE independent causal traversals over the same flat
    (train+valid+test, in that order) row list and joins their outputs by
    row index:
      1. iter9's compute_causal_features   (date-grouped flat activity/tab/rate)
      2. iter18's compute_momentum_features (time_ms-level last1/lastk/gap)
      3. iter16's compute_decay_features    (date-grouped exponential decay)
    Each traversal only reads columns it documents needing (see module
    docstring) and is causally self-contained -- combining them is a pure
    join, not a shared mutable pass, so no cross-family leakage is possible
    by construction.
    """
    cpath = _cache_path(halflives)
    if use_cache and os.path.exists(cpath):
        with open(cpath, 'rb') as fh:
            return pickle.load(fh)

    splits = _load_raw_time(data_dir)
    order = ('train', 'valid', 'test')
    flat, owner = [], []
    for name in order:
        for r in splits[name]:
            flat.append(r)
            owner.append(name)

    day_feats = compute_causal_features(flat)              # indices 0,1,4,6
    mom_feats = compute_momentum_features(flat, K=K)        # indices 1,6,8,9
    decayed_pos, decayed_total = compute_decay_features(flat, halflives)  # indices 0,1,6

    ext = {name: [] for name in order}
    for i, (r, name) in enumerate(zip(flat, owner)):
        extra = [day_feats['activity'][i], day_feats['tab_pos'][i],
                 day_feats['prior_pos'][i], day_feats['prior_total'][i],
                 mom_feats['last1'][i], mom_feats['lastk_sum'][i],
                 mom_feats['lastk_cnt'][i], mom_feats['gap_ms'][i]]
        for h in range(len(halflives)):
            extra.append(decayed_pos[i, h])
            extra.append(decayed_total[i, h])
        ext[name].append(r + tuple(extra))

    if use_cache:
        with open(cpath, 'wb') as fh:
            pickle.dump(ext, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return ext


def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1])


def encode_ext(splits, feature_set=('decay_rate_3', 'decay_act_3', 'tab'), halflives=HALFLIVES):
    """splits: dict from load_ext(). feature_set: subset/order of:
      'activity','tab','rate'         iter9 flat features (bucketed)
      'decay_rate_H','decay_act_H'    iter16 decayed features, H in halflives (bucketed)
      'last1'                         iter18 categorical ('0'/'1'/'UNK')
      'lastk_rate'                    iter18 continuous, bucketed
      'gap'                           iter18 continuous (ms), bucketed, 'UNK' for first row
    Returns (enc, field_dims_sum) with enc[name] = (X, y, users).
    """
    tr = splits['train']
    dur_edges = _bucket_edges([x[IDX['duration_ms']] for x in tr])

    def raw_dur(x):
        return str(int(np.searchsorted(dur_edges, x[IDX['duration_ms']])))

    def parse_feat(name):
        if name in ('activity', 'tab', 'rate', 'last1', 'lastk_rate', 'gap'):
            return name, None
        if name.startswith('decay_rate_'):
            return 'decay_rate', int(name.rsplit('_', 1)[1])
        if name.startswith('decay_act_'):
            return 'decay_act', int(name.rsplit('_', 1)[1])
        raise ValueError(name)

    extra_edges = {}
    for name in feature_set:
        kind, h = parse_feat(name)
        if kind == 'activity':
            extra_edges[name] = _bucket_edges([x[IDX['activity']] for x in tr])
        elif kind == 'tab':
            extra_edges[name] = _bucket_edges([x[IDX['tab_pos']] for x in tr])
        elif kind == 'rate':
            extra_edges[name] = _bucket_edges(
                [(x[IDX['prior_pos']] + ALPHA) / (x[IDX['prior_total']] + 2 * ALPHA) for x in tr])
        elif kind == 'decay_rate':
            pcol, tcol = _halflife_col(h, halflives)
            extra_edges[name] = _bucket_edges([(x[pcol] + ALPHA) / (x[tcol] + 2 * ALPHA) for x in tr])
        elif kind == 'decay_act':
            pcol, tcol = _halflife_col(h, halflives)
            extra_edges[name] = _bucket_edges([x[tcol] for x in tr])
        elif kind == 'lastk_rate':
            extra_edges[name] = _bucket_edges(
                [(x[IDX['lastk_sum']] + ALPHA) / (x[IDX['lastk_cnt']] + 2 * ALPHA) for x in tr])
        elif kind == 'gap':
            gaps = [x[IDX['gap_ms']] for x in tr if x[IDX['gap_ms']] >= 0]
            extra_edges[name] = _bucket_edges(gaps)
        # 'last1' needs no edges (raw categorical)

    def extra_val(x, name):
        kind, h = parse_feat(name)
        if kind == 'activity':
            return str(int(np.searchsorted(extra_edges[name], x[IDX['activity']])))
        elif kind == 'tab':
            return str(int(np.searchsorted(extra_edges[name], x[IDX['tab_pos']])))
        elif kind == 'rate':
            r = (x[IDX['prior_pos']] + ALPHA) / (x[IDX['prior_total']] + 2 * ALPHA)
            return str(int(np.searchsorted(extra_edges[name], r)))
        elif kind == 'decay_rate':
            pcol, tcol = _halflife_col(h, halflives)
            r = (x[pcol] + ALPHA) / (x[tcol] + 2 * ALPHA)
            return str(int(np.searchsorted(extra_edges[name], r)))
        elif kind == 'decay_act':
            pcol, tcol = _halflife_col(h, halflives)
            return str(int(np.searchsorted(extra_edges[name], x[tcol])))
        elif kind == 'last1':
            v = x[IDX['last1']]
            return 'UNK' if v == -1 else str(int(v))
        elif kind == 'lastk_rate':
            r = (x[IDX['lastk_sum']] + ALPHA) / (x[IDX['lastk_cnt']] + 2 * ALPHA)
            return str(int(np.searchsorted(extra_edges[name], r)))
        elif kind == 'gap':
            g = x[IDX['gap_ms']]
            if g < 0:
                return 'UNK'
            return str(int(np.searchsorted(extra_edges[name], g)))
        raise ValueError(name)

    def raw(x):
        return [x[1], x[2], x[3], x[4], raw_dur(x)] + [extra_val(x, nm) for nm in feature_set]

    fields = BASE_FIELDS + list(feature_set)
    vocabs = [dict() for _ in fields]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(fields)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))


def compute_final_decayed_pos(train_rows, halflife=3):
    """Copied verbatim from iter22/data_ext.py's compute_final_decayed_pos.
    NON-CAUSAL, single scalar per user -- the recency-decayed count of that
    user's TRAIN positive rows, decayed to a single fixed reference date =
    the END of the train period (max date present in train). This is a
    TRAINING-TIME SAMPLING WEIGHT, the direct decayed analog of `pos_len`
    (the flat raw positive-row count iter3/iter9/iter16/iter19 all use to
    weight which users get sampled for BPR pairs) -- NOT a per-row feature
    fed to the model. It uses the SAME lazy-decay exponential formula
    (0.5 ** (gap_days / halflife)) as compute_decay_features' `decayed_pos`
    output (matching decay_act_3/decay_rate_3's halflife=3d when called with
    the default), but evaluated once at the final decay state per user
    rather than causally per-row -- matching how `pos_len` itself is already
    a single non-causal aggregate over ALL of train (build_pos_neg_index has
    no per-row causal restriction either; it's a global sampling-frequency
    choice, not something the model sees as a feature). No leakage concern:
    this value never enters any row's feature vector, it only controls how
    OFTEN a user's (already causally-correct) rows get drawn as BPR anchors
    during training.

    train_rows: list of row tuples with date at index 0, user_id at index 1,
    label at index 6 -- this module's extended row tuples share that prefix
    with the raw data.py rows, so this works unmodified against either
    splits['train'] from load_ext() or the raw loader's output.

    Returns: dict user_id -> decayed positive count (float). Users with zero
    train positives are simply absent (matches pos_len==0 users, who are
    never in `eligible` anyway since BPR needs >=1 pos AND >=1 neg per user).
    """
    ref_ord = max(_date_to_ordinal(r[0]) for r in train_rows)
    decayed_pos = collections.defaultdict(float)
    day_mult = 0.5 ** (1.0 / halflife)
    for r in train_rows:
        if r[6] == 1:
            u = r[1]
            gap = ref_ord - _date_to_ordinal(r[0])
            decayed_pos[u] += day_mult ** gap
    return dict(decayed_pos)


if __name__ == '__main__':
    import argparse, datetime, time
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(_THIS_DIR, '..', '..',
                                                         'KuaiRand-Pure', 'data'))
    ap.add_argument('--no_cache', action='store_true')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    t0 = time.time()
    ext = load_ext(a.data_dir, use_cache=not a.no_cache)
    print({k: len(v) for k, v in ext.items()}, f"  ({time.time()-t0:.1f}s)")

    flat = ext['train'] + ext['valid'] + ext['test']
    n = len(flat)

    # ================================================================
    # PART A: decay-family causal spot-checks (adapted from iter16's
    # __main__, re-run here against the COMBINED row tuple to verify the
    # imported compute_decay_features still behaves identically when fed
    # this module's wider row tuple, not just iter16's original one).
    # ================================================================
    print("\n=== PART A: decay-feature causal spot-checks (brute force) ===")
    for h in HALFLIVES:
        pcol, tcol = _halflife_col(h)
        tot = np.array([r[tcol] for r in flat])
        print(f"halflife={h:2d}d  decayed_total>0 coverage: {np.mean(tot > 0)*100:.2f}%  "
              f"(mean={tot.mean():.3f}, max={tot.max():.3f})")

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=25, replace=False)
    max_err = 0.0
    for idx in sample_idx:
        r = flat[idx]
        date, uid = r[0], r[1]
        d_ord = _date_to_ordinal(date)
        earlier = [rr for rr in flat if rr[1] == uid and rr[0] < date]
        for h in HALFLIVES:
            pcol, tcol = _halflife_col(h)
            manual_pos = sum(0.5 ** ((d_ord - _date_to_ordinal(rr[0])) / h)
                              for rr in earlier if rr[6] == 1)
            manual_tot = sum(0.5 ** ((d_ord - _date_to_ordinal(rr[0])) / h) for rr in earlier)
            err_p = abs(manual_pos - r[pcol])
            err_t = abs(manual_tot - r[tcol])
            max_err = max(max_err, err_p, err_t)
            assert err_p < 1e-6, f"CAUSALITY BUG: decayed_pos mismatch h={h} idx={idx}"
            assert err_t < 1e-6, f"CAUSALITY BUG: decayed_total mismatch h={h} idx={idx}"
    print(f"25 random rows x {len(HALFLIVES)} halflives: decayed_pos/decayed_total match brute "
          f"force (max abs err {max_err:.2e}). No leakage detected.")

    zero_examples = [r for r in flat if r[IDX['activity']] == 0][:5]
    for r in zero_examples:
        for h in HALFLIVES:
            pcol, tcol = _halflife_col(h)
            assert r[pcol] == 0.0 and r[tcol] == 0.0, "CAUSALITY BUG: zero-activity row has nonzero decay!"
    print(f"zero-activity rows ({len(zero_examples)} checked): decayed_pos/total correctly 0.0.")

    by_u_date = collections.defaultdict(list)
    for idx, r in enumerate(flat):
        if r[6] == 1:
            by_u_date[(r[1], r[0])].append(idx)
    same_date_case = next((v for v in by_u_date.values() if len(v) >= 2), None)
    if same_date_case:
        h = HALFLIVES[0]
        pcol, tcol = _halflife_col(h)
        vals = [(flat[idx][pcol], flat[idx][tcol]) for idx in same_date_case]
        dp0, dt0 = vals[0]
        for dp, dt in vals:
            assert abs(dp - dp0) < 1e-9 and abs(dt - dt0) < 1e-9, \
                "CAUSALITY BUG: same-date pair should show identical decay state!"
        print(f"same-date-pair edge case ({len(same_date_case)} rows): decay values identical "
              "across the pair, as expected (same-date rows never see each other). PASSED.")
    else:
        print("(no same-date-pair found -- skipping, not a failure)")

    # ================================================================
    # PART B: momentum-family causal spot-checks (adapted from iter18's
    # __main__), re-run against the COMBINED row tuple.
    # ================================================================
    print("\n=== PART B: momentum-feature causal spot-checks (brute force) ===")
    last1 = np.array([r[IDX['last1']] for r in flat])
    lastk_cnt = np.array([r[IDX['lastk_cnt']] for r in flat])
    gap_ms = np.array([r[IDX['gap_ms']] for r in flat])
    print(f"last1 coverage (not user's first row): {np.mean(last1 >= 0)*100:.2f}%")
    print(f"gap coverage (not user's first row): {np.mean(gap_ms >= 0)*100:.2f}%")

    by_user = collections.defaultdict(list)
    for idx, r in enumerate(flat):
        by_user[r[IDX['user_id']]].append(idx)
    candidate_users = [u for u, idxs in by_user.items() if 8 <= len(idxs) <= 14][:3]

    def manual_check(u):
        idxs = sorted(by_user[u], key=lambda i: (flat[i][IDX['time_ms']], flat[i][IDX['orig_idx']]))
        print(f"\nuser={u}  ({len(idxs)} rows, chronological order)")
        window = []
        prev_t = None
        for pos, i in enumerate(idxs):
            r = flat[i]
            t = r[IDX['time_ms']]
            manual_last1 = window[-1] if window else -1
            manual_lastk_sum = sum(window[-5:])
            manual_lastk_cnt = len(window[-5:])
            manual_gap = (t - prev_t) if prev_t is not None else -1
            ok = (manual_last1 == r[IDX['last1']] and manual_lastk_sum == r[IDX['lastk_sum']]
                  and manual_lastk_cnt == r[IDX['lastk_cnt']] and manual_gap == r[IDX['gap_ms']])
            flag = "OK" if ok else "MISMATCH!!"
            print(f"  pos={pos:2d} date={r[IDX['date']]} time_ms={t} label={r[IDX['label']]} "
                  f"| last1 got={r[IDX['last1']]} manual={manual_last1} "
                  f"| lastk_sum got={r[IDX['lastk_sum']]} manual={manual_lastk_sum} "
                  f"| gap_ms got={r[IDX['gap_ms']]} manual={manual_gap}  [{flag}]")
            assert ok, f"CAUSALITY BUG for user {u} at pos {pos}!"
            window.append(r[IDX['label']])
            prev_t = t

    for u in candidate_users:
        manual_check(u)

    print("\n--- synthetic same-time_ms tie stress test (momentum) ---")
    fake_rows = [
        (20220410, 'TIEUSER', 'v1', 'a1', '0', 1000.0, 1, 1000, 5000, 100),
        (20220410, 'TIEUSER', 'v2', 'a1', '0', 1000.0, 0, 1000, 5000, 101),
        (20220410, 'TIEUSER', 'v3', 'a1', '0', 1000.0, 1, 1005, 5500, 99),
    ]
    feats = compute_momentum_features(fake_rows, K=5)
    assert feats['last1'][0] == -1
    assert feats['last1'][1] == 1
    assert feats['last1'][2] == 0
    assert feats['gap_ms'][0] == -1
    assert feats['gap_ms'][1] == 0
    assert feats['gap_ms'][2] == 500
    print("tie stress test: all assertions passed.")

    # ================================================================
    # PART C: cross-family joint edge case -- a same-user, same-date PAIR
    # where the two rows ALSO differ in time_ms (the realistic case: same
    # calendar date, different times of day). Verifies the decay family
    # (date-blind within a date) and momentum family (time_ms-aware) give
    # DIFFERENT, individually-correct answers on the same physical row
    # pair -- i.e. the join didn't accidentally cross-contaminate the two
    # traversals' outputs.
    # ================================================================
    print("\n=== PART C: cross-family joint edge case (same-date, different time_ms pair) ===")
    joint_case = None
    for (uid, date), idxs in by_u_date.items():
        if len(idxs) >= 2:
            times = sorted(set(flat[i][IDX['time_ms']] for i in idxs))
            if len(times) >= 2:
                joint_case = (uid, date, idxs)
                break
    if joint_case:
        uid, date, idxs = joint_case
        idxs_sorted = sorted(idxs, key=lambda i: (flat[i][IDX['time_ms']], flat[i][IDX['orig_idx']]))
        h = HALFLIVES[0]
        pcol, tcol = _halflife_col(h)
        print(f"user={uid} date={date}: {len(idxs_sorted)} rows, same calendar date, distinct time_ms")
        decay_vals = set()
        for rank, i in enumerate(idxs_sorted):
            r = flat[i]
            decay_vals.add((round(r[pcol], 9), round(r[tcol], 9)))
            print(f"  rank={rank} time_ms={r[IDX['time_ms']]} label={r[IDX['label']]} "
                  f"decay_pos={r[pcol]:.4f} decay_total={r[tcol]:.4f} "
                  f"last1={r[IDX['last1']]} gap_ms={r[IDX['gap_ms']]}")
        assert len(decay_vals) == 1, \
            "CAUSALITY BUG: decay feature should be IDENTICAL across a same-date pair " \
            "regardless of time_ms (decay traversal is date-level, blind to same-date order)"
        # momentum features must NOT all be identical (they resolve the sub-date order that
        # decay is blind to) -- specifically last1 for the 2nd-in-time-order row must reflect
        # the 1st row's label, proving the momentum traversal used time_ms, not just date.
        last1_vals = [flat[i][IDX['last1']] for i in idxs_sorted]
        assert last1_vals[1] == flat[idxs_sorted[0]][IDX['label']], \
            "CAUSALITY BUG: momentum last1 for later-time_ms row should equal earlier row's label"
        print("  -> decay features IDENTICAL across the pair (date-level, correctly blind to "
              "intra-date order); momentum last1 correctly DIFFERS and resolves the true "
              "time_ms order. Two families verified independently correct on the same rows, "
              "no cross-contamination from the join.")
    else:
        print("(no same-user/same-date pair with >=2 distinct time_ms found -- skipping)")

    # ================================================================
    # PART D: decay-aware BPR SAMPLING-WEIGHT spot-check (iter23-specific).
    # compute_final_decayed_pos is a TRAINING-TIME-only quantity (never a
    # per-row feature, so no leakage risk by construction -- see its
    # docstring), but its arithmetic must still match iter22's original
    # brute-force-verified formula: for each user, sum 0.5**(gap_days/hl)
    # over ALL of that user's TRAIN positive rows, gap measured to the END
    # of the train period (not per-row/causal -- a single reference date
    # shared by every user). Spot-check against a brute-force recount for
    # 25 random users (default halflife=3, matching decay_act_3/
    # decay_rate_3's halflife so the sampling weight and the model's input
    # features share the same time constant).
    # ================================================================
    print("\n=== PART D: decay-aware sampling-weight spot-check (brute force) ===")
    train_rows = ext['train']
    decayed_pos_dict = compute_final_decayed_pos(train_rows, halflife=3)
    ref_ord = max(_date_to_ordinal(r[IDX['date']]) for r in train_rows)
    print(f"train period end (reference date ordinal): {ref_ord}  "
          f"({len(decayed_pos_dict)} users with >=1 train positive)")

    by_user_train = collections.defaultdict(list)
    for r in train_rows:
        by_user_train[r[IDX['user_id']]].append(r)

    rng2 = np.random.default_rng(1)
    all_users = sorted(by_user_train.keys())
    sample_users = rng2.choice(all_users, size=min(30, len(all_users)), replace=False)
    max_err_d = 0.0
    n_checked = 0
    for u in sample_users:
        rows_u = by_user_train[u]
        manual = sum(0.5 ** ((ref_ord - _date_to_ordinal(r[IDX['date']])) / 3)
                     for r in rows_u if r[IDX['label']] == 1)
        got = decayed_pos_dict.get(u, 0.0)
        err = abs(manual - got)
        max_err_d = max(max_err_d, err)
        assert err < 1e-6, (f"CAUSALITY/ARITHMETIC BUG: decayed_pos sampling weight mismatch "
                             f"user={u} manual={manual} got={got}")
        n_checked += 1
    print(f"{n_checked} random users: compute_final_decayed_pos matches brute-force recount "
          f"of 0.5**(gap_days/3) over all TRAIN positive rows "
          f"(max abs err {max_err_d:.2e}). No arithmetic error, matches iter22's formula.")

    # zero-positive users must be absent from the dict (never sampled -- consistent
    # with pos_len==0 users who are excluded from `eligible` in build_pos_neg_index)
    zero_pos_users = [u for u, rows_u in by_user_train.items()
                       if all(r[IDX['label']] == 0 for r in rows_u)][:5]
    for u in zero_pos_users:
        assert u not in decayed_pos_dict, \
            f"BUG: zero-train-positive user {u} should be absent from decayed_pos dict!"
    if zero_pos_users:
        print(f"zero-train-positive users ({len(zero_pos_users)} checked): correctly absent "
              "from decayed_pos dict (never contribute sampling weight).")

    # sanity cross-check: for a user with ALL train positives on the reference (last) date,
    # decayed_pos should equal exactly their positive count (gap=0 -> weight 1.0 each).
    last_date = max(r[IDX['date']] for r in train_rows)
    same_day_users = [u for u, rows_u in by_user_train.items()
                       if any(r[IDX['date']] == last_date and r[IDX['label']] == 1 for r in rows_u)
                       and all(r[IDX['date']] == last_date for r in rows_u if r[IDX['label']] == 1)]
    if same_day_users:
        u = same_day_users[0]
        pos_count = sum(1 for r in by_user_train[u] if r[IDX['label']] == 1)
        assert abs(decayed_pos_dict[u] - pos_count) < 1e-9, \
            "BUG: all-positives-on-reference-date user should have decayed_pos == raw pos count"
        print(f"reference-date edge case (user {u}, {pos_count} positives all on last train "
              f"date {last_date}): decayed_pos == raw count exactly ({decayed_pos_dict[u]:.6f}). "
              "PASSED.")

    print("\nAll causal spot-checks (decay + momentum + cross-family joint + sampling-weight) "
          "passed.")
