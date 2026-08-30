"""iter18: fine-grained timestamp-level causal MOMENTUM features, stacked on
top of iter9's date-grouped causal history features (activity/tab_pos/rate).

iter9's traversal only resolves ordering at DAY granularity (strict `<` on
`date`) -- two rows on the same date are treated as simultaneous even if they
happened hours apart. This module instead uses the raw log's `time_ms`
column (a genuine millisecond epoch timestamp) to build a strict per-user
CHRONOLOGICAL total order, and computes short-term "session momentum"
features from it:

  - `last1`     : was the user's IMMEDIATELY PRECEDING row (by time_ms) a
                  long_view? Categorical: '0' / '1' / 'UNK' (UNK = this is
                  the user's first row in the combined timeline).
  - `lastk_rate`: Laplace-smoothed positive rate over the user's last K
                  (default 5) rows strictly before this one:
                  (sum(last K labels) + ALPHA) / (min(K, available) + 2*ALPHA)
                  Always well-defined (falls back to the 0.5 neutral prior
                  when 0 rows are available), bucketed like iter9's `rate`.
  - `gap`       : bucketed time gap (ms) since the user's immediately
                  preceding row -- pure recency-of-engagement signal,
                  independent of what that prior row's label was. 'UNK' for
                  a user's first row (no prior gap exists); bucket edges are
                  quantile-fit on TRAIN rows with a defined gap only.

This is fundamentally different information from iter9's `rate` (a long-run
aggregate over the user's ENTIRE prior history): `last1`/`lastk_rate` are
short-term, session-scale signals reflecting the user's current mood/session
state.

Causal-correctness pattern (stricter version of iter9's date-grouped one):
  - combine train+valid+test raw rows into one flat "global timeline" (same
    as iter9)
  - for each user, sort ALL of that user's rows (across every split) by
    `time_ms` ascending, breaking exact time_ms ties with a STABLE
    original-row-order index (`orig_idx` = position in the raw CSV read
    order, file1 then file2, BEFORE any date-range split filtering -- so the
    tiebreak is fixed regardless of which split a row lands in)
  - walk that per-user sequence one row at a time; a row's momentum features
    are read from state accumulated over rows STRICTLY EARLIER in this total
    order, and only AFTER reading are they folded into the state (so a row
    can never see its own label, and ties can never leak either direction)

Does NOT modify ../../data.py or ../iter9_history_dense/data_ext.py --
reuses iter9's `compute_causal_features` (date-grouped activity/tab_pos/rate)
unmodified for feature-parity re-derivation, and reuses ../../data.py's
`SPLITS` dict for the date ranges (kept in sync, not hand-copied).
"""
import os, sys, csv, collections, importlib.util
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from data import SPLITS  # noqa: E402  (date ranges only, kept in sync w/ data.py)

# iter9's module is also named data_ext.py -- load it under a distinct module
# name (instead of `sys.path.insert` + `import data_ext`) so it never collides
# with / shadows this file's own module identity.
_iter9_path = os.path.join(_THIS_DIR, '..', 'iter9_history_dense', 'data_ext.py')
_spec = importlib.util.spec_from_file_location('iter9_data_ext', _iter9_path)
_iter9_data_ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_iter9_data_ext)
compute_causal_features = _iter9_data_ext.compute_causal_features  # noqa: E402  (iter9's date-grouped features, unmodified)

LABEL = 'long_view'
BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
ALPHA = 1.0     # Laplace smoothing constant, matches iter9's `rate`
K_DEFAULT = 5   # window size for lastk_rate

# extended-tuple column layout (see load_ext docstring)
IDX = dict(date=0, user_id=1, video_id=2, author_id=3, tab=4, duration_ms=5, label=6,
           hourmin=7, time_ms=8, orig_idx=9,
           activity=10, tab_pos=11, prior_pos=12, prior_total=13,
           last1=14, lastk_sum=15, lastk_cnt=16, gap_ms=17)


def _load_raw_time(data_dir):
    """Mirrors data.py's load() exactly (same files, same row order, same
    vid2author join, same date-range filtering) but keeps `hourmin` and
    `time_ms` per row, and assigns `orig_idx` = position in the flat
    file1-then-file2 read order (BEFORE date filtering) as a stable,
    split-independent tiebreak key.

    Returns dict split -> list of
      (date, user_id, video_id, author_id, tab, duration_ms, label,
       hourmin, time_ms, orig_idx)
    """
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


def compute_momentum_features(rows, K=K_DEFAULT):
    """rows: list of extended tuples (must have time_ms at index 8, orig_idx
    at index 9, user_id at index 1, label at index 6), in ANY order.
    Returns dict of lists aligned index-for-index with `rows`:
      'last1'      : -1 (UNK, user's first row) or 0/1 (prior row's label)
      'lastk_sum'  : sum of labels over up to K strictly-prior rows
      'lastk_cnt'  : count of strictly-prior rows considered (<=K)
      'gap_ms'     : -1 (UNK, first row) or time_ms - previous row's time_ms

    STRICT causality by construction: for each user we sort indices by
    (time_ms, orig_idx) to get one unambiguous total order, then walk it
    once; state is only updated with a row's own label AFTER that row's
    features have already been read. A row therefore can never see its own
    label or any later row's label -- not even in a same-time_ms tie, since
    the tiebreak (orig_idx) makes the order strict and total.
    """
    n = len(rows)
    last1 = [-1] * n
    lastk_sum = [0] * n
    lastk_cnt = [0] * n
    gap_ms = [-1] * n

    by_user = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_user[r[1]].append(i)

    for u, idxs in by_user.items():
        idxs.sort(key=lambda i: (rows[i][8], rows[i][9]))  # (time_ms, orig_idx)
        window = collections.deque(maxlen=K)
        prev_time = None
        for i in idxs:
            # 1) READ state as it stood strictly before this row
            last1[i] = window[-1] if window else -1
            lastk_sum[i] = sum(window)
            lastk_cnt[i] = len(window)
            gap_ms[i] = (rows[i][8] - prev_time) if prev_time is not None else -1
            # 2) only now fold this row's own label into the state
            window.append(rows[i][6])
            prev_time = rows[i][8]

    return {'last1': last1, 'lastk_sum': lastk_sum, 'lastk_cnt': lastk_cnt, 'gap_ms': gap_ms}


def load_ext(data_dir, K=K_DEFAULT):
    """Returns dict split -> list of extended rows (18-tuple):
      (date, user_id, video_id, author_id, tab, duration_ms, label,
       hourmin, time_ms, orig_idx,
       activity, tab_pos, prior_pos, prior_total,
       last1, lastk_sum, lastk_cnt, gap_ms)
    See IDX for the column-name -> index mapping.
    """
    splits = _load_raw_time(data_dir)
    order = ('train', 'valid', 'test')
    flat, owner = [], []
    for name in order:
        for r in splits[name]:
            flat.append(r)
            owner.append(name)

    day_feats = compute_causal_features(flat)     # iter9: date-grouped (indices 0,1,4,6 only)
    mom_feats = compute_momentum_features(flat, K=K)  # this iteration: time_ms-total-order

    ext = {name: [] for name in order}
    for i, (r, name) in enumerate(zip(flat, owner)):
        ext[name].append(r + (
            day_feats['activity'][i], day_feats['tab_pos'][i],
            day_feats['prior_pos'][i], day_feats['prior_total'][i],
            mom_feats['last1'][i], mom_feats['lastk_sum'][i],
            mom_feats['lastk_cnt'][i], mom_feats['gap_ms'][i]))
    return ext


def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1])


def encode_ext(splits, feature_set=('activity', 'tab', 'rate')):
    """splits: dict from load_ext(), each row an 18-tuple (see IDX).
    feature_set: subset/order of {'activity','tab','rate','last1','lastk_rate','gap'}.
      - 'activity','tab','rate' : iter9's date-grouped features, re-derived
        here for parity checking (identical semantics/bucketing to iter9).
      - 'last1'     : categorical, raw values '0'/'1'/'UNK' (no bucketing --
        already low-cardinality).
      - 'lastk_rate': continuous, Laplace-smoothed, train-fit quantile
        bucketed (10 buckets), same pattern as 'rate'.
      - 'gap'       : continuous (ms), train-fit quantile bucketed (10
        buckets) using only TRAIN rows with a defined gap (gap_ms >= 0);
        rows with gap_ms == -1 (user's first row) get a dedicated 'UNK'
        category instead of falling into bucket 0.
    Returns (enc, field_dims_sum) with enc[name] = (X, y, users).
    """
    tr = splits['train']
    dur_edges = _bucket_edges([x[IDX['duration_ms']] for x in tr])

    def raw_dur(x):
        return str(int(np.searchsorted(dur_edges, x[IDX['duration_ms']])))

    extra_edges = {}
    if 'activity' in feature_set:
        extra_edges['activity'] = _bucket_edges([x[IDX['activity']] for x in tr])
    if 'tab' in feature_set:
        extra_edges['tab'] = _bucket_edges([x[IDX['tab_pos']] for x in tr])
    if 'rate' in feature_set:
        def rate_of(x):
            return (x[IDX['prior_pos']] + ALPHA) / (x[IDX['prior_total']] + 2 * ALPHA)
        extra_edges['rate'] = _bucket_edges([rate_of(x) for x in tr])
    if 'lastk_rate' in feature_set:
        def lastk_rate_of(x):
            return (x[IDX['lastk_sum']] + ALPHA) / (x[IDX['lastk_cnt']] + 2 * ALPHA)
        extra_edges['lastk_rate'] = _bucket_edges([lastk_rate_of(x) for x in tr])
    if 'gap' in feature_set:
        gaps = [x[IDX['gap_ms']] for x in tr if x[IDX['gap_ms']] >= 0]
        extra_edges['gap'] = _bucket_edges(gaps)

    def extra_val(x, name):
        if name == 'activity':
            return str(int(np.searchsorted(extra_edges['activity'], x[IDX['activity']])))
        elif name == 'tab':
            return str(int(np.searchsorted(extra_edges['tab'], x[IDX['tab_pos']])))
        elif name == 'rate':
            r = (x[IDX['prior_pos']] + ALPHA) / (x[IDX['prior_total']] + 2 * ALPHA)
            return str(int(np.searchsorted(extra_edges['rate'], r)))
        elif name == 'last1':
            v = x[IDX['last1']]
            return 'UNK' if v == -1 else str(int(v))
        elif name == 'lastk_rate':
            r = (x[IDX['lastk_sum']] + ALPHA) / (x[IDX['lastk_cnt']] + 2 * ALPHA)
            return str(int(np.searchsorted(extra_edges['lastk_rate'], r)))
        elif name == 'gap':
            g = x[IDX['gap_ms']]
            if g < 0:
                return 'UNK'
            return str(int(np.searchsorted(extra_edges['gap'], g)))
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


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(_THIS_DIR, '..', '..',
                                                         'KuaiRand-Pure', 'data'))
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    ext = load_ext(a.data_dir)
    print({k: len(v) for k, v in ext.items()})

    flat = ext['train'] + ext['valid'] + ext['test']
    n = len(flat)
    last1 = np.array([r[IDX['last1']] for r in flat])
    lastk_cnt = np.array([r[IDX['lastk_cnt']] for r in flat])
    gap_ms = np.array([r[IDX['gap_ms']] for r in flat])

    print(f"\ntotal rows: {n}")
    print(f"last1 coverage (not user's first row): {np.mean(last1 >= 0)*100:.2f}%  "
          f"({int(np.sum(last1 >= 0))}/{n})")
    print(f"  of covered rows: last1==1 (prior long_view) rate = "
          f"{np.mean(last1[last1 >= 0] == 1)*100:.2f}%")
    print(f"lastk_cnt distribution: "
          f"{ {int(c): int(np.sum(lastk_cnt == c)) for c in sorted(set(lastk_cnt.tolist()))} }")
    print(f"gap coverage (not user's first row): {np.mean(gap_ms >= 0)*100:.2f}%  "
          f"({int(np.sum(gap_ms >= 0))}/{n})")
    if np.any(gap_ms >= 0):
        g = gap_ms[gap_ms >= 0] / 1000.0
        print(f"  gap(s) stats: median={np.median(g):.1f}s p25={np.percentile(g,25):.1f}s "
              f"p75={np.percentile(g,75):.1f}s max={np.max(g):.1f}s")

    # ---- causal spot-checks: brute-force recount against real per-user chronological sequences ----
    print("\n--- causal spot-checks: full chronological sequences for 3 real users ---")
    by_user = collections.defaultdict(list)
    for idx, r in enumerate(flat):
        by_user[r[IDX['user_id']]].append(idx)
    # pick users with a decent number of rows so the window/gap logic is exercised
    candidate_users = [u for u, idxs in by_user.items() if 8 <= len(idxs) <= 14][:3]

    def manual_check(u):
        idxs = sorted(by_user[u], key=lambda i: (flat[i][IDX['time_ms']], flat[i][IDX['orig_idx']]))
        print(f"\nuser={u}  ({len(idxs)} rows, chronological order)")
        window = []
        prev_t = None
        for pos, i in enumerate(idxs):
            r = flat[i]
            t = r[IDX['time_ms']]; label = r[IDX['label']]
            manual_last1 = window[-1] if window else -1
            manual_lastk_sum = sum(window[-5:])
            manual_lastk_cnt = len(window[-5:])
            manual_gap = (t - prev_t) if prev_t is not None else -1
            got_last1 = r[IDX['last1']]
            got_lastk_sum = r[IDX['lastk_sum']]
            got_lastk_cnt = r[IDX['lastk_cnt']]
            got_gap = r[IDX['gap_ms']]
            ok = (manual_last1 == got_last1 and manual_lastk_sum == got_lastk_sum
                  and manual_lastk_cnt == got_lastk_cnt and manual_gap == got_gap)
            flag = "OK" if ok else "MISMATCH!!"
            print(f"  pos={pos:2d} date={r[IDX['date']]} time_ms={t} label={label} "
                  f"| last1: got={got_last1} manual={manual_last1} "
                  f"| lastk_sum: got={got_lastk_sum} manual={manual_lastk_sum} "
                  f"| lastk_cnt: got={got_lastk_cnt} manual={manual_lastk_cnt} "
                  f"| gap_ms: got={got_gap} manual={manual_gap}  [{flag}]")
            assert ok, f"CAUSALITY BUG for user {u} at pos {pos}!"
            window.append(label)
            prev_t = t

    for u in candidate_users:
        manual_check(u)

    # explicit same-time_ms tie stress test (synthetic, since real ties are ~absent) ------------
    print("\n--- synthetic same-time_ms tie stress test ---")
    fake_rows = [
        # (date, user_id, video_id, author_id, tab, duration_ms, label, hourmin, time_ms, orig_idx)
        (20220410, 'TIEUSER', 'v1', 'a1', '0', 1000.0, 1, 1000, 5000, 100),
        (20220410, 'TIEUSER', 'v2', 'a1', '0', 1000.0, 0, 1000, 5000, 101),  # exact tie on time_ms
        (20220410, 'TIEUSER', 'v3', 'a1', '0', 1000.0, 1, 1005, 5500, 99),   # later time but SMALLER orig_idx
    ]
    feats = compute_momentum_features(fake_rows, K=5)
    for i, r in enumerate(fake_rows):
        print(f"  row {i}: video={r[2]} label={r[6]} time_ms={r[8]} orig_idx={r[9]} "
              f"-> last1={feats['last1'][i]} lastk_sum={feats['lastk_sum'][i]} "
              f"lastk_cnt={feats['lastk_cnt'][i]} gap_ms={feats['gap_ms'][i]}")
    # expected total order by (time_ms, orig_idx): row0 (t=5000,idx100), row1 (t=5000,idx101),
    # row2 (t=5500,idx99) -- orig_idx breaks the exact tie between row0/row1 deterministically,
    # and row2 (later time_ms) must come after BOTH regardless of its smaller orig_idx.
    assert feats['last1'][0] == -1        # first in order -> UNK
    assert feats['last1'][1] == 1         # sees only row0's label (1)
    assert feats['last1'][2] == 0         # sees only row1's label (0), NOT row0's (no leakage across the tie)
    assert feats['gap_ms'][0] == -1
    assert feats['gap_ms'][1] == 0        # exact tie -> 0ms gap
    assert feats['gap_ms'][2] == 500      # 5500-5000
    print("tie stress test: all assertions passed (row2's time_ms=5500 is causally after "
          "the tie pair despite its smaller orig_idx -- orig_idx only breaks EXACT time_ms ties, "
          "it never overrides real chronological order).")

    print("\nAll causal spot-checks passed. No same-timestamp or future leakage detected.")
