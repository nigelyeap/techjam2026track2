"""iter35: date-shifted-split robustness check of iter27's winning TRIPLE-
FUSION config (iter24's refined feature set + iter23's decay-aware BPR
user-sampling weight + iter25's Laplace-alpha/n_buckets constants).

This file is a line-for-line copy of `iter27_triple_fusion/data_ext.py`
(feature computation, encode_ext, compute_final_decayed_pos -- ALL
UNCHANGED) with exactly ONE structural change, mirroring how
`iter29_bucket_robustness/data_ext.py` adapted `iter25_retune_v2/data_ext.py`
for the same purpose: `_load_raw_time` / `load_ext` now take an explicit
`splits_map` (a {name: (lo, hi)} dict) instead of hardcoding an import of
`SPLITS` from `../../data.py`. This lets the SAME feature/encoding code run
against EITHER the OFFICIAL split (for the harness-fidelity check against
iter27's own published numbers) OR iter29's exact SHIFTED split (for the
robustness check this iteration exists to run), selected by a caller-passed
`splits_map` + `split_tag` (the latter only used to keep the on-disk pickle
caches from the two splits from colliding).

SPLITS_SHIFTED below is copied VERBATIM from
`iter29_bucket_robustness/data_ext.py` (train 2022-04-05..18 / valid
2022-04-19..25 / test 2022-04-26..05-05, i.e. each window moved 3 days
earlier than the official split) -- see that file's module docstring for
the full derivation/justification of the delta=3 choice. This module does
NOT re-derive or second-guess that choice; it reuses it exactly so the
comparison in RESULT.md is apples-to-apples with iter29's own isolated-
effect numbers.

Does NOT modify data.py, or any iterN/data_ext.py file it reuses/imports
(iter18_momentum's compute_momentum_features, imported unchanged below).

--- original iter27 module docstring follows, describing the fused feature
logic itself, which this file does not modify ---

iter27: TRIPLE fusion of three mutually non-overlapping Round-7 wins that
had never been combined before that iteration:

  - iter24's refined FEATURE SET: decay_rate_H/decay_act_H fine halflife grid
    (HALFLIVES = [2, 2.5, 3, 3.5] days, winner H=2.5) + decayed `tab_pos`
    (TAB_HALFLIVES = [3, 7] days, winner 3d) + iter18/iter19's momentum
    fields (last1/lastk_rate/gap).
  - iter23's BPR training-time user-SAMPLING WEIGHT: `compute_final_decayed_pos`
    (decayed_pos_total[user], evaluated once per user as of the end of
    train, halflife=3d by default) replacing the flat `pos_len[user]`
    weight, used as `decayed_pos_total[user] ** sampling_alpha`. This is a
    TRAINING-TIME-only scalar, never a per-row feature -- no leakage risk
    by construction, and (relevant to THIS iteration) its "end of train"
    reference date automatically follows whichever split's train window is
    in effect, official or shifted.
  - iter25's retuned FORMULA CONSTANTS: the Laplace-smoothing constant
    `alpha` used inside the decay_rate/decay_act/lastk_rate/rate ratio
    formulas (module-level default 1.0, retuned to 0.5), and `n_buckets`,
    the quantile-bucket count used for every bucketed continuous field
    (previously hardcoded 10, retuned to 20).

Row tuple layout (see IDX / _halflife_col / _tab_halflife_col for the
authoritative index map, identical to iter24/iter27's):
  0 date, 1 user_id, 2 video_id, 3 author_id, 4 tab, 5 duration_ms, 6 label,
  7 hourmin, 8 time_ms, 9 orig_idx,
  10 activity, 11 tab_pos, 12 prior_pos, 13 prior_total,        <- flat (iter9)
  14 last1, 15 lastk_sum, 16 lastk_cnt, 17 gap_ms,               <- momentum (iter18)
  18.. decay_pos_h0, decay_total_h0, decay_pos_h1, decay_total_h1, ...  <- decay (iter16/iter20/iter24 fine grid)
       (2*len(HALFLIVES) columns, h order follows HALFLIVES)
  then decay_tab_h0, decay_tab_h1, ...                           <- decayed tab_pos (iter20/iter24)
       (len(TAB_HALFLIVES) columns, h order follows TAB_HALFLIVES)
"""
import os, sys, csv, collections, datetime, importlib.util, pickle
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from data import SPLITS as OFFICIAL_SPLITS  # noqa: E402  (official date ranges, kept in sync w/ data.py)

# iter29's exact shifted split, copied verbatim from
# iter29_bucket_robustness/data_ext.py's SPLITS_SHIFTED.
SPLITS_SHIFTED = {'train': (20220405, 20220418),
                   'valid': (20220419, 20220425),
                   'test':  (20220426, 20220505)}

LABEL = 'long_view'
BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
ALPHA = 1.0
K_DEFAULT = 5

# Axis A grid: fine, bracketing iter16's 3d peak / iter20's 2.5d peak, this
# time to be re-evaluated WITH momentum features present. (iter27, unchanged)
HALFLIVES = [2, 2.5, 3, 3.5]  # days
# Axis B grid: candidate halflives for decayed tab_pos (iter20's grid). (iter27, unchanged)
TAB_HALFLIVES = [3, 7]  # days


def _load_module(name, rel_path):
    path = os.path.join(_THIS_DIR, *rel_path.split('/'))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# iter18's momentum function is imported (not copied) -- same pattern
# iter19/iter24/iter27 used.
_iter18_de = _load_module('iter18_data_ext', '../iter18_momentum/data_ext.py')
compute_momentum_features = _iter18_de.compute_momentum_features


def _date_to_ordinal(d):
    y, m, day = d // 10000, (d // 100) % 100, d % 100
    return datetime.date(y, m, day).toordinal()


def compute_causal_features(rows):
    """Identical to iter9/iter16/iter20/iter24/iter27's compute_causal_features
    -- copied verbatim (flat activity/tab_pos/prior_pos/prior_total)."""
    n = len(rows)
    order = sorted(range(n), key=lambda i: rows[i][0])
    activity = [0] * n
    tab_pos = [0] * n
    prior_pos = [0] * n

    user_total = collections.defaultdict(int)
    user_pos = collections.defaultdict(int)
    user_tab_pos = collections.defaultdict(int)

    i = 0
    while i < n:
        j = i
        d = rows[order[i]][0]
        while j < n and rows[order[j]][0] == d:
            j += 1
        day_idx = order[i:j]
        for idx in day_idx:
            r = rows[idx]
            u, tab = r[1], r[4]
            activity[idx] = user_total[u]
            prior_pos[idx] = user_pos[u]
            tab_pos[idx] = user_tab_pos[(u, tab)]
        for idx in day_idx:
            r = rows[idx]
            u, tab, label = r[1], r[4], r[6]
            user_total[u] += 1
            if label == 1:
                user_pos[u] += 1
                user_tab_pos[(u, tab)] += 1
        i = j
    return {'activity': activity, 'tab_pos': tab_pos, 'prior_pos': prior_pos,
            'prior_total': activity}


def compute_decay_features(rows, halflives=HALFLIVES):
    """Identical mechanism to iter16/iter20/iter27's compute_decay_features --
    copied verbatim (lazy-decay running state per user, exact not
    approximate). Returns (decayed_pos, decayed_total): each an (n, H)
    float64 array."""
    n = len(rows)
    H = len(halflives)
    order = sorted(range(n), key=lambda i: rows[i][0])
    day_mult = [0.5 ** (1.0 / h) for h in halflives]

    decayed_pos = np.zeros((n, H), dtype=np.float64)
    decayed_total = np.zeros((n, H), dtype=np.float64)

    ord_cache = {}
    def ordf(d):
        v = ord_cache.get(d)
        if v is None:
            v = _date_to_ordinal(d)
            ord_cache[d] = v
        return v

    user_last_ord = {}
    user_pos_state = {}
    user_total_state = {}

    i = 0
    while i < n:
        j = i
        d = rows[order[i]][0]
        while j < n and rows[order[j]][0] == d:
            j += 1
        day_idx = order[i:j]
        d_ord = ordf(d)

        for idx in day_idx:
            r = rows[idx]
            u = r[1]
            last = user_last_ord.get(u)
            if last is not None:
                gap = d_ord - last
                pstate = user_pos_state[u]
                tstate = user_total_state[u]
                for h in range(H):
                    f = day_mult[h] ** gap
                    decayed_pos[idx, h] = pstate[h] * f
                    decayed_total[idx, h] = tstate[h] * f

        day_pos_count = collections.defaultdict(int)
        day_total_count = collections.defaultdict(int)
        for idx in day_idx:
            r = rows[idx]
            u, label = r[1], r[6]
            day_total_count[u] += 1
            if label == 1:
                day_pos_count[u] += 1
        touched_users = set(day_total_count.keys())
        for u in touched_users:
            last = user_last_ord.get(u)
            if last is not None:
                gap = d_ord - last
                pstate = user_pos_state[u]
                tstate = user_total_state[u]
                new_p = [pstate[h] * (day_mult[h] ** gap) for h in range(H)]
                new_t = [tstate[h] * (day_mult[h] ** gap) for h in range(H)]
            else:
                new_p = [0.0] * H
                new_t = [0.0] * H
            pc = day_pos_count.get(u, 0)
            tc = day_total_count[u]
            for h in range(H):
                new_p[h] += pc
                new_t[h] += tc
            user_pos_state[u] = new_p
            user_total_state[u] = new_t
            user_last_ord[u] = d_ord
        i = j
    return decayed_pos, decayed_total


def compute_decay_tab_features(rows, halflives=TAB_HALFLIVES):
    """Identical to iter20/iter27's compute_decay_tab_features -- copied
    verbatim. Decayed analogue of flat `tab_pos` (count of user's prior
    POSITIVE rows in the SAME tab), keyed by (user, tab) instead of user.
    Two-phase date-grouped traversal, same causal guarantee as
    compute_decay_features: same-date rows never see each other, no future
    leakage. Returns decayed_tab_pos: (n, H) float64 array, column order
    matching `halflives`."""
    n = len(rows)
    H = len(halflives)
    order = sorted(range(n), key=lambda i: rows[i][0])
    day_mult = [0.5 ** (1.0 / h) for h in halflives]

    decayed_tab_pos = np.zeros((n, H), dtype=np.float64)

    ord_cache = {}
    def ordf(d):
        v = ord_cache.get(d)
        if v is None:
            v = _date_to_ordinal(d)
            ord_cache[d] = v
        return v

    key_last_ord = {}
    key_state = {}

    i = 0
    while i < n:
        j = i
        d = rows[order[i]][0]
        while j < n and rows[order[j]][0] == d:
            j += 1
        day_idx = order[i:j]
        d_ord = ordf(d)

        for idx in day_idx:
            r = rows[idx]
            u, tab = r[1], r[4]
            key = (u, tab)
            last = key_last_ord.get(key)
            if last is not None:
                gap = d_ord - last
                state = key_state[key]
                for h in range(H):
                    decayed_tab_pos[idx, h] = state[h] * (day_mult[h] ** gap)

        day_pos_count = collections.defaultdict(int)
        for idx in day_idx:
            r = rows[idx]
            u, tab, label = r[1], r[4], r[6]
            if label == 1:
                day_pos_count[(u, tab)] += 1
        for key, pc in day_pos_count.items():
            last = key_last_ord.get(key)
            if last is not None:
                gap = d_ord - last
                state = key_state[key]
                new_s = [state[h] * (day_mult[h] ** gap) for h in range(H)]
            else:
                new_s = [0.0] * H
            for h in range(H):
                new_s[h] += pc
            key_state[key] = new_s
            key_last_ord[key] = d_ord
        i = j
    return decayed_tab_pos


def _load_raw_time(data_dir, splits_map):
    """Adapted from iter18/iter19/iter24/iter27's _load_raw_time (same
    files, same row order, same vid2author join, same orig_idx assignment):
    the ONLY change from iter27's version is that the split boundaries are
    now an explicit `splits_map` argument instead of the module-level
    `SPLITS` imported from ../../data.py -- this is what lets the SAME
    function serve both the official split (harness-fidelity check) and
    iter29's shifted split (the robustness check), selected by the caller."""
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
    for name, (lo, hi) in splits_map.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


IDX = dict(date=0, user_id=1, video_id=2, author_id=3, tab=4, duration_ms=5, label=6,
           hourmin=7, time_ms=8, orig_idx=9,
           activity=10, tab_pos=11, prior_pos=12, prior_total=13,
           last1=14, lastk_sum=15, lastk_cnt=16, gap_ms=17)
DECAY_BASE = 18  # decay_pos_h0 starts here


def _halflife_col(h, halflives=HALFLIVES):
    pos = halflives.index(h)
    return DECAY_BASE + 2 * pos, DECAY_BASE + 2 * pos + 1


def _tab_halflife_col(h, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES):
    base = DECAY_BASE + 2 * len(halflives)
    return base + tab_halflives.index(h)


_CACHE_VERSION = 1


def _cache_path(halflives, tab_halflives, split_tag):
    key = '-'.join(str(h) for h in halflives) + '__tab_' + '-'.join(str(h) for h in tab_halflives)
    return os.path.join(_THIS_DIR, f'.cache_v{_CACHE_VERSION}_{split_tag}_{key}.pkl')


def load_ext(data_dir, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES, K=K_DEFAULT, use_cache=True,
             splits_map=None, split_tag='official'):
    """Returns dict split -> list of extended rows (see IDX / DECAY_BASE /
    _tab_halflife_col for layout). Runs FOUR independent causal traversals
    over the same flat (train+valid+test, in that order) row list and joins
    their outputs by row index -- IDENTICAL mechanism to iter27's load_ext.

    `splits_map`: {name: (lo, hi)} date-boundary dict. Defaults to
    OFFICIAL_SPLITS (data.py's official split) if not given. Pass
    SPLITS_SHIFTED for iter29's shifted-split robustness check.
    `split_tag`: only used to namespace the on-disk pickle cache so the
    official-split and shifted-split extended-feature caches never collide
    or get confused with each other (or with iter27's own cache files,
    which live in a different directory anyway).
    """
    splits_map = splits_map if splits_map is not None else OFFICIAL_SPLITS
    cpath = _cache_path(halflives, tab_halflives, split_tag)
    if use_cache and os.path.exists(cpath):
        with open(cpath, 'rb') as fh:
            return pickle.load(fh)

    splits = _load_raw_time(data_dir, splits_map)
    order = ('train', 'valid', 'test')
    flat, owner = [], []
    for name in order:
        for r in splits[name]:
            flat.append(r)
            owner.append(name)

    day_feats = compute_causal_features(flat)
    mom_feats = compute_momentum_features(flat, K=K)
    decayed_pos, decayed_total = compute_decay_features(flat, halflives)
    decayed_tab_pos = compute_decay_tab_features(flat, tab_halflives)

    ext = {name: [] for name in order}
    for i, (r, name) in enumerate(zip(flat, owner)):
        extra = [day_feats['activity'][i], day_feats['tab_pos'][i],
                 day_feats['prior_pos'][i], day_feats['prior_total'][i],
                 mom_feats['last1'][i], mom_feats['lastk_sum'][i],
                 mom_feats['lastk_cnt'][i], mom_feats['gap_ms'][i]]
        for h in range(len(halflives)):
            extra.append(decayed_pos[i, h])
            extra.append(decayed_total[i, h])
        for h in range(len(tab_halflives)):
            extra.append(decayed_tab_pos[i, h])
        ext[name].append(r + tuple(extra))

    if use_cache:
        with open(cpath, 'wb') as fh:
            pickle.dump(ext, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return ext


def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1])


def encode_ext(splits, feature_set=('decay_rate_3', 'decay_act_3', 'tab', 'last1', 'lastk_rate', 'gap'),
               halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES, alpha=ALPHA, n_buckets=10):
    """IDENTICAL to iter27's encode_ext -- copied verbatim, no split-related
    change needed here (this function only ever sees whatever `splits` dict
    it's handed by load_ext, official or shifted -- it has no awareness of
    date boundaries itself)."""
    tr = splits['train']
    dur_edges = _bucket_edges([x[IDX['duration_ms']] for x in tr], n=n_buckets)

    def raw_dur(x):
        return str(int(np.searchsorted(dur_edges, x[IDX['duration_ms']])))

    def parse_feat(name):
        if name in ('activity', 'tab', 'rate', 'last1', 'lastk_rate', 'gap'):
            return name, None
        if name.startswith('decay_rate_'):
            return 'decay_rate', float(name.rsplit('_', 1)[1])
        if name.startswith('decay_act_'):
            return 'decay_act', float(name.rsplit('_', 1)[1])
        if name.startswith('decay_tab_'):
            return 'decay_tab', float(name.rsplit('_', 1)[1])
        raise ValueError(name)

    extra_edges = {}
    for name in feature_set:
        kind, h = parse_feat(name)
        if kind == 'activity':
            extra_edges[name] = _bucket_edges([x[IDX['activity']] for x in tr], n=n_buckets)
        elif kind == 'tab':
            extra_edges[name] = _bucket_edges([x[IDX['tab_pos']] for x in tr], n=n_buckets)
        elif kind == 'rate':
            extra_edges[name] = _bucket_edges(
                [(x[IDX['prior_pos']] + alpha) / (x[IDX['prior_total']] + 2 * alpha) for x in tr], n=n_buckets)
        elif kind == 'decay_rate':
            pcol, tcol = _halflife_col(h, halflives)
            extra_edges[name] = _bucket_edges(
                [(x[pcol] + alpha) / (x[tcol] + 2 * alpha) for x in tr], n=n_buckets)
        elif kind == 'decay_act':
            pcol, tcol = _halflife_col(h, halflives)
            extra_edges[name] = _bucket_edges([x[tcol] for x in tr], n=n_buckets)
        elif kind == 'decay_tab':
            tcol = _tab_halflife_col(h, halflives, tab_halflives)
            extra_edges[name] = _bucket_edges([x[tcol] for x in tr], n=n_buckets)
        elif kind == 'lastk_rate':
            extra_edges[name] = _bucket_edges(
                [(x[IDX['lastk_sum']] + alpha) / (x[IDX['lastk_cnt']] + 2 * alpha) for x in tr], n=n_buckets)
        elif kind == 'gap':
            gaps = [x[IDX['gap_ms']] for x in tr if x[IDX['gap_ms']] >= 0]
            extra_edges[name] = _bucket_edges(gaps, n=n_buckets)
        # 'last1' needs no edges (raw categorical)

    def extra_val(x, name):
        kind, h = parse_feat(name)
        if kind == 'activity':
            return str(int(np.searchsorted(extra_edges[name], x[IDX['activity']])))
        elif kind == 'tab':
            return str(int(np.searchsorted(extra_edges[name], x[IDX['tab_pos']])))
        elif kind == 'rate':
            r = (x[IDX['prior_pos']] + alpha) / (x[IDX['prior_total']] + 2 * alpha)
            return str(int(np.searchsorted(extra_edges[name], r)))
        elif kind == 'decay_rate':
            pcol, tcol = _halflife_col(h, halflives)
            r = (x[pcol] + alpha) / (x[tcol] + 2 * alpha)
            return str(int(np.searchsorted(extra_edges[name], r)))
        elif kind == 'decay_act':
            pcol, tcol = _halflife_col(h, halflives)
            return str(int(np.searchsorted(extra_edges[name], x[tcol])))
        elif kind == 'decay_tab':
            tcol = _tab_halflife_col(h, halflives, tab_halflives)
            return str(int(np.searchsorted(extra_edges[name], x[tcol])))
        elif kind == 'last1':
            v = x[IDX['last1']]
            return 'UNK' if v == -1 else str(int(v))
        elif kind == 'lastk_rate':
            r = (x[IDX['lastk_sum']] + alpha) / (x[IDX['lastk_cnt']] + 2 * alpha)
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
    """Copied verbatim from iter27_triple_fusion/data_ext.py's
    compute_final_decayed_pos. NON-CAUSAL, single scalar per user -- the
    recency-decayed count of that user's TRAIN positive rows, decayed to a
    single fixed reference date = the END of the train period (max date
    PRESENT IN TRAIN). Note this reference date automatically tracks
    whichever train window is active (official train ends 2022-04-21;
    shifted train ends 2022-04-18) since it's derived from `train_rows`
    itself, not hardcoded -- exactly the behavior needed for this
    iteration's shifted-split run to be a faithful analog of iter27's
    official-split sampling weight, not an accidentally-stale one.

    train_rows: list of row tuples with date at index 0, user_id at index 1,
    label at index 6.

    Returns: dict user_id -> decayed positive count (float).
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
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(_THIS_DIR, '..', '..',
                                                         'KuaiRand-Pure', 'data'))
    ap.add_argument('--shifted', action='store_true', help='use iter29 shifted split instead of official')
    ap.add_argument('--no_cache', action='store_true')
    a = ap.parse_args()
    smap = SPLITS_SHIFTED if a.shifted else OFFICIAL_SPLITS
    stag = 'shifted' if a.shifted else 'official'
    print(f"loading {a.data_dir} ... split={stag} {smap}")
    t0 = time.time()
    ext = load_ext(a.data_dir, use_cache=not a.no_cache, splits_map=smap, split_tag=stag)
    print({k: len(v) for k, v in ext.items()}, f"  ({time.time()-t0:.1f}s)")

    # Row-count / date-boundary sanity check against iter29's reported numbers
    # (only meaningful with --shifted; see RESULT.md for the cross-check).
    if a.shifted:
        expect = {'train': 1079797, 'valid': 143394, 'test': 170150}
        got = {k: len(v) for k, v in ext.items()}
        print(f"expected (iter29): {expect}")
        print(f"got:                {got}")
        assert got == expect, "SHIFTED-SPLIT ROW COUNT MISMATCH vs iter29's reported sizes!"
        print("Row counts MATCH iter29's reported shifted-split sizes exactly.")

    flat = ext['train'] + ext['valid'] + ext['test']
    n = len(flat)

    print("\n=== quick decay-feature causal spot-check (brute force, 15 rows) ===")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=15, replace=False)
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
            assert err_p < 1e-6 and err_t < 1e-6, f"CAUSALITY BUG h={h} idx={idx}"
    print(f"15 random rows x {len(HALFLIVES)} halflives: match brute force (max abs err {max_err:.2e}). "
          f"Split={stag}. No leakage detected (full PART A-E suite already verified in "
          f"iter27_triple_fusion/data_ext.py against this same, unmodified feature code).")
