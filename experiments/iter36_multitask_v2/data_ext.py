"""iter27: TRIPLE fusion of three mutually non-overlapping Round-7 wins that
had never been combined before this iteration:

  - iter24's refined FEATURE SET: decay_rate_H/decay_act_H fine halflife grid
    (HALFLIVES = [2, 2.5, 3, 3.5] days, winner H=2.5) + decayed `tab_pos`
    (TAB_HALFLIVES = [3, 7] days, winner 3d) + iter18/iter19's momentum
    fields (last1/lastk_rate/gap). Feature computation code below is COPIED
    VERBATIM from iter24_decay_tab_refine/data_ext.py (which itself copied
    it verbatim from iter20/iter9/iter16/iter18) -- unmodified.
  - iter23's BPR training-time user-SAMPLING WEIGHT: `compute_final_decayed_pos`
    (decayed_pos_total[user], evaluated once per user as of the end of
    train, halflife=3d by default) replacing the flat `pos_len[user]`
    weight, used as `decayed_pos_total[user] ** sampling_alpha`. Copied
    verbatim from iter23_fused_decay_sampling/data_ext.py (itself copied
    verbatim from iter22). This is a TRAINING-TIME-only scalar, never a
    per-row feature -- no leakage risk by construction.
  - iter25's retuned FORMULA CONSTANTS: the Laplace-smoothing constant
    `alpha` used inside the decay_rate/decay_act/lastk_rate/rate ratio
    formulas (module-level default 1.0, retuned to 0.5), and `n_buckets`,
    the quantile-bucket count used for every bucketed continuous field
    (previously hardcoded 10, retuned to 20). Threaded through `encode_ext`
    exactly as iter25 did to iter19's data_ext.py, extended here to also
    cover the new `decay_tab` feature kind (iter25 predates decay_tab, so
    it never had to bucket it).

IMPORTANT NAMING NOTE (per dispatch instructions): there are TWO distinct
"alpha" parameters in this fusion and they must not be conflated:
  - `alpha` (this module's ALPHA / encode_ext's `alpha` kwarg) = the
    Laplace-SMOOTHING constant inside the decay/rate FEATURE formulas
    (iter25's axis). Starting value in this iteration: 0.5.
  - `sampling_alpha` (train.py's run_bpr_ext kwarg, NOT defined in this
    file) = the BPR user-SAMPLING weight exponent applied to
    decayed_pos_total[user] (iter23's axis). Starting value in this
    iteration: 0.5 (iter23's own winning value), but re-swept briefly on
    the fused feature set since the feature set changed since iter23's own
    sweep.
These are independent axes; the fact that both start at 0.5 in this
iteration's default config is a coincidence of what each individual
predecessor found best, not a shared parameter.

Architecture (mirrors iter19/iter24's own design choice, explicitly
required by the dispatch prompt): FOUR independent causal traversals over
the same flat per-row data, joined onto the same rows by row index -- NOT
merged into one traversal. This keeps each family (decay, decayed-tab,
momentum) provably free of cross-contamination from the others (verified in
__main__ PART D below), plus a fifth NON-CAUSAL once-per-user aggregate
(`compute_final_decayed_pos`) that only affects training-time sampling
frequency, never a feature.

  1. compute_causal_features   -- iter9/iter16/iter20/iter24's flat
                                   date-grouped activity/tab_pos/prior_pos/
                                   prior_total. Copied verbatim from iter24's
                                   data_ext.py.
  2. compute_decay_features    -- iter16/iter20/iter24's exponential-decay
                                   rate/act. Copied verbatim from iter24's
                                   data_ext.py (fine HALFLIVES grid).
  3. compute_decay_tab_features -- iter20/iter24's decayed tab_pos. Copied
                                   verbatim from iter24's data_ext.py.
  4. compute_momentum_features -- iter18's time_ms-level last1/lastk/gap.
                                   Imported (not copied) via importlib from
                                   iter18_momentum/data_ext.py, exactly as
                                   iter19/iter24 did -- needs user_id@1,
                                   label@6, time_ms@8, orig_idx@9, all
                                   present in this module's row tuple
                                   (iter19/iter24's 10-col loader, reused
                                   verbatim below).
  5. compute_final_decayed_pos -- iter22/iter23's non-causal per-user BPR
                                   sampling-weight scalar. Copied verbatim
                                   from iter23_fused_decay_sampling/data_ext.py.

Row tuple layout (see IDX / _halflife_col / _tab_halflife_col for the
authoritative index map, identical to iter24's):
  0 date, 1 user_id, 2 video_id, 3 author_id, 4 tab, 5 duration_ms, 6 label,
  7 hourmin, 8 time_ms, 9 orig_idx,
  10 activity, 11 tab_pos, 12 prior_pos, 13 prior_total,        <- flat (iter9)
  14 last1, 15 lastk_sum, 16 lastk_cnt, 17 gap_ms,               <- momentum (iter18)
  18.. decay_pos_h0, decay_total_h0, decay_pos_h1, decay_total_h1, ...  <- decay (iter16/iter20/iter24 fine grid)
       (2*len(HALFLIVES) columns, h order follows HALFLIVES)
  then decay_tab_h0, decay_tab_h1, ...                           <- decayed tab_pos (iter20/iter24)
       (len(TAB_HALFLIVES) columns, h order follows TAB_HALFLIVES)

Does NOT modify data.py, or any iterN/data_ext.py file it reuses/imports.
"""
import os, sys, csv, collections, datetime, importlib.util, pickle
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from data import SPLITS  # noqa: E402  (date ranges only, kept in sync w/ data.py)

LABEL = 'long_view'
BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
ALPHA = 1.0
K_DEFAULT = 5

# Axis A grid: fine, bracketing iter16's 3d peak / iter20's 2.5d peak, this
# time to be re-evaluated WITH momentum features present.
HALFLIVES = [2, 2.5, 3, 3.5]  # days
# Axis B grid: candidate halflives for decayed tab_pos (iter20's grid).
TAB_HALFLIVES = [3, 7]  # days


def _load_module(name, rel_path):
    path = os.path.join(_THIS_DIR, *rel_path.split('/'))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# iter18's momentum function is imported (not copied) -- same pattern iter19
# used. Loaded under a distinct module name since every iterN dir has its own
# data_ext.py (a plain `import data_ext` would collide).
_iter18_de = _load_module('iter18_data_ext', '../iter18_momentum/data_ext.py')
compute_momentum_features = _iter18_de.compute_momentum_features


def _date_to_ordinal(d):
    y, m, day = d // 10000, (d // 100) % 100, d % 100
    return datetime.date(y, m, day).toordinal()


def compute_causal_features(rows):
    """Identical to iter9/iter16/iter20's compute_causal_features -- copied
    verbatim (flat activity/tab_pos/prior_pos/prior_total)."""
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
    """Identical mechanism to iter16/iter20's compute_decay_features --
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
    """Identical to iter20's compute_decay_tab_features -- copied verbatim.
    Decayed analogue of flat `tab_pos` (count of user's prior POSITIVE rows
    in the SAME tab), keyed by (user, tab) instead of user. Two-phase
    date-grouped traversal, same causal guarantee as compute_decay_features:
    same-date rows never see each other, no future leakage.
    Returns decayed_tab_pos: (n, H) float64 array, column order matching
    `halflives`."""
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


def _load_raw_time(data_dir):
    """Verbatim copy of iter18/iter19's _load_raw_time (same files, same row
    order, same vid2author join, same date-range filtering, same orig_idx
    assignment) -- needed because momentum requires time_ms/orig_idx which
    data.py's plain load() doesn't expose."""
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


def _cache_path(halflives, tab_halflives):
    key = '-'.join(str(h) for h in halflives) + '__tab_' + '-'.join(str(h) for h in tab_halflives)
    return os.path.join(_THIS_DIR, f'.cache_v{_CACHE_VERSION}_{key}.pkl')


def load_ext(data_dir, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES, K=K_DEFAULT, use_cache=True):
    """Returns dict split -> list of extended rows (see IDX / DECAY_BASE /
    _tab_halflife_col for layout). Runs FOUR independent causal traversals
    over the same flat (train+valid+test, in that order) row list and joins
    their outputs by row index:
      1. compute_causal_features    (date-grouped flat activity/tab_pos/rate)
      2. compute_momentum_features  (time_ms-level last1/lastk/gap, iter18)
      3. compute_decay_features     (date-grouped exponential decay, fine grid)
      4. compute_decay_tab_features (date-grouped exponential decay of tab_pos)
    Each traversal only reads columns it documents needing and is causally
    self-contained -- combining them is a pure join, not a shared mutable
    pass, so no cross-family leakage is possible by construction.
    """
    cpath = _cache_path(halflives, tab_halflives)
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
    """splits: dict from load_ext(). feature_set: subset/order of:
      'activity','tab','rate'         flat features (bucketed)
      'decay_rate_H','decay_act_H'    decayed rate/act, H in halflives (bucketed)
      'decay_tab_H'                   decayed tab_pos, H in tab_halflives (bucketed)
      'last1'                         categorical ('0'/'1'/'UNK')
      'lastk_rate'                    continuous, bucketed
      'gap'                           continuous (ms), bucketed, 'UNK' for first row
    `alpha` (iter25 axis): Laplace-smoothing constant used in the
    rate/decay_rate/lastk_rate ratio formulas -- NOT the BPR sampling-weight
    exponent (that lives in train.py as `sampling_alpha`, a wholly separate
    axis; see module docstring's naming note).
    `n_buckets` (iter25 axis): quantile bucket count for ALL bucketed
    continuous features, including dur_bucket, decay_act, and decay_tab.
    Returns (enc, field_dims_sum) with enc[name] = (X, y, users).
    """
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
    """Copied verbatim from iter23_fused_decay_sampling/data_ext.py's
    compute_final_decayed_pos (itself copied verbatim from iter22's).
    NON-CAUSAL, single scalar per user -- the recency-decayed count of that
    user's TRAIN positive rows, decayed to a single fixed reference date =
    the END of the train period (max date present in train). This is a
    TRAINING-TIME SAMPLING WEIGHT (iter23's axis), the direct decayed analog
    of `pos_len` (the flat raw positive-row count iter3/iter9/iter16/iter19/
    iter24 all use to weight which users get sampled for BPR pairs) -- NOT a
    per-row feature fed to the model. It uses the SAME lazy-decay
    exponential formula (0.5 ** (gap_days / halflife)) as
    compute_decay_features' `decayed_pos` output, but evaluated once at the
    final decay state per user rather than causally per-row -- matching how
    `pos_len` itself is already a single non-causal aggregate over ALL of
    train (build_pos_neg_index has no per-row causal restriction either;
    it's a global sampling-frequency choice, not something the model sees as
    a feature). No leakage concern: this value never enters any row's
    feature vector, it only controls how OFTEN a user's (already
    causally-correct) rows get drawn as BPR anchors during training.

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


AUX_LABELS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']


def load_aux_labels(data_dir):
    """Verbatim from iter31_multitask/data_ext.py: loads the 5 auxiliary
    engagement labels per split, in the same row order as `_load_raw_time`'s
    per-split lists (same files, same date-range predicate) -- so
    load_aux_labels(...)['train'][label][i] lines up with row i of
    splits['train']/Xtr[i] with no join/sort step needed. Not features --
    train.py must only ever read the 'train' entry (see iter36 RESULT.md's
    leakage argument)."""
    raw = {name: [] for name in SPLITS}
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                d = int(r['date'])
                vals = tuple(1.0 if r[c] != '0' else 0.0 for c in AUX_LABELS)
                for name, (lo, hi) in SPLITS.items():
                    if lo <= d <= hi:
                        raw[name].append(vals)
                        break
    out = {}
    for name, lst in raw.items():
        arr = np.array(lst, dtype=np.float32) if lst else np.zeros((0, len(AUX_LABELS)), dtype=np.float32)
        out[name] = {c: arr[:, i] for i, c in enumerate(AUX_LABELS)}
    return out


if __name__ == '__main__':
    import argparse, time
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
    # PART A: decay-family (rate/act, fine grid) causal spot-checks
    # ================================================================
    print("\n=== PART A: decay-feature (rate/act) causal spot-checks (brute force) ===")
    for h in HALFLIVES:
        pcol, tcol = _halflife_col(h)
        tot = np.array([r[tcol] for r in flat])
        print(f"halflife={h:4.1f}d  decayed_total>0 coverage: {np.mean(tot > 0)*100:.2f}%  "
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
        for h in TAB_HALFLIVES:
            tcol = _tab_halflife_col(h)
            assert r[tcol] == 0.0, "CAUSALITY BUG: zero-activity row has nonzero decay_tab!"
    print(f"zero-activity rows ({len(zero_examples)} checked): decayed_pos/total/tab correctly 0.0.")

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
    # PART B: decayed tab_pos (decay_tab, iter20-style) causal spot-checks
    # ================================================================
    print("\n=== PART B: decayed tab_pos causal spot-checks (brute force) ===")
    for h in TAB_HALFLIVES:
        tcol = _tab_halflife_col(h)
        tot = np.array([r[tcol] for r in flat])
        print(f"tab halflife={h:4.1f}d  decayed_tab_pos>0 coverage: {np.mean(tot > 0)*100:.2f}%  "
              f"(mean={tot.mean():.3f}, max={tot.max():.3f})")

    max_err_tab = 0.0
    sample_idx2 = rng.choice(n, size=30, replace=False)
    for idx in sample_idx2:
        r = flat[idx]
        date, uid, tab = r[0], r[1], r[4]
        d_ord = _date_to_ordinal(date)
        earlier_same_tab = [rr for rr in flat if rr[1] == uid and rr[4] == tab and rr[0] < date and rr[6] == 1]
        for h in TAB_HALFLIVES:
            tcol = _tab_halflife_col(h)
            manual = sum(0.5 ** ((d_ord - _date_to_ordinal(rr[0])) / h) for rr in earlier_same_tab)
            err = abs(manual - r[tcol])
            max_err_tab = max(max_err_tab, err)
            assert err < 1e-6, f"CAUSALITY BUG: decayed_tab_pos mismatch h={h} idx={idx} manual={manual} got={r[tcol]}"
    print(f"30 random rows x {len(TAB_HALFLIVES)} tab-halflives: all decayed_tab_pos match brute force "
          f"(max abs err {max_err_tab:.2e}). No leakage detected.")

    zero_tab_examples = [r for r in flat if r[IDX['tab_pos']] == 0 and r[IDX['activity']] > 0][:5]
    for r in zero_tab_examples:
        for h in TAB_HALFLIVES:
            tcol = _tab_halflife_col(h)
            assert r[tcol] == 0.0, "CAUSALITY BUG: zero-tab_pos row (with other activity) has nonzero decay_tab!"
    if zero_tab_examples:
        print(f"zero-tab_pos-but-nonzero-activity rows ({len(zero_tab_examples)} checked): "
              f"decayed_tab_pos correctly 0.0 despite nonzero decay_rate/act.")

    by_u_tab_date = collections.defaultdict(list)
    for idx, r in enumerate(flat):
        if r[6] == 1:
            by_u_tab_date[(r[1], r[4], r[0])].append(idx)
    same_date_tab_case = next((v for v in by_u_tab_date.values() if len(v) >= 2), None)
    if same_date_tab_case:
        h = TAB_HALFLIVES[0]
        tcol = _tab_halflife_col(h)
        vals = [(flat[idx][1], flat[idx][4], flat[idx][0], flat[idx][tcol]) for idx in same_date_tab_case]
        dv0 = vals[0][3]
        for uid, tab, date, dv in vals:
            assert abs(dv - dv0) < 1e-9, \
                "CAUSALITY BUG: same-date pair should show identical (mutually-exclusive) decay_tab state!"
        print(f"same-date-pair edge case (decay_tab, {len(same_date_tab_case)} rows): decayed_tab_pos "
              "identical across the pair, as expected. PASSED.")
    else:
        print("(no same-user/same-tab/same-date positive pair found -- skipping, not a failure)")

    # ================================================================
    # PART C: momentum-family causal spot-checks (adapted from iter18/
    # iter19's __main__), re-run against the COMBINED row tuple.
    # ================================================================
    print("\n=== PART C: momentum-feature causal spot-checks (brute force) ===")
    last1 = np.array([r[IDX['last1']] for r in flat])
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
    # PART D: cross-family joint edge case -- a same-user, same-date pair
    # with distinct time_ms. Now THREE families (decay, decay_tab,
    # momentum) must be checked pairwise on the same rows: decay/decay_tab
    # must be IDENTICAL across the pair (date-level, blind to intra-date
    # order); momentum must DIFFER and correctly resolve the true time_ms
    # order. This directly checks all pairwise interactions the dispatch
    # prompt calls out: decay vs decay_tab (should agree: both date-blind),
    # decay vs momentum (should disagree: date-blind vs time_ms-aware),
    # decay_tab vs momentum (same).
    # ================================================================
    print("\n=== PART D: cross-family joint edge case (same-date, different time_ms pair) ===")
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
        th = TAB_HALFLIVES[0]
        ttcol = _tab_halflife_col(th)
        print(f"user={uid} date={date}: {len(idxs_sorted)} rows, same calendar date, distinct time_ms")
        decay_vals, decay_tab_vals = set(), set()
        for rank, i in enumerate(idxs_sorted):
            r = flat[i]
            decay_vals.add((round(r[pcol], 9), round(r[tcol], 9)))
            decay_tab_vals.add(round(r[ttcol], 9))
            print(f"  rank={rank} time_ms={r[IDX['time_ms']]} label={r[IDX['label']]} "
                  f"decay_pos={r[pcol]:.4f} decay_total={r[tcol]:.4f} decay_tab={r[ttcol]:.4f} "
                  f"last1={r[IDX['last1']]} gap_ms={r[IDX['gap_ms']]}")
        assert len(decay_vals) == 1, \
            "CAUSALITY BUG: decay (rate/act) feature should be IDENTICAL across a same-date pair"
        assert len(decay_tab_vals) == 1, \
            "CAUSALITY BUG: decay_tab feature should be IDENTICAL across a same-date pair"
        last1_vals = [flat[i][IDX['last1']] for i in idxs_sorted]
        assert last1_vals[1] == flat[idxs_sorted[0]][IDX['label']], \
            "CAUSALITY BUG: momentum last1 for later-time_ms row should equal earlier row's label"
        print("  -> decay AND decay_tab features are IDENTICAL across the pair (both date-level, "
              "correctly blind to intra-date order); momentum last1 correctly DIFFERS and resolves "
              "the true time_ms order. All three families verified independently correct on the "
              "same rows -- no cross-contamination from the join (decay<->decay_tab agree by "
              "construction, both disagree with momentum by construction).")
    else:
        print("(no same-user/same-date pair with >=2 distinct time_ms found -- skipping)")

    # ================================================================
    # PART E: decay-aware BPR sampling-weight spot-check (iter22/iter23's
    # PART D, re-run here against this module's own row tuples/loader to
    # confirm no drift was introduced by combining it with the 4-traversal
    # feature harness above).
    # ================================================================
    print("\n=== PART E: decay-aware sampling-weight spot-check (brute force) ===")
    train_rows = ext['train']
    ref_ord = max(_date_to_ordinal(r[0]) for r in train_rows)
    print(f"train period end (reference date ordinal): {ref_ord}  "
          f"({len(set(r[1] for r in train_rows if r[6] == 1))} users with >=1 train positive)")

    decayed_pos_dict = compute_final_decayed_pos(train_rows, halflife=3)
    rng2 = np.random.default_rng(1)
    users_with_pos = sorted(decayed_pos_dict.keys())
    sample_users = rng2.choice(users_with_pos, size=min(30, len(users_with_pos)), replace=False)
    max_err_samp = 0.0
    for u in sample_users:
        manual = sum(0.5 ** ((ref_ord - _date_to_ordinal(r[0])) / 3)
                     for r in train_rows if r[1] == u and r[6] == 1)
        err = abs(manual - decayed_pos_dict[u])
        max_err_samp = max(max_err_samp, err)
        assert err < 1e-6, f"CAUSALITY BUG: compute_final_decayed_pos mismatch user={u}"
    print(f"{len(sample_users)} random users: compute_final_decayed_pos matches brute-force "
          f"recount of 0.5**(gap_days/3) over all TRAIN positive rows "
          f"(max abs err {max_err_samp:.2e}). No arithmetic error.")

    zero_pos_users = [u for u in set(r[1] for r in train_rows) if u not in decayed_pos_dict][:5]
    for u in zero_pos_users:
        assert all(r[6] == 0 for r in train_rows if r[1] == u), \
            "CAUSALITY BUG: a user with a train positive is missing from decayed_pos_dict!"
    if zero_pos_users:
        print(f"zero-train-positive users ({len(zero_pos_users)} checked): correctly absent from "
              f"decayed_pos dict (never contribute sampling weight).")

    last_train_date = max(rr[0] for rr in train_rows)
    last_date_users = [r[1] for r in train_rows if r[0] == last_train_date and r[6] == 1]
    if last_date_users:
        u = last_date_users[0]
        n_pos_last_date = sum(1 for r in train_rows if r[1] == u and r[6] == 1
                               and r[0] == last_train_date)
        n_pos_total = sum(1 for r in train_rows if r[1] == u and r[6] == 1)
        if n_pos_total == n_pos_last_date:
            # all of this user's positives fall on the reference date itself -> decayed == raw count
            assert abs(decayed_pos_dict[u] - n_pos_total) < 1e-9, \
                "CAUSALITY BUG: reference-date edge case should give decayed_pos == raw count!"
            print(f"reference-date edge case (user {u}, {n_pos_total} positives all on last train "
                  f"date): decayed_pos == raw count exactly ({decayed_pos_dict[u]:.6f}). PASSED.")

    print("\nAll causal spot-checks (decay + decay_tab + momentum + cross-family joint + "
          "sampling-weight) passed.")
