"""iter16: recency-decayed (exponential half-life) versions of iter9's causal
history counters, instead of flat cumulative counts.

iter9 (experiments/iter9_history_dense/data_ext.py) computes, via a strict-
causal (`<`, never `<=`) two-phase date-grouped traversal:
  - activity    = count of user's prior rows (any label)
  - tab_pos     = count of user's prior positive rows, same tab
  - rate        = (prior_pos + 1) / (prior_total + 2)   [flat Laplace smoothing]

iter11 found `rate` is by far the dominant of the three. This iteration asks:
does a TIME-DECAYED version of `rate` (recent positives count more than old
ones) carry more signal than the flat cumulative version? The log window is
only ~3 weeks, so decay may or may not matter.

Decay formulation (half-life parameterized):
  decayed_pos(u, d)   = sum over user u's PRIOR positive rows r (date(r) < d)
                         of 0.5 ** ((d - date(r)) / halflife_days)
  decayed_total(u, d) = same sum but over ALL prior rows (not just positive)
  decayed_rate         = (decayed_pos + 1) / (decayed_total + 2)   [same Laplace shape]
  decayed_activity      = decayed_total itself (a decayed version of iter9's `activity`)

CRITICAL correctness note: this is computed with the EXACT SAME two-phase
date-grouped traversal pattern as iter9 (read pre-day-boundary state for every
row in a date group, THEN fold that group's rows into the running state) —
same-date rows never see each other, and future dates never leak into past
ones. Multiple half-lives are computed in a single pass for efficiency, using
a lazy-decay running-state trick (see `compute_decay_features` docstring)
that is mathematically exact (not an approximation) — verified by brute-force
spot-checks in `__main__`, including a same-date-pair edge case.

Reuses ../../data.py's load() unmodified; does NOT modify data.py or iter9.
"""
import os, sys, collections, datetime, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from data import load as _load_raw  # noqa: E402  (reuse original raw loader, untouched)

LABEL = 'long_view'
BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
ALPHA = 1.0  # Laplace smoothing constant, same shape as iter9's `rate`
HALFLIVES = [1, 3, 7, 14, 30]  # days; 30 ~= effectively flat (sanity check vs iter9)


def _date_to_ordinal(d):
    y, m, day = d // 10000, (d // 100) % 100, d % 100
    return datetime.date(y, m, day).toordinal()


def compute_causal_features(rows):
    """Identical to iter9's compute_causal_features — copied verbatim (flat
    activity/tab_pos/prior_pos/prior_total), kept so flat `rate` can still be
    used standalone or combined with decayed features."""
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
    """Same strict-causal two-phase date-grouped traversal as iter9's
    compute_causal_features, generalized to exponential time-decay with
    multiple half-lives computed together in a single pass.

    Lazy-decay trick (exact, not approximate): for each user we keep a
    running state (last_update_ordinal_date, pos_value, total_value) where
    pos_value/total_value are the decayed sums AS OF last_update_ordinal_date
    (i.e. already decayed forward to that date, with a fresh contribution of
    weight 1.0 added at that date for each row observed there). To read the
    decayed sum as of a LATER date d, we just multiply the stored value by
    0.5 ** ((d - last_update_ordinal_date) / halflife) — mathematically
    identical to summing 0.5**((d - date(r))/halflife) over every individual
    prior row r, because exponential decay composes multiplicatively over
    elapsed time regardless of how it's chunked.

    Because we always READ (phase 1) for every row in a date group BEFORE
    folding that same date group's contributions into the running state
    (phase 2), and a user's last_update_ordinal_date is always strictly
    earlier than the date group currently being read (it can only have been
    set by an EARLIER date group), same-date rows never see each other or
    themselves, and no future row ever leaks backward — identical causal
    guarantee to iter9's flat counters.

    Returns (decayed_pos, decayed_total): each an (n, H) float64 numpy array,
    H = len(halflives), column order matching `halflives`.
    """
    n = len(rows)
    H = len(halflives)
    order = sorted(range(n), key=lambda i: rows[i][0])
    day_mult = [0.5 ** (1.0 / h) for h in halflives]  # per-1-day-gap decay multiplier

    decayed_pos = np.zeros((n, H), dtype=np.float64)
    decayed_total = np.zeros((n, H), dtype=np.float64)

    ord_cache = {}
    def ordf(d):
        v = ord_cache.get(d)
        if v is None:
            v = _date_to_ordinal(d)
            ord_cache[d] = v
        return v

    user_last_ord = {}          # user -> ordinal date of last state update
    user_pos_state = {}         # user -> list[float] len H, decayed pos as-of last_ord
    user_total_state = {}       # user -> list[float] len H, decayed total as-of last_ord

    i = 0
    while i < n:
        j = i
        d = rows[order[i]][0]
        while j < n and rows[order[j]][0] == d:
            j += 1
        day_idx = order[i:j]
        d_ord = ordf(d)

        # ---- phase 1: read pre-day-boundary decayed state for every row ----
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
            # else: no prior history -> stays 0.0 (already initialized)

        # ---- phase 2: fold this date group's rows into the running state ----
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
                new_p[h] += pc   # weight 1.0 per positive row observed AT date d
                new_t[h] += tc   # weight 1.0 per row (any label) observed AT date d
            user_pos_state[u] = new_p
            user_total_state[u] = new_t
            user_last_ord[u] = d_ord
        i = j
    return decayed_pos, decayed_total


_CACHE_VERSION = 2


def _cache_path(data_dir, halflives):
    key = '-'.join(str(h) for h in halflives)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f'.cache_v{_CACHE_VERSION}_{key}.pkl')


def load_ext(data_dir, halflives=HALFLIVES, use_cache=True):
    """Returns dict of (train/valid/test) -> list of extended rows:
    (date, user_id, video_id, author_id, tab, duration_ms, label,
     activity, tab_pos, prior_pos, prior_total,             <- iter9 flat feats (idx 7-10)
     decay_pos_h0, decay_total_h0, decay_pos_h1, decay_total_h1, ...)  <- idx 11+
    where h0,h1,... follow the order of `halflives`.
    """
    cpath = _cache_path(data_dir, halflives)
    if use_cache and os.path.exists(cpath):
        with open(cpath, 'rb') as fh:
            return pickle.load(fh)

    splits = _load_raw(data_dir)
    order = ('train', 'valid', 'test')
    flat, owner = [], []
    for name in order:
        for r in splits[name]:
            flat.append(r)
            owner.append(name)

    flat_feats = compute_causal_features(flat)
    decayed_pos, decayed_total = compute_decay_features(flat, halflives)

    ext = {name: [] for name in order}
    for i, (r, name) in enumerate(zip(flat, owner)):
        extra = [flat_feats['activity'][i], flat_feats['tab_pos'][i],
                  flat_feats['prior_pos'][i], flat_feats['prior_total'][i]]
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


def _halflife_col(h, halflives=HALFLIVES):
    """Column indices (decay_pos, decay_total) for half-life h in the extended tuple."""
    pos = halflives.index(h)
    return 11 + 2 * pos, 11 + 2 * pos + 1


def encode_ext(splits, feature_set=('rate',), halflives=HALFLIVES):
    """splits: dict from load_ext(), each row is a tuple as documented above.
    feature_set: subset/order of:
      'activity'      flat iter9 activity (bucketed)
      'tab'            flat iter9 tab_pos (bucketed)
      'rate'           flat iter9 Laplace-smoothed rate (bucketed)
      'decay_rate_H'   decayed Laplace-smoothed rate at half-life H days (bucketed)
      'decay_act_H'    decayed activity (=decayed_total) at half-life H days (bucketed)
    where H must be one of `halflives`. Mirrors iter9's encode_ext exactly for
    the base 5 fields + flat extras; each requested field gets its own
    train-fit quantile bucketing (10 buckets).
    Returns (enc, field_dims_sum) with enc[name] = (X, y, users)."""
    tr = splits['train']
    dur_edges = _bucket_edges([x[5] for x in tr])

    def raw_dur(x):
        return str(int(np.searchsorted(dur_edges, x[5])))

    def parse_feat(name):
        if name in ('activity', 'tab', 'rate'):
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
            extra_edges[name] = _bucket_edges([x[7] for x in tr])
        elif kind == 'tab':
            extra_edges[name] = _bucket_edges([x[8] for x in tr])
        elif kind == 'rate':
            extra_edges[name] = _bucket_edges([(x[9] + ALPHA) / (x[10] + 2 * ALPHA) for x in tr])
        elif kind == 'decay_rate':
            pcol, tcol = _halflife_col(h, halflives)
            extra_edges[name] = _bucket_edges([(x[pcol] + ALPHA) / (x[tcol] + 2 * ALPHA) for x in tr])
        elif kind == 'decay_act':
            pcol, tcol = _halflife_col(h, halflives)
            extra_edges[name] = _bucket_edges([x[tcol] for x in tr])

    def extra_val(x, name):
        kind, h = parse_feat(name)
        if kind == 'activity':
            return str(int(np.searchsorted(extra_edges[name], x[7])))
        elif kind == 'tab':
            return str(int(np.searchsorted(extra_edges[name], x[8])))
        elif kind == 'rate':
            r = (x[9] + ALPHA) / (x[10] + 2 * ALPHA)
            return str(int(np.searchsorted(extra_edges[name], r)))
        elif kind == 'decay_rate':
            pcol, tcol = _halflife_col(h, halflives)
            r = (x[pcol] + ALPHA) / (x[tcol] + 2 * ALPHA)
            return str(int(np.searchsorted(extra_edges[name], r)))
        elif kind == 'decay_act':
            pcol, tcol = _halflife_col(h, halflives)
            return str(int(np.searchsorted(extra_edges[name], x[tcol])))
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
    """iter22: NON-CAUSAL, single scalar per user -- the recency-decayed count
    of that user's TRAIN positive rows, decayed to a single fixed reference
    date = the END of the train period (max date present in train). This is a
    TRAINING-TIME SAMPLING WEIGHT, the direct decayed analog of `pos_len`
    (iter3/iter9/iter16's flat raw positive-row count used to weight which
    users get sampled for BPR pairs) -- NOT a per-row feature fed to the
    model. It intentionally uses the SAME lazy-decay exponential formula
    (0.5 ** (gap_days / halflife)) as compute_decay_features' `decayed_pos`
    output (same halflife=3d as decay_act_3/decay_rate_3), but evaluated once
    at the current/final decay state per user rather than causally per-row --
    matching how `pos_len` itself is already a single non-causal aggregate
    over ALL of train (build_pos_neg_index has no per-row causal restriction
    either; it's a global sampling-frequency choice, not something the model
    sees as a feature). No leakage concern: this value never enters any row's
    feature vector, it only controls how OFTEN a user's (already causally-
    correct) rows get drawn as BPR anchors during training.

    train_rows: list of raw/extended row tuples with date at index 0,
    user_id at index 1, label at index 6 (same layout as data.py/data_ext.py
    rows -- works on either the raw `splits['train']` from data.py or the
    extended tuples from load_ext(), both share this prefix layout).

    Returns: dict user_id -> decayed positive count (float), users with zero
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(os.path.dirname(__file__), '..', '..',
                                                         'KuaiRand-Pure', 'data'))
    ap.add_argument('--no_cache', action='store_true')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    t0 = __import__('time').time()
    ext = load_ext(a.data_dir, use_cache=not a.no_cache)
    print({k: len(v) for k, v in ext.items()}, f"  ({__import__('time').time()-t0:.1f}s)")

    flat = ext['train'] + ext['valid'] + ext['test']
    n = len(flat)

    # ---- coverage report ----
    for h in HALFLIVES:
        pcol, tcol = _halflife_col(h)
        tot = np.array([r[tcol] for r in flat])
        print(f"halflife={h:2d}d  decayed_total>0 coverage: {np.mean(tot > 0)*100:.2f}%  "
              f"(mean={tot.mean():.3f}, max={tot.max():.3f})")

    # ---- causal spot-checks (brute-force, same methodology as iter9) ----
    print("\n--- causal spot-checks: decayed_pos/decayed_total (brute force) ---")
    rng = np.random.default_rng(0)
    HL_CHECK = [1, 7, 30]
    sample_idx = rng.choice(n, size=25, replace=False)
    max_err = 0.0
    for idx in sample_idx:
        r = flat[idx]
        date, uid = r[0], r[1]
        d_ord = _date_to_ordinal(date)
        earlier = [rr for rr in flat if rr[1] == uid and rr[0] < date]
        for h in HL_CHECK:
            pcol, tcol = _halflife_col(h)
            manual_pos = sum(0.5 ** ((d_ord - _date_to_ordinal(rr[0])) / h)
                              for rr in earlier if rr[6] == 1)
            manual_tot = sum(0.5 ** ((d_ord - _date_to_ordinal(rr[0])) / h) for rr in earlier)
            err_p = abs(manual_pos - r[pcol])
            err_t = abs(manual_tot - r[tcol])
            max_err = max(max_err, err_p, err_t)
            assert err_p < 1e-6, f"CAUSALITY BUG: decayed_pos mismatch h={h} idx={idx} manual={manual_pos} got={r[pcol]}"
            assert err_t < 1e-6, f"CAUSALITY BUG: decayed_total mismatch h={h} idx={idx} manual={manual_tot} got={r[tcol]}"
    print(f"25 random rows x {len(HL_CHECK)} halflives: all decayed_pos/decayed_total match brute force "
          f"(max abs err {max_err:.2e}). No leakage detected.")

    # ---- zero-history rows must have exactly 0 decayed values ----
    zero_examples = [r for r in flat if r[7] == 0][:5]  # activity==0 -> genuinely no prior rows
    for r in zero_examples:
        for h in HALFLIVES:
            pcol, tcol = _halflife_col(h)
            assert r[pcol] == 0.0 and r[tcol] == 0.0, "CAUSALITY BUG: zero-activity row has nonzero decay!"
    print(f"zero-activity rows ({len(zero_examples)} checked): decayed_pos/total correctly 0.0.")

    # ---- same-date-pair edge case ----
    print("\n--- same-date-pair edge case (two same-user positives on the same date) ---")
    by_u_date = collections.defaultdict(list)
    for idx, r in enumerate(flat):
        if r[6] == 1:
            by_u_date[(r[1], r[0])].append(idx)
    same_date_case = next((v for v in by_u_date.values() if len(v) >= 2), None)
    if same_date_case:
        h = 7
        pcol, tcol = _halflife_col(h)
        vals = [(flat[idx][1], flat[idx][0], flat[idx][pcol], flat[idx][tcol]) for idx in same_date_case]
        print(f"halflife={h}d, user/date pair with {len(same_date_case)} same-date positives:")
        for uid, date, dp, dt in vals:
            print(f"  user={uid} date={date} decayed_pos={dp:.4f} decayed_total={dt:.4f}")
        # both rows in the pair must show IDENTICAL decayed_pos/total (neither sees the other,
        # since same-date rows never contribute to each other's read state)
        dp0 = vals[0][2]; dt0 = vals[0][3]
        for uid, date, dp, dt in vals:
            assert abs(dp - dp0) < 1e-9 and abs(dt - dt0) < 1e-9, \
                "CAUSALITY BUG: same-date pair should show identical (mutually-exclusive) decay state!"
        print("  -> identical decayed_pos/decayed_total across the same-date pair, as expected "
              "(same-date rows never see each other). PASSED.")
    else:
        print("  (no same-date-pair found in this data — skipping, not a failure)")

    print("\nAll causal spot-checks passed. No same-date or future leakage detected.")
