"""iter20: two refinements of iter16's recency-decayed causal history features.

Axis A — finer half-life grid: iter16's own 5-point sweep (1,3,7,14,30 days)
found a non-monotonic peak at 3 days but never tested anything between 1 and
7. This module recomputes `decay_rate_H`/`decay_act_H` (identical mechanism
to iter16 — see compute_decay_features, copied verbatim) over a finer grid
HALFLIVES = [1.5, 2, 2.5, 3, 3.5, 4, 5] days bracketing the peak.

Axis B — decayed `tab_pos`: iter16 decayed `rate`/`activity` (both computed
over ALL of a user's prior rows) but left `tab_pos` (count of a user's PRIOR
POSITIVE rows in the SAME tab) flat/undecayed. This module adds
`compute_decay_tab_features`, a NEW function applying the exact same
exponential half-life decay mechanism to the (user, tab) keyed running state
instead of iter16's (user,) keyed state — i.e. a decayed analogue of
`tab_pos` the same way iter16's `decay_act` is a decayed analogue of
`activity`. Swept over TAB_HALFLIVES = [3, 7] days (guided by Axis A).

CRITICAL correctness note: `compute_decay_tab_features` uses the identical
two-phase date-grouped traversal pattern validated in iter16/iter9 (read
pre-day-boundary state for every row in a date group, THEN fold that group's
POSITIVE rows into the running per-(user,tab) state) — same-date rows never
see each other, future dates never leak into past ones. This is verified
independently in this module's own `__main__` block (brute-force spot-check,
zero-history rows, same-date-pair edge case), not assumed from iter16.

Reuses ../../data.py's load() unmodified; does NOT modify data.py, iter9, or
iter16.
"""
import os, sys, collections, datetime, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from data import load as _load_raw  # noqa: E402  (reuse original raw loader, untouched)

LABEL = 'long_view'
BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
ALPHA = 1.0  # Laplace smoothing constant, same shape as iter9/iter16's `rate`

# Axis A grid: fine, bracketing iter16's 3-day peak with several points each side.
HALFLIVES = [1.5, 2, 2.5, 3, 3.5, 4, 5]  # days
# Axis B grid: candidate halflives for the NEW decayed tab_pos feature.
TAB_HALFLIVES = [3, 7]  # days


def _date_to_ordinal(d):
    y, m, day = d // 10000, (d // 100) % 100, d % 100
    return datetime.date(y, m, day).toordinal()


def compute_causal_features(rows):
    """Identical to iter9/iter16's compute_causal_features — copied verbatim
    (flat activity/tab_pos/prior_pos/prior_total), kept so flat `rate`/`tab`
    can still be used standalone or combined with decayed features."""
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
    """Identical mechanism to iter16's compute_decay_features — copied
    verbatim (lazy-decay running state per user, exact not approximate).
    Returns (decayed_pos, decayed_total): each an (n, H) float64 array."""
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
    """NEW for iter20 — decayed analogue of iter9/iter16's flat `tab_pos`
    (count of user's prior POSITIVE rows in the SAME tab). Same lazy-decay
    running-state trick as compute_decay_features, but keyed by (user, tab)
    instead of user, and tracking only a positive-count state (mirroring
    `tab_pos` itself, which is a raw count, not a rate — there is no
    "tab_total" analogue in the base feature set, so none is computed here).

    Two-phase date-grouped traversal, identical causal guarantee to
    compute_decay_features: phase 1 reads every row's pre-day-boundary state
    (BEFORE any of that date's rows are folded in), phase 2 folds that date's
    POSITIVE rows into the (user, tab)-keyed running state. A key's
    last-update date is always strictly earlier than the date group currently
    being read, so same-date rows never see each other or themselves, and no
    future row leaks backward.

    Returns decayed_tab_pos: (n, H) float64 array, column order matching
    `halflives`.
    """
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

    key_last_ord = {}   # (user, tab) -> ordinal date of last state update
    key_state = {}       # (user, tab) -> list[float] len H, decayed pos as-of last_ord

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
            u, tab = r[1], r[4]
            key = (u, tab)
            last = key_last_ord.get(key)
            if last is not None:
                gap = d_ord - last
                state = key_state[key]
                for h in range(H):
                    decayed_tab_pos[idx, h] = state[h] * (day_mult[h] ** gap)
            # else: no prior (user,tab) history -> stays 0.0 (already initialized)

        # ---- phase 2: fold this date group's POSITIVE rows into running state ----
        day_pos_count = collections.defaultdict(int)  # keyed by (user, tab)
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


_CACHE_VERSION = 1


def _cache_path(data_dir, halflives, tab_halflives):
    key = '-'.join(str(h) for h in halflives) + '__tab_' + '-'.join(str(h) for h in tab_halflives)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f'.cache_v{_CACHE_VERSION}_{key}.pkl')


def load_ext(data_dir, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES, use_cache=True):
    """Returns dict of (train/valid/test) -> list of extended rows:
    (date, user_id, video_id, author_id, tab, duration_ms, label,
     activity, tab_pos, prior_pos, prior_total,                          <- idx 7-10 (flat, iter9)
     decay_pos_h0, decay_total_h0, decay_pos_h1, decay_total_h1, ...,    <- idx 11+ (decay_rate/act, iter16-style, fine grid)
     decay_tab_h0, decay_tab_h1, ...)                                    <- after that (NEW, decayed tab_pos)
    where h0,h1,... follow the order of `halflives` (rate/act block) then
    `tab_halflives` (tab block).
    """
    cpath = _cache_path(data_dir, halflives, tab_halflives)
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
    decayed_tab_pos = compute_decay_tab_features(flat, tab_halflives)

    ext = {name: [] for name in order}
    for i, (r, name) in enumerate(zip(flat, owner)):
        extra = [flat_feats['activity'][i], flat_feats['tab_pos'][i],
                  flat_feats['prior_pos'][i], flat_feats['prior_total'][i]]
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


def _halflife_col(h, halflives=HALFLIVES):
    """Column indices (decay_pos, decay_total) for half-life h in the extended tuple."""
    pos = halflives.index(h)
    return 11 + 2 * pos, 11 + 2 * pos + 1


def _tab_halflife_col(h, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES):
    """Column index (single, positive-only) for decayed-tab half-life h."""
    base = 11 + 2 * len(halflives)
    return base + tab_halflives.index(h)


def encode_ext(splits, feature_set=('rate',), halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES):
    """splits: dict from load_ext(), each row is a tuple as documented above.
    feature_set: subset/order of:
      'activity'        flat iter9 activity (bucketed)
      'tab'              flat iter9 tab_pos (bucketed)
      'rate'             flat iter9 Laplace-smoothed rate (bucketed)
      'decay_rate_H'     decayed Laplace-smoothed rate at half-life H days (bucketed)
      'decay_act_H'      decayed activity (=decayed_total) at half-life H days (bucketed)
      'decay_tab_H'      NEW: decayed tab_pos at half-life H days (bucketed), H in tab_halflives
    Mirrors iter16's encode_ext exactly for the base fields + flat/decay
    extras, plus the new decay_tab_H option; each requested field gets its
    own train-fit quantile bucketing (10 buckets).
    Returns (enc, field_dims_sum) with enc[name] = (X, y, users)."""
    tr = splits['train']
    dur_edges = _bucket_edges([x[5] for x in tr])

    def raw_dur(x):
        return str(int(np.searchsorted(dur_edges, x[5])))

    def parse_feat(name):
        if name in ('activity', 'tab', 'rate'):
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
        elif kind == 'decay_tab':
            tcol = _tab_halflife_col(h, halflives, tab_halflives)
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
        elif kind == 'decay_tab':
            tcol = _tab_halflife_col(h, halflives, tab_halflives)
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

    # ---- coverage report: decay_rate/decay_act (Axis A fine grid) ----
    for h in HALFLIVES:
        pcol, tcol = _halflife_col(h)
        tot = np.array([r[tcol] for r in flat])
        print(f"halflife={h:4.1f}d  decayed_total>0 coverage: {np.mean(tot > 0)*100:.2f}%  "
              f"(mean={tot.mean():.3f}, max={tot.max():.3f})")

    # ---- coverage report: decay_tab (Axis B, NEW) ----
    for h in TAB_HALFLIVES:
        tcol = _tab_halflife_col(h)
        tot = np.array([r[tcol] for r in flat])
        print(f"tab halflife={h:4.1f}d  decayed_tab_pos>0 coverage: {np.mean(tot > 0)*100:.2f}%  "
              f"(mean={tot.mean():.3f}, max={tot.max():.3f})")

    # ---- causal spot-checks (brute-force): decay_rate/decay_act, sanity re-check ----
    print("\n--- causal spot-checks: decayed_pos/decayed_total (rate/act, brute force) ---")
    rng = np.random.default_rng(0)
    HL_CHECK = [1.5, 3, 5]
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

    # ---- NEW causal spot-checks (brute-force): decay_tab_pos ----
    print("\n--- causal spot-checks: decayed_tab_pos (NEW, brute force) ---")
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

    # ---- zero-history rows must have exactly 0 decayed values (incl. NEW decay_tab) ----
    zero_examples = [r for r in flat if r[7] == 0][:5]  # activity==0 -> genuinely no prior rows at all
    for r in zero_examples:
        for h in HALFLIVES:
            pcol, tcol = _halflife_col(h)
            assert r[pcol] == 0.0 and r[tcol] == 0.0, "CAUSALITY BUG: zero-activity row has nonzero rate/act decay!"
        for h in TAB_HALFLIVES:
            tcol = _tab_halflife_col(h)
            assert r[tcol] == 0.0, "CAUSALITY BUG: zero-activity row has nonzero decay_tab!"
    print(f"zero-activity rows ({len(zero_examples)} checked): decayed_pos/total/tab correctly 0.0.")

    # also check zero-history specifically for tab (tab_pos==0, i.e. no prior SAME-TAB positive,
    # even if user has other prior activity) -> decay_tab must be 0 even when decay_rate/act are not
    zero_tab_examples = [r for r in flat if r[8] == 0 and r[7] > 0][:5]
    for r in zero_tab_examples:
        for h in TAB_HALFLIVES:
            tcol = _tab_halflife_col(h)
            assert r[tcol] == 0.0, "CAUSALITY BUG: zero-tab_pos row (with other activity) has nonzero decay_tab!"
    if zero_tab_examples:
        print(f"zero-tab_pos-but-nonzero-activity rows ({len(zero_tab_examples)} checked): "
              f"decayed_tab_pos correctly 0.0 despite nonzero decay_rate/act.")

    # ---- same-date-pair edge case: two same-user, same-tab positives on the same date ----
    print("\n--- same-date-pair edge case (decay_tab; two same-user same-tab positives, same date) ---")
    by_u_tab_date = collections.defaultdict(list)
    for idx, r in enumerate(flat):
        if r[6] == 1:
            by_u_tab_date[(r[1], r[4], r[0])].append(idx)
    same_date_case = next((v for v in by_u_tab_date.values() if len(v) >= 2), None)
    if same_date_case:
        h = TAB_HALFLIVES[0]
        tcol = _tab_halflife_col(h)
        vals = [(flat[idx][1], flat[idx][4], flat[idx][0], flat[idx][tcol]) for idx in same_date_case]
        print(f"tab halflife={h}d, (user,tab,date) triple with {len(same_date_case)} same-date positives:")
        for uid, tab, date, dv in vals:
            print(f"  user={uid} tab={tab} date={date} decayed_tab_pos={dv:.4f}")
        dv0 = vals[0][3]
        for uid, tab, date, dv in vals:
            assert abs(dv - dv0) < 1e-9, \
                "CAUSALITY BUG: same-date pair should show identical (mutually-exclusive) decay_tab state!"
        print("  -> identical decayed_tab_pos across the same-date pair, as expected "
              "(same-date rows never see each other). PASSED.")
    else:
        print("  (no same-date same-tab positive pair found in this data — skipping, not a failure)")

    print("\nAll causal spot-checks passed (rate/act fine grid + NEW decay_tab). No same-date or future leakage detected.")
