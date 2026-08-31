"""iter63: decayed per-tab POSITIVE RATE, not just raw decayed count.

Motivation: `decay_tab_3` (in every feature set since iter24) is the raw
decayed COUNT of a user's prior positives within the current row's tab
(`compute_decay_tab_features` in iter20/iter24/iter27's data_ext.py). It
has no denominator -- it conflates "engaged with this tab" with "visits
this tab a lot" the same way a raw activity count would, whereas the
*overall* (non-tab) decay features already learned this lesson: iter9/
iter16 use `rate` = decayed_pos/decayed_total (Laplace-smoothed), not a
raw count, precisely because rate is a cleaner, magnitude-normalized
signal. `decay_tab` never got the same treatment because the original
`compute_decay_tab_features` only ever tracked the numerator
(decayed_tab_pos), never a matching per-(user,tab) denominator
(decayed_tab_total = decayed count of ALL visits to that tab, not just
positive ones).

This file is a byte-for-byte copy of iter27_triple_fusion/data_ext.py
EXCEPT `compute_decay_tab_features`, which is extended to also track and
return `decayed_tab_total` (a second parallel per-(user,tab) decayed
counter, identical lazy-decay mechanism, just counting all rows in that
tab instead of only positive ones), and `load_ext`, which appends the new
total columns after the existing decayed_tab_pos columns. Does NOT modify
data.py or any file iter27 itself imports (iter18's momentum function).
"""
import os, sys, csv, collections, datetime, importlib.util, pickle
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from data import SPLITS  # noqa: E402

LABEL = 'long_view'
BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
ALPHA = 1.0
K_DEFAULT = 5

HALFLIVES = [2, 2.5, 3, 3.5]  # days
TAB_HALFLIVES = [3, 7]  # days


def _load_module(name, rel_path):
    path = os.path.join(_THIS_DIR, *rel_path.split('/'))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_iter18_de = _load_module('iter63_iter18_data_ext', '../iter18_momentum/data_ext.py')
compute_momentum_features = _iter18_de.compute_momentum_features


def _date_to_ordinal(d):
    y, m, day = d // 10000, (d // 100) % 100, d % 100
    return datetime.date(y, m, day).toordinal()


def compute_causal_features(rows):
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
    """MODIFIED from iter27: now tracks BOTH decayed_tab_pos (positive rows
    in this (user,tab)) AND decayed_tab_total (ALL rows in this (user,tab),
    positive or not) via the same two parallel per-(user,tab) lazy-decay
    states, so a Laplace-smoothed per-tab RATE can be derived downstream
    (decayed_tab_pos+alpha)/(decayed_tab_total+2*alpha) -- exactly mirroring
    how compute_decay_features tracks decayed_pos AND decayed_total (not
    just decayed_pos) so decay_rate can be computed. Returns
    (decayed_tab_pos, decayed_tab_total): each an (n, H) float64 array."""
    n = len(rows)
    H = len(halflives)
    order = sorted(range(n), key=lambda i: rows[i][0])
    day_mult = [0.5 ** (1.0 / h) for h in halflives]

    decayed_tab_pos = np.zeros((n, H), dtype=np.float64)
    decayed_tab_total = np.zeros((n, H), dtype=np.float64)

    ord_cache = {}
    def ordf(d):
        v = ord_cache.get(d)
        if v is None:
            v = _date_to_ordinal(d)
            ord_cache[d] = v
        return v

    key_last_ord = {}
    key_pos_state = {}
    key_total_state = {}

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
                pstate = key_pos_state[key]
                tstate = key_total_state[key]
                for h in range(H):
                    f = day_mult[h] ** gap
                    decayed_tab_pos[idx, h] = pstate[h] * f
                    decayed_tab_total[idx, h] = tstate[h] * f

        day_pos_count = collections.defaultdict(int)
        day_total_count = collections.defaultdict(int)
        for idx in day_idx:
            r = rows[idx]
            u, tab, label = r[1], r[4], r[6]
            key = (u, tab)
            day_total_count[key] += 1
            if label == 1:
                day_pos_count[key] += 1
        touched_keys = set(day_total_count.keys())
        for key in touched_keys:
            last = key_last_ord.get(key)
            if last is not None:
                gap = d_ord - last
                pstate = key_pos_state[key]
                tstate = key_total_state[key]
                new_p = [pstate[h] * (day_mult[h] ** gap) for h in range(H)]
                new_t = [tstate[h] * (day_mult[h] ** gap) for h in range(H)]
            else:
                new_p = [0.0] * H
                new_t = [0.0] * H
            pc = day_pos_count.get(key, 0)
            tc = day_total_count[key]
            for h in range(H):
                new_p[h] += pc
                new_t[h] += tc
            key_pos_state[key] = new_p
            key_total_state[key] = new_t
            key_last_ord[key] = d_ord
        i = j
    return decayed_tab_pos, decayed_tab_total


def _load_raw_time(data_dir):
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
DECAY_BASE = 18


def _halflife_col(h, halflives=HALFLIVES):
    pos = halflives.index(h)
    return DECAY_BASE + 2 * pos, DECAY_BASE + 2 * pos + 1


def _tab_halflife_col(h, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES):
    base = DECAY_BASE + 2 * len(halflives)
    return base + tab_halflives.index(h)


def _tab_halflife_total_col(h, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES):
    """NEW: column index of decayed_tab_total for halflife h -- lives after
    ALL decayed_tab_pos columns (kept contiguous for backward layout
    compatibility with iter27's decayed_tab_pos block)."""
    base = DECAY_BASE + 2 * len(halflives) + len(tab_halflives)
    return base + tab_halflives.index(h)


_CACHE_VERSION = 1


def _cache_path(halflives, tab_halflives):
    key = '-'.join(str(h) for h in halflives) + '__tab_' + '-'.join(str(h) for h in tab_halflives)
    return os.path.join(_THIS_DIR, f'.cache_v{_CACHE_VERSION}_{key}.pkl')


def load_ext(data_dir, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES, K=K_DEFAULT, use_cache=True):
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
    decayed_tab_pos, decayed_tab_total = compute_decay_tab_features(flat, tab_halflives)

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
        for h in range(len(tab_halflives)):  # NEW: total counts, appended after all pos columns
            extra.append(decayed_tab_total[i, h])
        ext[name].append(r + tuple(extra))

    if use_cache:
        with open(cpath, 'wb') as fh:
            pickle.dump(ext, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return ext


if __name__ == '__main__':
    import time
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
    print(f"loading {DATA_DIR} ...")
    t0 = time.time()
    ext = load_ext(DATA_DIR, use_cache=False)
    print({k: len(v) for k, v in ext.items()}, f"  ({time.time()-t0:.1f}s)")

    flat = ext['train'] + ext['valid'] + ext['test']
    n = len(flat)

    print("\n=== causal spot-check: decayed_tab_total (brute force) ===")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=30, replace=False)
    max_err = 0.0
    for idx in sample_idx:
        r = flat[idx]
        date, uid, tab = r[0], r[1], r[4]
        d_ord = _date_to_ordinal(date)
        earlier_same_tab = [rr for rr in flat if rr[1] == uid and rr[4] == tab and rr[0] < date]
        for h in TAB_HALFLIVES:
            tcol = _tab_halflife_total_col(h)
            manual = sum(0.5 ** ((d_ord - _date_to_ordinal(rr[0])) / h) for rr in earlier_same_tab)
            err = abs(manual - r[tcol])
            max_err = max(max_err, err)
            assert err < 1e-6, f"CAUSALITY BUG: decayed_tab_total mismatch h={h} idx={idx} manual={manual} got={r[tcol]}"
    print(f"30 random rows x {len(TAB_HALFLIVES)} tab-halflives: decayed_tab_total matches brute force "
          f"(max abs err {max_err:.2e}). No leakage detected.")

    # sanity: decayed_tab_total >= decayed_tab_pos everywhere (total includes positives)
    bad = 0
    for h in TAB_HALFLIVES:
        pcol = None
    for h in TAB_HALFLIVES:
        tot_col = _tab_halflife_total_col(h)
        # find matching pos col via base offset (pos cols are contiguous block just before total block)
        base_pos = DECAY_BASE + 2 * len(HALFLIVES)
        pos_col = base_pos + TAB_HALFLIVES.index(h)
        for r in flat[:5000]:
            if r[tot_col] + 1e-9 < r[pos_col]:
                bad += 1
    assert bad == 0, f"CAUSALITY BUG: found {bad} rows where decayed_tab_total < decayed_tab_pos!"
    print(f"sanity (5000 rows x {len(TAB_HALFLIVES)} halflives): decayed_tab_total >= decayed_tab_pos everywhere. OK.")

    zero_examples = [r for r in flat if r[IDX['activity']] == 0][:5]
    for r in zero_examples:
        for h in TAB_HALFLIVES:
            tcol = _tab_halflife_total_col(h)
            assert r[tcol] == 0.0, "CAUSALITY BUG: zero-activity row has nonzero decay_tab_total!"
    print(f"zero-activity rows ({len(zero_examples)} checked): decayed_tab_total correctly 0.0.")

    print("\nAll causal spot-checks passed.")
