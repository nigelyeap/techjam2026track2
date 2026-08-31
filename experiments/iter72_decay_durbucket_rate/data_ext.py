"""iter72: decayed per-(user, duration-bucket) RATE -- same lazy-decay-per-key
mechanism as iter63's decay_tab_rate_3, applied to a duration bucket instead
of `tab`.

Motivation (per iter71's diagnosis): the decayed-rate transformation adds
value specifically when applied to a grouping axis COARSER than what's
already in the categorical feature set (as `tab` is relative to `user_id`/
`video_id`/`author_id`) -- author and tag granularity, tested in iter69/71,
sit at-or-below the existing categoricals and were fully redundant nulls.
`duration_ms` is currently used only as a raw numeric (never bucketed, never
used as a grouping key), and is a coarse, cross-cutting CONTENT property
exactly like `tab` -- shared across many distinct videos/authors, at a
handful of discrete levels once bucketed. This tests whether "does this
user prefer short vs. long videos, decayed over time" carries a genuinely
different signal from raw `duration_ms` the same way `decay_tab_rate_3`
differs from raw `tab`.

Bucket edges (ms): [0, 15s, 30s, 60s, 120s, 300s, inf) -- 6 buckets, chosen
from the train duration_ms distribution's quantiles (10th~11.5s,
25th~25.6s, median~69s, 75th~135s, 90th~235s), a fixed content property
computed with no label information, so no leakage risk.

Byte-for-byte copy of iter71_decay_author_rate/data_ext.py, with
compute_decay_durbucket_features added (identical algorithm, keyed on
(user_id, duration_bucket(duration_ms)) instead of (user_id, author_id)),
and load_ext extended to append the new columns after the author-decay
block.
"""
import os, sys, csv, collections, datetime, importlib.util, pickle
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from data import SPLITS  # noqa: E402

LABEL = 'long_view'
ALPHA = 1.0
K_DEFAULT = 5

HALFLIVES = [2, 2.5, 3, 3.5]  # days
TAB_HALFLIVES = [3, 7]  # days
DUR_BUCKET_EDGES = [15000, 30000, 60000, 120000, 300000]  # ms; produces 6 buckets
DUR_HALFLIVES = [3, 7]  # days


def _dur_bucket(duration_ms):
    for i, edge in enumerate(DUR_BUCKET_EDGES):
        if duration_ms < edge:
            return i
    return len(DUR_BUCKET_EDGES)


def _load_module(name, rel_path):
    path = os.path.join(_THIS_DIR, *rel_path.split('/'))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_iter18_de = _load_module('iter72_iter18_data_ext', '../iter18_momentum/data_ext.py')
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


def _compute_decay_key_features(rows, key_fn, halflives):
    n = len(rows)
    H = len(halflives)
    order = sorted(range(n), key=lambda i: rows[i][0])
    day_mult = [0.5 ** (1.0 / h) for h in halflives]

    decayed_key_pos = np.zeros((n, H), dtype=np.float64)
    decayed_key_total = np.zeros((n, H), dtype=np.float64)

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
            key = key_fn(r)
            last = key_last_ord.get(key)
            if last is not None:
                gap = d_ord - last
                pstate = key_pos_state[key]
                tstate = key_total_state[key]
                for h in range(H):
                    f = day_mult[h] ** gap
                    decayed_key_pos[idx, h] = pstate[h] * f
                    decayed_key_total[idx, h] = tstate[h] * f

        day_pos_count = collections.defaultdict(int)
        day_total_count = collections.defaultdict(int)
        for idx in day_idx:
            r = rows[idx]
            key = key_fn(r)
            label = r[6]
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
    return decayed_key_pos, decayed_key_total


def compute_decay_tab_features(rows, halflives=TAB_HALFLIVES):
    return _compute_decay_key_features(rows, lambda r: (r[1], r[4]), halflives)


def compute_decay_durbucket_features(rows, halflives=DUR_HALFLIVES):
    return _compute_decay_key_features(rows, lambda r: (r[1], _dur_bucket(r[5])), halflives)


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
    base = DECAY_BASE + 2 * len(halflives) + len(tab_halflives)
    return base + tab_halflives.index(h)


def _dur_halflife_col(h, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES, dur_halflives=DUR_HALFLIVES):
    base = DECAY_BASE + 2 * len(halflives) + 2 * len(tab_halflives)
    return base + dur_halflives.index(h)


def _dur_halflife_total_col(h, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES, dur_halflives=DUR_HALFLIVES):
    base = DECAY_BASE + 2 * len(halflives) + 2 * len(tab_halflives) + len(dur_halflives)
    return base + dur_halflives.index(h)


_CACHE_VERSION = 1


def _cache_path(halflives, tab_halflives, dur_halflives):
    key = ('-'.join(str(h) for h in halflives) + '__tab_' + '-'.join(str(h) for h in tab_halflives)
           + '__dur_' + '-'.join(str(h) for h in dur_halflives))
    return os.path.join(_THIS_DIR, f'.cache_v{_CACHE_VERSION}_{key}.pkl')


def load_ext(data_dir, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
             dur_halflives=DUR_HALFLIVES, K=K_DEFAULT, use_cache=True):
    cpath = _cache_path(halflives, tab_halflives, dur_halflives)
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
    decayed_dur_pos, decayed_dur_total = compute_decay_durbucket_features(flat, dur_halflives)

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
        for h in range(len(tab_halflives)):
            extra.append(decayed_tab_total[i, h])
        for h in range(len(dur_halflives)):
            extra.append(decayed_dur_pos[i, h])
        for h in range(len(dur_halflives)):
            extra.append(decayed_dur_total[i, h])
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

    print("\n=== bucket distribution (train) ===")
    from collections import Counter
    c = Counter(_dur_bucket(r[5]) for r in ext['train'])
    print(dict(sorted(c.items())))

    print("\n=== causal spot-check: decayed_dur_total (brute force) ===")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=30, replace=False)
    max_err = 0.0
    for idx in sample_idx:
        r = flat[idx]
        date, uid, bucket = r[0], r[1], _dur_bucket(r[5])
        d_ord = _date_to_ordinal(date)
        earlier_same_bucket = [rr for rr in flat if rr[1] == uid and _dur_bucket(rr[5]) == bucket and rr[0] < date]
        for h in DUR_HALFLIVES:
            tcol = _dur_halflife_total_col(h)
            manual = sum(0.5 ** ((d_ord - _date_to_ordinal(rr[0])) / h) for rr in earlier_same_bucket)
            err = abs(manual - r[tcol])
            max_err = max(max_err, err)
            assert err < 1e-6, f"CAUSALITY BUG: decayed_dur_total mismatch h={h} idx={idx} manual={manual} got={r[tcol]}"
    print(f"30 random rows x {len(DUR_HALFLIVES)} dur-halflives: decayed_dur_total matches brute force "
          f"(max abs err {max_err:.2e}). No leakage detected.")

    bad = 0
    for h in DUR_HALFLIVES:
        tot_col = _dur_halflife_total_col(h)
        pos_col = _dur_halflife_col(h)
        for r in flat[:5000]:
            if r[tot_col] + 1e-9 < r[pos_col]:
                bad += 1
    assert bad == 0, f"CAUSALITY BUG: found {bad} rows where decayed_dur_total < decayed_dur_pos!"
    print(f"sanity (5000 rows x {len(DUR_HALFLIVES)} halflives): decayed_dur_total >= decayed_dur_pos everywhere. OK.")

    print("\nAll causal spot-checks passed.")
