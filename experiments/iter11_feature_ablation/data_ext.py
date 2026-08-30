"""iter9: coarse-grained causal history features (denser than iter6's per-author
affinity, which was only nonzero for 0.70% of rows — see LEDGER.md iter6).

Reuses ../../data.py's load()/SPLITS unmodified; does NOT modify data.py.

Three candidate causal features, all computed with the exact same
strict-causal date-grouped pattern iter6 validated:
  - combine train+valid+test raw rows into one flat "global timeline"
  - sort by date, process one date-group at a time
  - for every row in a date group, first READ the counter state as it stood
    BEFORE that date (so same-date rows never see each other or themselves —
    strict `<`, never `<=`)
  - only AFTER all rows in the date group have been read does the group's
    label==1 rows get folded into the counters (so the *next* date sees them)
Because SPLITS' date ranges are non-overlapping and monotonic
(train < valid < test), this single combined-timeline pass automatically
gives correct causal semantics for every split with no special-casing.

Feature 1 — `activity`: count of this user's PRIOR rows (any label) seen so
  far anywhere in the log. Pure "how active/experienced is this user" signal.
Feature 2 — `tab_pos`: count of this user's prior label==1 rows within the
  SAME `tab` value as the current row. `tab` has only 15 distinct values
  (vs. tens of thousands of authors), so this should have much higher
  coverage than iter6's per-author affinity.
Feature 3 — `rate`: this user's prior positive rate so far
  (prior_pos + alpha) / (prior_total + 2*alpha), Laplace-smoothed with
  alpha=1.0 so early-history users get a sensible prior (0.5) instead of a
  degenerate 0/0.

All three are computed in a single pass (compute_causal_features) since they
share the same date-grouped traversal.
"""
import os, sys, collections
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from data import load as _load_raw  # noqa: E402  (reuse original raw loader, untouched)

LABEL = 'long_view'
BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
ALPHA = 1.0  # Laplace smoothing constant for feature 3


def compute_causal_features(rows):
    """rows: list of (date, user_id, video_id, author_id, tab, duration_ms, label),
    in ANY order. Returns dict of lists aligned index-for-index with `rows`:
      'activity'    : count of user's strictly-earlier rows (any label)
      'tab_pos'     : count of user's strictly-earlier label==1 rows, same tab
      'prior_pos'   : count of user's strictly-earlier label==1 rows (any tab)
      'prior_total' : same as 'activity' (kept as separate key for clarity,
                      used as the denominator base for feature 3's rate)
    Same-date rows never count each other (strict `<`, not `<=`) — see the
    module docstring for the two-phase read-then-update date-group pattern.
    """
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
        # 1) read pre-day-boundary counter state for every row in this date group
        for idx in day_idx:
            r = rows[idx]
            u, tab = r[1], r[4]
            activity[idx] = user_total[u]
            prior_pos[idx] = user_pos[u]
            tab_pos[idx] = user_tab_pos[(u, tab)]
        # 2) only now fold this date group's rows into the counters
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


def load_ext(data_dir):
    """Returns dict of (train/valid/test) -> list of extended rows:
    (date, user_id, video_id, author_id, tab, duration_ms, label,
     activity, tab_pos, prior_pos, prior_total)
    i.e. original 7-tuple + 4 raw (unbucketed) causal feature values."""
    splits = _load_raw(data_dir)
    order = ('train', 'valid', 'test')
    flat, owner = [], []
    for name in order:
        for r in splits[name]:
            flat.append(r)
            owner.append(name)

    feats = compute_causal_features(flat)

    ext = {name: [] for name in order}
    for i, (r, name) in enumerate(zip(flat, owner)):
        ext[name].append(r + (feats['activity'][i], feats['tab_pos'][i],
                               feats['prior_pos'][i], feats['prior_total'][i]))
    return ext


def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1])


def encode_ext(splits, feature_set=('activity',), alpha=ALPHA):
    """splits: dict from load_ext(), each row is an 11-tuple as documented above.
    feature_set: subset/order of {'activity', 'tab', 'rate'} — which extra
    causal fields to append after the base 5 (user_id, video_id, author_id,
    tab, dur_bucket). Mirrors data.py's encode() exactly for the base 5
    fields (dur_bucket via train-fit quantile edges); each requested extra
    field gets its own train-fit quantile bucketing (10 buckets, same
    pattern as dur_bucket) EXCEPT 'rate' which is Laplace-smoothed first.
    alpha: Laplace smoothing constant used in 'rate's formula
    (prior_pos+alpha)/(prior_total+2*alpha). iter11 addition — parametrized
    out of the module-level ALPHA constant so it can be swept; defaults to
    the original iter9 value (1.0) when not overridden.
    Returns (enc, field_dims_sum) with enc[name] = (X, y, users)."""
    tr = splits['train']
    dur_edges = _bucket_edges([x[5] for x in tr])

    def raw_dur(x):
        return str(int(np.searchsorted(dur_edges, x[5])))

    extra_edges = {}
    if 'activity' in feature_set:
        extra_edges['activity'] = _bucket_edges([x[7] for x in tr])
    if 'tab' in feature_set:
        extra_edges['tab'] = _bucket_edges([x[8] for x in tr])
    if 'rate' in feature_set:
        def rate_of(x):
            return (x[9] + alpha) / (x[10] + 2 * alpha)
        extra_edges['rate'] = _bucket_edges([rate_of(x) for x in tr])

    def extra_val(x, name):
        if name == 'activity':
            return str(int(np.searchsorted(extra_edges['activity'], x[7])))
        elif name == 'tab':
            return str(int(np.searchsorted(extra_edges['tab'], x[8])))
        elif name == 'rate':
            r = (x[9] + alpha) / (x[10] + 2 * alpha)
            return str(int(np.searchsorted(extra_edges['rate'], r)))
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
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    ext = load_ext(a.data_dir)
    print({k: len(v) for k, v in ext.items()})

    flat = ext['train'] + ext['valid'] + ext['test']
    n = len(flat)

    # ---- coverage report ----
    activity = np.array([r[7] for r in flat])
    tab_pos = np.array([r[8] for r in flat])
    prior_pos = np.array([r[9] for r in flat])
    prior_total = np.array([r[10] for r in flat])
    print(f"\ntotal rows: {n}")
    print(f"feature 1 (activity)         nonzero coverage: {np.mean(activity > 0)*100:.2f}%  "
          f"({int(np.sum(activity > 0))}/{n})")
    print(f"feature 2 (tab_pos)          nonzero coverage: {np.mean(tab_pos > 0)*100:.2f}%  "
          f"({int(np.sum(tab_pos > 0))}/{n})")
    print(f"feature 3 (prior_total>0, i.e. user has ANY prior history for a smoothed rate "
          f"that differs from the 0.5 default): {np.mean(prior_total > 0)*100:.2f}%  "
          f"({int(np.sum(prior_total > 0))}/{n})")

    # ---- causality spot-checks (same methodology as iter6) ----
    print("\n--- causal spot-checks: feature 1 (activity) ---")
    examples = [r for r in flat if r[7] >= 5][:5]
    for r in examples:
        date, uid, vid, aid, tab, dur, label, act, tabp, pp, pt = r
        manual = sum(1 for rr in flat if rr[1] == uid and rr[0] < date)
        print(f"date={date} user={uid} activity={act}  manual(rows with date<{date})={manual}")
        assert manual == act, "CAUSALITY BUG: activity recount mismatch!"
    zero_examples = [r for r in flat if r[7] == 0][:3]
    for r in zero_examples:
        date, uid, vid, aid, tab, dur, label, act, tabp, pp, pt = r
        manual = sum(1 for rr in flat if rr[1] == uid and rr[0] < date)
        assert manual == 0, "CAUSALITY BUG: activity==0 row has earlier rows!"
    print("activity: all spot-checks passed.")

    print("\n--- causal spot-checks: feature 2 (tab_pos) ---")
    examples = [r for r in flat if r[8] >= 5][:5]
    for r in examples:
        date, uid, vid, aid, tab, dur, label, act, tabp, pp, pt = r
        manual = sum(1 for rr in flat
                     if rr[1] == uid and rr[4] == tab and rr[0] < date and rr[6] == 1)
        print(f"date={date} user={uid} tab={tab} tab_pos={tabp}  manual={manual}")
        assert manual == tabp, "CAUSALITY BUG: tab_pos recount mismatch!"
    # same-date-pair edge case: find a (user, tab) with >=2 positives on the same date
    by_ut_date = collections.defaultdict(list)
    for idx, r in enumerate(flat):
        if r[6] == 1:
            by_ut_date[(r[1], r[4], r[0])].append(idx)
    same_date_case = next((v for v in by_ut_date.values() if len(v) >= 2), None)
    if same_date_case:
        print("same-date-pair edge case (should both show tab_pos excluding each other):")
        for idx in same_date_case:
            r = flat[idx]
            print(f"  user={r[1]} tab={r[4]} date={r[0]} label={r[6]} tab_pos={r[8]}")
    print("tab_pos: all spot-checks passed.")

    print("\n--- causal spot-checks: feature 3 (prior_pos / prior_total) ---")
    examples = [r for r in flat if r[10] >= 5][:5]
    for r in examples:
        date, uid, vid, aid, tab, dur, label, act, tabp, pp, pt = r
        manual_total = sum(1 for rr in flat if rr[1] == uid and rr[0] < date)
        manual_pos = sum(1 for rr in flat if rr[1] == uid and rr[0] < date and rr[6] == 1)
        assert manual_total == pt and manual_pos == pp, "CAUSALITY BUG: prior_pos/total mismatch!"
    print("prior_pos/prior_total: all spot-checks passed.")

    print("\nAll causal spot-checks passed. No same-date or future leakage detected.")
