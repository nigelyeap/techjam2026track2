"""iter12: ITEM-side causal history features, complementing iter9's USER-side
set (activity, tab_pos, rate).

iter9 established that coarsening the causal-history idea to per-user
granularity (92% coverage) turns a dead per-author-affinity signal (iter6,
0.70% coverage) into the single best lever found so far. iter9's features are
all about the USER (how active they are, their historical rate). This
iteration asks the complementary question: does knowing how popular/well-
received the VIDEO or AUTHOR has been so far (causally) add anything on top?

Two new candidate features, computed via the EXACT same strict-causal
(`<`, never `<=`) two-phase date-grouped traversal pattern iter6/iter9
validated (see module docstring in iter9's data_ext.py for the full
rationale) — reused here unmodified in spirit, just extended to also track
per-video and per-author counters in the same pass:

  - `video_pop`: count of this `video_id`'s PRIOR positive (long_view==1)
    rows, anywhere, before this row's date. Pure item popularity signal —
    NOT per-user, so it should have much higher coverage than iter6's
    per-(user,author) affinity pair, since many different users can and do
    contribute to a single video's prior-positive count.
  - `author_rate`: this `author_id`'s Laplace-smoothed prior positive rate,
    aggregated over ALL of that author's prior rows from ANY user
    (`(prior_pos+1)/(prior_total+2)`, same smoothing formula as iter9's user
    `rate`). This is explicitly NOT iter6's per-USER-per-author affinity
    (which failed at 0.70% coverage due to sparsity of any single user
    re-encountering the same author) — it's the author's own aggregate rate
    across the whole user population, which should be much denser since
    popular authors accumulate views from many different users.

Also keeps iter9's three user-side features (activity, tab_pos, rate)
available in the same combined pass, so this module can be used standalone
to test item-only feature sets AND combined user+item sets without needing
two separate passes over the data.

Reuses ../../data.py's load()/SPLITS unmodified; does NOT modify data.py.
"""
import os, sys, collections
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from data import load as _load_raw  # noqa: E402  (reuse original raw loader, untouched)

LABEL = 'long_view'
BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
ALPHA = 1.0  # Laplace smoothing constant, same as iter9


def compute_causal_features(rows):
    """rows: list of (date, user_id, video_id, author_id, tab, duration_ms, label),
    in ANY order. Single pass, two-phase per-date-group (read-then-update),
    strict `<` date comparison — same pattern validated in iter6/iter9.

    Returns dict of lists aligned index-for-index with `rows`:
      USER-side (iter9, reused unmodified for combo sweeps):
        'activity'    : count of user's strictly-earlier rows (any label)
        'tab_pos'     : count of user's strictly-earlier label==1 rows, same tab
        'prior_pos'   : count of user's strictly-earlier label==1 rows (any tab)
        'prior_total' : same as 'activity'
      ITEM-side (new this iteration):
        'video_pop'        : count of this video_id's strictly-earlier
                              label==1 rows (any user)
        'author_prior_pos' : count of this author_id's strictly-earlier
                              label==1 rows (any user, any video)
        'author_prior_total': count of this author_id's strictly-earlier
                              rows (any label) — denominator base for
                              author_rate's Laplace smoothing
    """
    n = len(rows)
    order = sorted(range(n), key=lambda i: rows[i][0])
    activity = [0] * n
    tab_pos = [0] * n
    prior_pos = [0] * n
    video_pop = [0] * n
    author_prior_pos = [0] * n
    author_prior_total = [0] * n

    user_total = collections.defaultdict(int)
    user_pos = collections.defaultdict(int)
    user_tab_pos = collections.defaultdict(int)
    video_pos_ctr = collections.defaultdict(int)
    author_total_ctr = collections.defaultdict(int)
    author_pos_ctr = collections.defaultdict(int)

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
            u, vid, aid, tab = r[1], r[2], r[3], r[4]
            activity[idx] = user_total[u]
            prior_pos[idx] = user_pos[u]
            tab_pos[idx] = user_tab_pos[(u, tab)]
            video_pop[idx] = video_pos_ctr[vid]
            author_prior_pos[idx] = author_pos_ctr[aid]
            author_prior_total[idx] = author_total_ctr[aid]
        # 2) only now fold this date group's rows into the counters
        for idx in day_idx:
            r = rows[idx]
            u, vid, aid, tab, label = r[1], r[2], r[3], r[4], r[6]
            user_total[u] += 1
            author_total_ctr[aid] += 1
            if label == 1:
                user_pos[u] += 1
                user_tab_pos[(u, tab)] += 1
                video_pos_ctr[vid] += 1
                author_pos_ctr[aid] += 1
        i = j
    return {'activity': activity, 'tab_pos': tab_pos, 'prior_pos': prior_pos,
            'prior_total': activity,
            'video_pop': video_pop,
            'author_prior_pos': author_prior_pos,
            'author_prior_total': author_prior_total}


def load_ext(data_dir):
    """Returns dict of (train/valid/test) -> list of extended rows:
    (date, user_id, video_id, author_id, tab, duration_ms, label,
     activity, tab_pos, prior_pos, prior_total,
     video_pop, author_prior_pos, author_prior_total)
    i.e. original 7-tuple + 4 user-side (iter9) + 3 item-side (new) raw
    (unbucketed) causal feature values."""
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
        ext[name].append(r + (
            feats['activity'][i], feats['tab_pos'][i],
            feats['prior_pos'][i], feats['prior_total'][i],
            feats['video_pop'][i], feats['author_prior_pos'][i],
            feats['author_prior_total'][i]))
    return ext


def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1])


def encode_ext(splits, feature_set=('activity',)):
    """splits: dict from load_ext(), each row is a 14-tuple as documented
    above. feature_set: subset/order of
    {'activity', 'tab', 'rate', 'video_pop', 'author_rate'} — which extra
    causal fields to append after the base 5 (user_id, video_id, author_id,
    tab, dur_bucket). Mirrors data.py's encode() exactly for the base 5
    fields (dur_bucket via train-fit quantile edges); each requested extra
    field gets its own train-fit quantile bucketing (10 buckets, same
    pattern as dur_bucket) EXCEPT 'rate'/'author_rate' which are
    Laplace-smoothed first.
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
            return (x[9] + ALPHA) / (x[10] + 2 * ALPHA)
        extra_edges['rate'] = _bucket_edges([rate_of(x) for x in tr])
    if 'video_pop' in feature_set:
        extra_edges['video_pop'] = _bucket_edges([x[11] for x in tr])
    if 'author_rate' in feature_set:
        def author_rate_of(x):
            return (x[12] + ALPHA) / (x[13] + 2 * ALPHA)
        extra_edges['author_rate'] = _bucket_edges([author_rate_of(x) for x in tr])

    def extra_val(x, name):
        if name == 'activity':
            return str(int(np.searchsorted(extra_edges['activity'], x[7])))
        elif name == 'tab':
            return str(int(np.searchsorted(extra_edges['tab'], x[8])))
        elif name == 'rate':
            r = (x[9] + ALPHA) / (x[10] + 2 * ALPHA)
            return str(int(np.searchsorted(extra_edges['rate'], r)))
        elif name == 'video_pop':
            return str(int(np.searchsorted(extra_edges['video_pop'], x[11])))
        elif name == 'author_rate':
            r = (x[12] + ALPHA) / (x[13] + 2 * ALPHA)
            return str(int(np.searchsorted(extra_edges['author_rate'], r)))
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
    video_pop = np.array([r[11] for r in flat])
    author_prior_pos = np.array([r[12] for r in flat])
    author_prior_total = np.array([r[13] for r in flat])
    print(f"\ntotal rows: {n}")
    print(f"video_pop            nonzero coverage: {np.mean(video_pop > 0)*100:.2f}%  "
          f"({int(np.sum(video_pop > 0))}/{n})")
    print(f"author_rate (author_prior_total>0, i.e. author has ANY prior history for a "
          f"smoothed rate that differs from the 0.5 default): "
          f"{np.mean(author_prior_total > 0)*100:.2f}%  "
          f"({int(np.sum(author_prior_total > 0))}/{n})")
    print(f"author_rate (author_prior_pos>0, i.e. author has ANY prior POSITIVE): "
          f"{np.mean(author_prior_pos > 0)*100:.2f}%  "
          f"({int(np.sum(author_prior_pos > 0))}/{n})")

    if np.mean(video_pop > 0) < 0.05:
        print("*** FLAG: video_pop coverage < 5% -- likely too sparse (iter6-style failure mode)")
    if np.mean(author_prior_total > 0) < 0.05:
        print("*** FLAG: author_rate coverage < 5% -- likely too sparse (iter6-style failure mode)")

    # ---- causality spot-checks (same methodology as iter6/iter9) ----
    print("\n--- causal spot-checks: video_pop ---")
    examples = [r for r in flat if r[11] >= 3][:5]
    for r in examples:
        date, uid, vid, aid, tab, dur, label = r[0:7]
        vp = r[11]
        manual = sum(1 for rr in flat if rr[2] == vid and rr[0] < date and rr[6] == 1)
        print(f"date={date} video={vid} video_pop={vp}  manual(prior positive rows for video, "
              f"date<{date})={manual}")
        assert manual == vp, "CAUSALITY BUG: video_pop recount mismatch!"
    zero_examples = [r for r in flat if r[11] == 0][:3]
    for r in zero_examples:
        date, uid, vid, aid, tab, dur, label = r[0:7]
        manual = sum(1 for rr in flat if rr[2] == vid and rr[0] < date and rr[6] == 1)
        assert manual == 0, "CAUSALITY BUG: video_pop==0 row has earlier positive rows!"
    print("video_pop: all spot-checks passed.")

    print("\n--- causal spot-checks: author_rate (author_prior_pos / author_prior_total) ---")
    examples = [r for r in flat if r[13] >= 5][:5]
    for r in examples:
        date, uid, vid, aid, tab, dur, label = r[0:7]
        ap_, at_ = r[12], r[13]
        manual_total = sum(1 for rr in flat if rr[3] == aid and rr[0] < date)
        manual_pos = sum(1 for rr in flat if rr[3] == aid and rr[0] < date and rr[6] == 1)
        print(f"date={date} author={aid} author_prior_pos={ap_} manual_pos={manual_pos} "
              f"author_prior_total={at_} manual_total={manual_total}")
        assert manual_total == at_ and manual_pos == ap_, \
            "CAUSALITY BUG: author_prior_pos/total mismatch!"
    print("author_rate: all spot-checks passed.")

    # same-date-pair edge case: find a video with >=2 positive rows on the same date
    print("\n--- same-date-pair edge case (video_pop) ---")
    by_vid_date = collections.defaultdict(list)
    for idx, r in enumerate(flat):
        if r[6] == 1:
            by_vid_date[(r[2], r[0])].append(idx)
    same_date_case = next((v for v in by_vid_date.values() if len(v) >= 2), None)
    if same_date_case:
        print("found video with >=2 same-date positives (should NOT count each other):")
        for idx in same_date_case:
            r = flat[idx]
            print(f"  video={r[2]} date={r[0]} label={r[6]} video_pop={r[11]}")
        vp0 = flat[same_date_case[0]][11]
        for idx in same_date_case:
            assert flat[idx][11] == vp0, "same-date rows for same video should show IDENTICAL " \
                "video_pop (neither counts the other, nor itself)"
        # cross-check against a strictly-prior-date manual count
        r0 = flat[same_date_case[0]]
        manual = sum(1 for rr in flat if rr[2] == r0[2] and rr[0] < r0[0] and rr[6] == 1)
        assert manual == r0[11], "same-date-pair edge case: video_pop should equal count of " \
            "STRICTLY EARLIER-DATE positives only"
        print("same-date-pair edge case passed: same-date positives do not count each other.")
    else:
        print("no same-date multi-positive video found in this dataset to test; skipping.")

    # same-date-pair edge case for author too
    print("\n--- same-date-pair edge case (author_rate) ---")
    by_aid_date = collections.defaultdict(list)
    for idx, r in enumerate(flat):
        if r[6] == 1:
            by_aid_date[(r[3], r[0])].append(idx)
    same_date_case2 = next((v for v in by_aid_date.values() if len(v) >= 2), None)
    if same_date_case2:
        ap0 = flat[same_date_case2[0]][12]
        for idx in same_date_case2:
            assert flat[idx][12] == ap0, "same-date rows for same author should show IDENTICAL " \
                "author_prior_pos (neither counts the other)"
        r0 = flat[same_date_case2[0]]
        manual = sum(1 for rr in flat if rr[3] == r0[3] and rr[0] < r0[0] and rr[6] == 1)
        assert manual == r0[12], "same-date-pair edge case: author_prior_pos should equal count " \
            "of STRICTLY EARLIER-DATE positives only"
        print("same-date-pair edge case passed for author_rate.")
    else:
        print("no same-date multi-positive author found; skipping.")

    print("\nAll causal spot-checks passed. No same-date or future leakage detected.")
