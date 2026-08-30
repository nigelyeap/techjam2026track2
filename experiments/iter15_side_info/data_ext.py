"""iter15: static side-info features from two dataset files never used
anywhere else in this run: `user_features_pure.csv` (one row/user_id,
demographic/account-state fields) and `video_features_statistic_pure.csv`
(one row/video_id, aggregate engagement counters).

Reuses ../../data.py's load() unmodified. Reuses iter9's validated causal
history-feature traversal verbatim (copied, not cross-imported, to keep this
directory self-contained like every other iterN dir) so `{activity,tab,rate}`
can be re-derived here as the iter9 baseline reference point, and so all four
sweep configs share identical base features except for the side-info addition.

*** LEAKAGE CAVEAT (see RESULT.md for full discussion) ***
Unlike iter9's causal features (computed with strict `<` date-comparison
traversal over the interaction log, so a row can only see counts from
strictly-earlier dates), these two side-info files are static per-entity
tables with NO date/timestamp column. They almost certainly describe
user/video state aggregated over some fixed window that is NOT guaranteed to
respect the train/valid/test date boundary (may include future information
relative to a given training-split row, or may even be a corpus-wide
snapshot). This is standard practice for public KuaiRand baselines (these
files are typically treated as static side info describing inherent
user/video characteristics rather than time-varying signals), but it is NOT
causally clean the way iter9's own features are proven to be.
- `user_features_pure.csv` fields used here (user_active_degree,
  is_live_streamer, is_video_author, follow/fans_user_num_range,
  register_days_range) are mostly slow-moving demographic/account-state
  attributes -> LOWER leakage risk.
- `video_features_statistic_pure.csv` fields used here (play_cnt, like_cnt,
  share_cnt, complete_play_cnt, follow_cnt) are aggregate ENGAGEMENT COUNTS
  that could plausibly be computed over a window including the eval period,
  or could just directly encode "this video is popular" in a way that
  correlates suspiciously well with long_view on the SAME rows that produced
  those counts -> HIGHER leakage risk.
This is exactly why the task evaluates user-side, video-side, and combined
side-info separately: to let a reader see how much of any gain depends on
the riskier file.
"""
import os, sys, csv, collections
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from data import load as _load_raw  # noqa: E402  (reuse original raw loader, untouched)

LABEL = 'long_view'
BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
ALPHA = 1.0  # Laplace smoothing constant for the 'rate' causal feature (matches iter9)

CAUSAL_FEATURES = ('activity', 'tab', 'rate')

# ---- user-side static fields (demographic/account-state; lower leak risk) ----
# Selected from user_features_pure.csv's 12 non-onehot columns. Dropped
# is_lowactive_period (constant == '0' for all 27,285 users in this file --
# zero information) and friend_user_num_range (redundant with
# follow/fans_user_num_range, kept feature count reasonable). The 18
# onehot_feat* columns are anonymized/undocumented -- left unused, out of
# scope for "a reasonable subset of useful-looking columns".
USER_FIELDS = ['u_active_degree', 'u_live_streamer', 'u_video_author',
               'u_follow_range', 'u_fans_range', 'u_register_range']
USER_CSV_COLS = ['user_active_degree', 'is_live_streamer', 'is_video_author',
                  'follow_user_num_range', 'fans_user_num_range', 'register_days_range']

# ---- video-side static fields (aggregate engagement counts; higher leak risk) ----
# 5 of the ~50 raw counters in video_features_statistic_pure.csv, chosen as
# the most directly relevant to "is this video good/popular" per the task's
# suggested list. Bucketed into quantiles fit on TRAIN split only (matches
# `data.py`'s `dur_bucket` pattern) since these are continuous/skewed counts.
VIDEO_FIELDS = ['v_play', 'v_like', 'v_share', 'v_complete', 'v_follow']
VIDEO_CSV_COLS = ['play_cnt', 'like_cnt', 'share_cnt', 'complete_play_cnt', 'follow_cnt']

UNK_USER_SENTINEL = '__UNK_USER_SIDEINFO__'
UNK_VIDEO_SENTINEL = None  # numeric fields: None means "no side-info row for this video"


def _load_user_side_info(data_dir):
    """user_id -> tuple of raw category strings (len == len(USER_FIELDS))."""
    lut = {}
    path = os.path.join(data_dir, 'user_features_pure.csv')
    with open(path) as fh:
        for r in csv.DictReader(fh):
            lut[r['user_id']] = tuple(r[c] for c in USER_CSV_COLS)
    return lut


def _load_video_side_info(data_dir):
    """video_id -> tuple of raw floats (len == len(VIDEO_FIELDS))."""
    lut = {}
    path = os.path.join(data_dir, 'video_features_statistic_pure.csv')
    with open(path) as fh:
        for r in csv.DictReader(fh):
            lut[r['video_id']] = tuple(float(r[c]) for c in VIDEO_CSV_COLS)
    return lut


def compute_causal_features(rows):
    """Verbatim copy of iter9_history_dense/data_ext.py::compute_causal_features.
    rows: list of (date, user_id, video_id, author_id, tab, duration_ms, label).
    Returns dict of lists aligned index-for-index with `rows`:
      'activity', 'tab_pos', 'prior_pos', 'prior_total'.
    Strict `<` (never `<=`) two-phase date-grouped traversal: same-date rows
    never count each other. See iter9's RESULT.md / module docstring for the
    causality proof and brute-force spot-check methodology (re-verified below
    in this file's __main__ block for this copy too).
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


def load_ext(data_dir):
    """Returns dict of (train/valid/test) -> list of extended rows:
    (date, user_id, video_id, author_id, tab, duration_ms, label,
     activity, tab_pos, prior_pos, prior_total,
     u_active_degree, u_live_streamer, u_video_author, u_follow_range,
     u_fans_range, u_register_range,
     v_play, v_like, v_share, v_complete, v_follow)
    i.e. original 7-tuple + 4 causal raw values + 6 user raw strings + 5 video
    raw floats-or-None. UNK/missing handling: any user_id absent from
    user_features_pure.csv gets UNK_USER_SENTINEL for all 6 user fields (a
    string that never appears as a real value, so it always falls in the
    trained vocab's UNK slot); any video_id absent from
    video_features_statistic_pure.csv gets None for all 5 video fields
    (mapped to a dedicated 'UNK' bucket string at encode time, distinct from
    every quantile-bucket index).
    """
    splits = _load_raw(data_dir)
    order = ('train', 'valid', 'test')
    flat, owner = [], []
    for name in order:
        for r in splits[name]:
            flat.append(r)
            owner.append(name)

    feats = compute_causal_features(flat)
    user_lut = _load_user_side_info(data_dir)
    video_lut = _load_video_side_info(data_dir)

    user_default = (UNK_USER_SENTINEL,) * len(USER_FIELDS)
    video_default = (None,) * len(VIDEO_FIELDS)

    n_user_missing = 0
    n_video_missing = 0

    ext = {name: [] for name in order}
    for i, (r, name) in enumerate(zip(flat, owner)):
        u_tuple = user_lut.get(r[1])
        if u_tuple is None:
            u_tuple = user_default
            n_user_missing += 1
        v_tuple = video_lut.get(r[2])
        if v_tuple is None:
            v_tuple = video_default
            n_video_missing += 1
        ext[name].append(r + (feats['activity'][i], feats['tab_pos'][i],
                               feats['prior_pos'][i], feats['prior_total'][i])
                          + u_tuple + v_tuple)

    if n_user_missing or n_video_missing:
        print(f"  [data_ext] side-info join: {n_user_missing}/{len(flat)} rows with "
              f"UNK user_id, {n_video_missing}/{len(flat)} rows with UNK video_id "
              f"(fell back to UNK bucket)")
    return ext


def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1])


# column index (within the extended row tuple) for each named feature's raw source
_ROW_IDX = {
    'activity': 7, 'tab_pos': 8,  # 'rate' derived from (9, 10) = (prior_pos, prior_total)
    'u_active_degree': 11, 'u_live_streamer': 12, 'u_video_author': 13,
    'u_follow_range': 14, 'u_fans_range': 15, 'u_register_range': 16,
    'v_play': 17, 'v_like': 18, 'v_share': 19, 'v_complete': 20, 'v_follow': 21,
}
_USER_CAT_FIELDS = set(USER_FIELDS)
_VIDEO_NUM_FIELDS = set(VIDEO_FIELDS)


def encode_ext(splits, feature_set=('activity', 'tab', 'rate')):
    """splits: dict from load_ext(), each row is a 22-tuple as documented above.
    feature_set: subset/order of {'activity','tab','rate'} union USER_FIELDS
    union VIDEO_FIELDS. Mirrors data.py's encode() for the base 5 fields
    (dur_bucket via train-fit quantile edges).
      - 'activity'/'tab': train-fit quantile-bucketed (10 buckets), same as iter9.
      - 'rate': Laplace-smoothed prior_pos/prior_total, then train-fit
        quantile-bucketed (10 buckets), same as iter9.
      - USER_FIELDS: used directly as raw categorical strings (already
        discrete in the source CSV); UNK sentinel rows fall to the field's
        UNK vocab slot automatically.
      - VIDEO_FIELDS: train-fit quantile-bucketed (10 buckets) on non-missing
        values only; missing (None) rows get a dedicated 'UNK' bucket string.
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
    for vf in VIDEO_FIELDS:
        if vf in feature_set:
            idx = _ROW_IDX[vf]
            vals = [x[idx] for x in tr if x[idx] is not None]
            extra_edges[vf] = _bucket_edges(vals)

    def extra_val(x, name):
        if name == 'activity':
            return str(int(np.searchsorted(extra_edges['activity'], x[7])))
        elif name == 'tab':
            return str(int(np.searchsorted(extra_edges['tab'], x[8])))
        elif name == 'rate':
            r = (x[9] + ALPHA) / (x[10] + 2 * ALPHA)
            return str(int(np.searchsorted(extra_edges['rate'], r)))
        elif name in _USER_CAT_FIELDS:
            return x[_ROW_IDX[name]]
        elif name in _VIDEO_NUM_FIELDS:
            v = x[_ROW_IDX[name]]
            if v is None:
                return 'UNK'
            return str(int(np.searchsorted(extra_edges[name], v)))
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

    # ---- causal feature spot-checks (re-verify this copy matches iter9's proven logic) ----
    print("\n--- causal spot-checks (copied from iter9, re-verified for this copy) ---")
    activity = np.array([r[7] for r in flat])
    examples = [r for r in flat if r[7] >= 5][:5]
    for r in examples:
        date, uid = r[0], r[1]
        manual = sum(1 for rr in flat if rr[1] == uid and rr[0] < date)
        assert manual == r[7], "CAUSALITY BUG: activity recount mismatch!"
    print("activity: spot-checks passed.")
    by_ut_date = collections.defaultdict(list)
    for idx, r in enumerate(flat):
        if r[6] == 1:
            by_ut_date[(r[1], r[4], r[0])].append(idx)
    same_date_case = next((v for v in by_ut_date.values() if len(v) >= 2), None)
    if same_date_case:
        print("same-date-pair edge case (tab_pos should exclude each other):")
        for idx in same_date_case:
            r = flat[idx]
            print(f"  user={r[1]} tab={r[4]} date={r[0]} label={r[6]} tab_pos={r[8]}")
    print(f"nonzero coverage: activity {np.mean(activity>0)*100:.2f}%")

    # ---- side-info join coverage ----
    n_user_unk = sum(1 for r in flat if r[11] == UNK_USER_SENTINEL)
    n_video_unk = sum(1 for r in flat if r[17] is None)
    print(f"\nuser side-info UNK coverage: {n_user_unk}/{n} ({100*n_user_unk/n:.3f}%)")
    print(f"video side-info UNK coverage: {n_video_unk}/{n} ({100*n_video_unk/n:.3f}%)")

    print("\nuser field value samples:")
    for i, f in enumerate(USER_FIELDS):
        vals = set(r[_ROW_IDX[f]] for r in flat[:2000])
        print(f"  {f}: e.g. {list(vals)[:5]}")
    print("video field value samples:")
    for f in VIDEO_FIELDS:
        vals = [r[_ROW_IDX[f]] for r in flat[:2000] if r[_ROW_IDX[f]] is not None]
        print(f"  {f}: min={min(vals):.2f} max={max(vals):.2f} mean={np.mean(vals):.2f}")
