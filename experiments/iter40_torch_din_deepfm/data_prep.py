"""iter40: causal user-history sequences for an end-to-end differentiable
DIN-style attention layer, on top of iter27's proven feature set.

Reuses iter27's `load_ext`/`encode_ext`/`compute_final_decayed_pos` VERBATIM
(imported via importlib to avoid the same-named-module collision this repo
has hit before -- see iter18/iter32's own docstrings) for every already-
validated piece: the decay/momentum engineered features, their causal
traversal, and the BPR user-sampling weight. This module adds exactly one
new thing: fixed-length, causally-correct, per-row history sequences of a
user's own past (video_id, author_id, label) interactions, for a target-
attention layer to consume.

Causality: identical discipline to iter18's momentum features and iter32's
attention feature -- per user, sort all of that user's rows (across every
split) by (time_ms, orig_idx) to get one strict total order, walk it once,
and read a row's history from a deque that is only updated with that row's
OWN (video, author, label) AFTER its history has already been read. A row
can never see itself or any later row, even under a time_ms tie.
"""
import os, sys, collections
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ITER27_DIR = os.path.join(_THIS_DIR, '..', 'iter27_triple_fusion')


def _load_iter27_data_ext():
    import importlib.util
    path = os.path.join(_ITER27_DIR, 'data_ext.py')
    spec = importlib.util.spec_from_file_location('iter27_data_ext', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_de = _load_iter27_data_ext()

FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')
L_HIST = 40  # matches iter32's winning attention window


def build_history(splits, L=L_HIST):
    """splits: dict from iter27's load_ext() -- {'train','valid','test'} ->
    list of extended row tuples (IDX-indexed, see iter27/data_ext.py).
    Returns dict split -> (hist_video_localrow, hist_author_localrow,
    hist_label, hist_mask), each array shaped (n_rows_in_split, L), aligned
    index-for-index with `splits[name]` (and therefore with encode_ext's
    per-split X/y/users, since encode_ext iterates `splits.items()` the same
    way). Values are LOCAL ROW INDICES into the flat (train+valid+test)
    concatenation -- the caller resolves them to embedding ids via the X
    matrix's video_id/author_id columns, so this module has zero dependency
    on encode_ext's vocab internals.
    """
    order = ('train', 'valid', 'test')
    flat = []
    split_bounds = {}
    for name in order:
        start = len(flat)
        flat.extend(splits[name])
        split_bounds[name] = (start, len(flat))

    IDX = _de.IDX
    n = len(flat)
    hist_row = np.full((n, L), -1, dtype=np.int64)   # -1 = pad sentinel
    hist_mask = np.zeros((n, L), dtype=np.float32)

    by_user = collections.defaultdict(list)
    for i, r in enumerate(flat):
        by_user[r[IDX['user_id']]].append(i)

    for u, idxs in by_user.items():
        idxs.sort(key=lambda i: (flat[i][IDX['time_ms']], flat[i][IDX['orig_idx']]))
        window = collections.deque(maxlen=L)
        for i in idxs:
            h = list(window)  # oldest-first, length <= L
            k = len(h)
            if k:
                hist_row[i, L - k:] = h
                hist_mask[i, L - k:] = 1.0
            window.append(i)

    out = {}
    for name in order:
        lo, hi = split_bounds[name]
        out[name] = (hist_row[lo:hi], hist_mask[lo:hi])
    return out, split_bounds, flat


def prepare(data_dir, feature_set=FEATURES, halflives=_de.HALFLIVES,
            tab_halflives=_de.TAB_HALFLIVES, alpha=0.5, n_buckets=20,
            L=L_HIST, use_cache=True):
    """Returns:
      enc: {'train','valid','test': (X, y, users)}   -- iter27's encode_ext output
      dim: total embedding-table size (does NOT include the PAD row; caller
           should size the embedding table dim+1 and use `dim` as PAD_IDX)
      hist: {'train','valid','test': (hist_video_id, hist_author_id, hist_label, hist_mask)}
            each (n_rows, L) int64/int64/float32/float32, using GLOBAL embedding
            ids for hist_video_id/hist_author_id (dim = PAD_IDX for padding),
            0.0 for hist_label at padding.
      decayed_pos_dict: user_id -> decayed positive count (iter23's BPR
            sampling weight, computed on train only, unchanged from iter27).
    """
    splits = _de.load_ext(data_dir, halflives=halflives, tab_halflives=tab_halflives, use_cache=use_cache)
    enc, dim = _de.encode_ext(splits, feature_set=feature_set, halflives=halflives,
                               tab_halflives=tab_halflives, alpha=alpha, n_buckets=n_buckets)
    hist_local, split_bounds, flat = build_history(splits, L=L)

    order = ('train', 'valid', 'test')
    flat_video = np.concatenate([enc[name][0][:, 1] for name in order])  # video_id global idx
    flat_author = np.concatenate([enc[name][0][:, 2] for name in order])  # author_id global idx
    flat_label = np.concatenate([enc[name][1] for name in order]).astype(np.float32)

    PAD_IDX = dim
    hist = {}
    for name in order:
        row_idx, mask = hist_local[name]
        valid = row_idx >= 0
        safe_idx = np.where(valid, row_idx, 0)
        hv = np.where(valid, flat_video[safe_idx], PAD_IDX).astype(np.int64)
        ha = np.where(valid, flat_author[safe_idx], PAD_IDX).astype(np.int64)
        hl = np.where(valid, flat_label[safe_idx], 0.0).astype(np.float32)
        hist[name] = (hv, ha, hl, mask.astype(np.float32))

    decayed_pos_dict = _de.compute_final_decayed_pos(splits['train'])
    return enc, dim, hist, decayed_pos_dict, PAD_IDX


if __name__ == '__main__':
    import time
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
    t0 = time.time()
    enc, dim, hist, decayed_pos, PAD_IDX = prepare(DATA_DIR)
    print(f"prepare() done in {time.time()-t0:.1f}s, dim={dim}, PAD_IDX={PAD_IDX}")
    for name in ('train', 'valid', 'test'):
        X, y, users = enc[name]
        hv, ha, hl, hm = hist[name]
        print(f"  {name}: X={X.shape} y={y.shape} hist={hv.shape} "
              f"mean_hist_len={hm.sum(1).mean():.2f} zero_hist_frac={(hm.sum(1)==0).mean():.4f}")

    # --- causality spot-check: brute-force re-derive history for a handful
    # of random rows and compare against the vectorized/deque-based build.
    print("\n=== causality spot-check (brute force) ===")
    splits = _de.load_ext(DATA_DIR)
    order = ('train', 'valid', 'test')
    flat = []
    for name in order:
        flat.extend(splits[name])
    IDX = _de.IDX
    rng = np.random.default_rng(0)
    Xall = {name: enc[name][0] for name in order}
    bounds = {}
    off = 0
    for name in order:
        bounds[name] = (off, off + len(splits[name]))
        off += len(splits[name])

    n_checked, n_mismatch = 0, 0
    for _ in range(30):
        name = rng.choice(order)
        lo, hi = bounds[name]
        local_i = rng.integers(0, hi - lo)
        global_i = lo + local_i
        r = flat[global_i]
        u, t, oi = r[IDX['user_id']], r[IDX['time_ms']], r[IDX['orig_idx']]
        # brute force: all OTHER rows of this user strictly earlier by (time_ms, orig_idx)
        earlier = [j for j, rr in enumerate(flat)
                   if rr[IDX['user_id']] == u and (rr[IDX['time_ms']], rr[IDX['orig_idx']]) < (t, oi)]
        earlier.sort(key=lambda j: (flat[j][IDX['time_ms']], flat[j][IDX['orig_idx']]))
        expected_tail = earlier[-L_HIST:]
        hv, ha, hl, hm = hist[name]
        got_mask = hm[local_i]
        got_len = int(got_mask.sum())
        n_checked += 1
        if got_len != len(expected_tail):
            n_mismatch += 1
            print(f"  MISMATCH len: row={global_i} got_len={got_len} expected={len(expected_tail)}")
            continue
        # compare video ids
        flat_video_all = np.concatenate([Xall[nm][:, 1] for nm in order])
        got_video = hv[local_i][L_HIST - got_len:]
        exp_video = np.array([flat_video_all[j] for j in expected_tail])
        if got_len and not np.array_equal(got_video, exp_video):
            n_mismatch += 1
            print(f"  MISMATCH video: row={global_i}")
    print(f"checked {n_checked} random rows, {n_mismatch} mismatches")
    assert n_mismatch == 0, "causality check FAILED"
    print("causality spot-check PASSED (zero leakage / zero mismatch)")
