"""iter64: a genuinely SEPARATE sequence model (SASRec-style self-attention
over each user's own causally-ordered interaction history), motivated by
iter40's own closing recommendation ("a lever... structurally different
from 'add a mechanism on top of FM' ... a non-FM-family model") and the
user's explicit permission to use any open-source library/paper.

Mechanistic difference from iter32/34/40 (all three REJECTed): those three
routed an attention signal back into FM's OWN shared embedding table and
loss, which the iter40 diagnosis pinned as the failure mode (a second
gradient path through the same embeddings the FM's bilinear term depends on
conflicts with the rank-invariant BPR objective). This module instead builds
a fully independent model: its OWN item-embedding table, its OWN causal
self-attention encoder, its OWN BPR loss -- trained standalone, and combined
with the existing best (iter63) blend ONLY via post-hoc score-blending
(matching how the FM and GBM components are already combined). No shared
parameters, no shared gradient path.

Causal-ordering discipline: identical pattern to iter18's
`compute_momentum_features` (imported, not copied) -- for each user, sort
ALL of that user's rows (across every split) by (time_ms, orig_idx), a
strict total order using the raw log's genuine millisecond timestamp with a
stable tiebreak. A row's history is exactly the video_ids of that user's
rows STRICTLY EARLIER in this order (deque updated only AFTER a row's
history has been read, so a row can never see itself or a same-time_ms tie
in either direction).

Item-id vocabulary: fit on TRAIN rows only (id 0 reserved for PAD, id 1 for
UNK/unseen-at-train-time video), matching data.py's/encode()'s own
train-only-vocab-with-UNK-fallback convention. This applies to both a
row's own target video_id and every video_id appearing in its history.

Does NOT modify data.py or any iterN/data_ext.py file it reuses/imports.
"""
import os, sys, collections, importlib.util
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from data import SPLITS  # noqa: E402  (date ranges only, kept in sync w/ data.py)

_iter18_path = os.path.join(_THIS_DIR, '..', 'iter18_momentum', 'data_ext.py')
_spec = importlib.util.spec_from_file_location('iter18_data_ext_for_iter64', _iter18_path)
_iter18_data_ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_iter18_data_ext)
_load_raw_time = _iter18_data_ext._load_raw_time  # noqa: E402

PAD, UNK = 0, 1
MAX_LEN_DEFAULT = 20


def build_vocab(train_rows):
    """train_rows: extended tuples (video_id at index 2). Train-only fit,
    matching data.py's encode() convention (UNK fallback for anything not
    seen at train time -- covers unseen video_ids in valid/test AND in a
    user's own history when that history reaches into valid/test)."""
    vids = sorted(set(r[2] for r in train_rows))
    return {v: i + 2 for i, v in enumerate(vids)}  # 0=PAD, 1=UNK, 2.. real items


def compute_history_sequences(rows, vocab, max_len=MAX_LEN_DEFAULT):
    """rows: extended tuples (user_id@1, video_id@2, time_ms@8, orig_idx@9),
    in ANY order. Returns:
      hist_ids : (n, max_len) int64, right-aligned (most recent item last),
                 left-padded with PAD=0
      hist_len : (n,) int64, number of real (non-pad) history items (<=max_len)
      item_id  : (n,) int64, this row's OWN target video_id mapped through vocab

    Same strict-causality construction as iter18's compute_momentum_features:
    per-user sort by (time_ms, orig_idx); a row's history is read BEFORE its
    own item_id is pushed onto that user's rolling window, so no leakage of
    the current or any later row into a row's own history.
    """
    n = len(rows)
    hist_ids = np.zeros((n, max_len), dtype=np.int64)
    hist_len = np.zeros(n, dtype=np.int64)
    item_id = np.zeros(n, dtype=np.int64)

    by_user = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_user[r[1]].append(i)

    for u, idxs in by_user.items():
        idxs.sort(key=lambda i: (rows[i][8], rows[i][9]))
        window = collections.deque(maxlen=max_len)
        for i in idxs:
            L = len(window)
            if L > 0:
                hist_ids[i, max_len - L:] = np.array(window, dtype=np.int64)
            hist_len[i] = L
            vid = rows[i][2]
            iid = vocab.get(vid, UNK)
            item_id[i] = iid
            window.append(iid)
    return hist_ids, hist_len, item_id


def load_ext(data_dir, max_len=MAX_LEN_DEFAULT):
    """Returns dict split -> (hist_ids, hist_len, item_id, labels, user_ids)
    each aligned index-for-index with data.py's load()[split] row order
    (both read the same two CSVs, same date-range filter, same order)."""
    splits = _load_raw_time(data_dir)
    order = ('train', 'valid', 'test')
    flat, owner = [], []
    for name in order:
        for r in splits[name]:
            flat.append(r)
            owner.append(name)

    vocab = build_vocab(splits['train'])
    hist_ids, hist_len, item_id = compute_history_sequences(flat, vocab, max_len=max_len)
    labels = np.array([r[6] for r in flat], dtype=np.int64)
    users = [r[1] for r in flat]

    out = {}
    idx_by_split = {name: [] for name in order}
    for i, name in enumerate(owner):
        idx_by_split[name].append(i)
    for name in order:
        idx = np.array(idx_by_split[name], dtype=np.int64)
        out[name] = (hist_ids[idx], hist_len[idx], item_id[idx], labels[idx],
                      [users[i] for i in idx])
    return out, vocab
