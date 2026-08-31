"""Brute-force causality spot-check for compute_history_sequences: for a
sample of rows, rebuild that row's expected history independently (sort ALL
of that user's rows across the combined flat list by (time_ms, orig_idx),
take the video_ids strictly before this row, map through vocab, right-align
into max_len) and assert it matches data_ext.py's vectorized output exactly.
"""
import os, sys, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, build_vocab, PAD, UNK, MAX_LEN_DEFAULT, _load_raw_time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'KuaiRand-Pure', 'data')


def main():
    splits = _load_raw_time(DATA_DIR)
    order = ('train', 'valid', 'test')
    flat, owner = [], []
    for name in order:
        for r in splits[name]:
            flat.append(r)
            owner.append(name)
    vocab = build_vocab(splits['train'])

    out, vocab2 = load_ext(DATA_DIR)
    assert vocab2 == vocab

    # rebuild owner->global-flat-index map to cross-reference with out[split]
    idx_by_split = {name: [] for name in order}
    for i, name in enumerate(owner):
        idx_by_split[name].append(i)

    by_user = {}
    for i, r in enumerate(flat):
        by_user.setdefault(r[1], []).append(i)
    for u in by_user:
        by_user[u].sort(key=lambda i: (flat[i][8], flat[i][9]))

    rng = random.Random(0)
    n_checked = 0
    n_fail = 0
    for name in order:
        hist_ids, hist_len, item_id, labels, users = out[name]
        local_idxs = idx_by_split[name]
        sample = rng.sample(range(len(local_idxs)), min(300, len(local_idxs)))
        for local_i in sample:
            global_i = local_idxs[local_i]
            r = flat[global_i]
            u = r[1]
            seq = by_user[u]
            pos = seq.index(global_i)
            prior = seq[:pos]
            prior_items = [vocab.get(flat[j][2], UNK) for j in prior[-MAX_LEN_DEFAULT:]]
            expected_len = len(prior_items)
            expected_hist = np.zeros(MAX_LEN_DEFAULT, dtype=np.int64)
            if expected_len > 0:
                expected_hist[MAX_LEN_DEFAULT - expected_len:] = expected_items = np.array(prior_items, dtype=np.int64)
            expected_item_id = vocab.get(r[2], UNK)

            n_checked += 1
            ok = (hist_len[local_i] == expected_len and
                  np.array_equal(hist_ids[local_i], expected_hist) and
                  item_id[local_i] == expected_item_id and
                  labels[local_i] == r[6] and
                  users[local_i] == u)
            if not ok:
                n_fail += 1
                if n_fail <= 5:
                    print(f"MISMATCH split={name} local_i={local_i} user={u}")
                    print(f"  got hist_len={hist_len[local_i]} expected={expected_len}")
                    print(f"  got hist_ids={hist_ids[local_i]}")
                    print(f"  exp hist_ids={expected_hist}")
                    print(f"  got item_id={item_id[local_i]} expected={expected_item_id}")

    print(f"checked {n_checked} rows across {order}, failures={n_fail}")
    assert n_fail == 0, "causality verification FAILED"
    print("PASS: causal history construction verified by brute-force spot-check")


if __name__ == '__main__':
    main()
