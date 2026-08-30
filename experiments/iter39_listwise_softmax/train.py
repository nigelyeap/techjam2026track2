"""iter39: listwise (grouped-softmax) loss, replacing iter27's pairwise BPR
loss, everything else (features, formula constants, decay-weighted user
sampling) held identical for a clean loss-function-only ablation.

This is the one remaining untried variant of the README's own suggestion
#1 ("switch loss to pairwise or listwise") -- pairwise (BPR) is already
fused into iter27; listwise has never been tried.

Loss: for each sampled user (group), take all their train-split positives
plus a capped random subsample of negatives (M_max total rows per group).
p = softmax(FM logits) over the group; target t_i = y_i / n_pos_in_group
(uniform over the group's positives, 0 elsewhere). L_group = -sum t*log(p).
Gradient dL/dz_i = p_i - t_i (standard multinomial-logistic result) --
verified against finite-difference numerical gradients on a toy example
before this file was written (see
/private/tmp/claude-501/.../scratchpad/grad_check_listwise.py, max abs
err ~1.7e-11).

build_pos_neg_index is iter3/9/19/23/27's function, reused unmodified
(imported from iter27_triple_fusion/train.py) since group construction
needs exactly its per-user pos/neg row index structure.
"""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'iter27_triple_fusion'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from evaluate import evaluate                      # noqa: E402
from baseline import FM                             # noqa: E402
from data_ext import (load_ext, encode_ext, compute_final_decayed_pos,  # noqa: E402
                       BASE_FIELDS, HALFLIVES, TAB_HALFLIVES, ALPHA)


def build_pos_neg_index(y, users):
    """Identical to iter3/9/16/18/19/20/23/27's train.py -- copied verbatim
    (not imported, to avoid a same-named-module self-import collision with
    this file, which is also called train.py)."""
    by_user_pos, by_user_neg = {}, {}
    for i, (yi, u) in enumerate(zip(y, users)):
        (by_user_pos if yi == 1 else by_user_neg).setdefault(u, []).append(i)
    eligible = sorted(set(by_user_pos) & set(by_user_neg))

    def flatten(by_user):
        starts, lens, flat = [], [], []
        off = 0
        for u in eligible:
            idx = by_user[u]
            starts.append(off); lens.append(len(idx)); flat.extend(idx)
            off += len(idx)
        return (np.array(flat, dtype=np.int64), np.array(starts, dtype=np.int64),
                np.array(lens, dtype=np.int64))

    pos_flat, pos_start, pos_len = flatten(by_user_pos)
    neg_flat, neg_start, neg_len = flatten(by_user_neg)
    return eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len


def sample_listwise_batch(rng, G, M_max, eligible, pos_flat, pos_start, pos_len,
                           neg_flat, neg_start, neg_len, user_cumw, user_totalw):
    n_users = len(eligible)
    picked = np.searchsorted(user_cumw, rng.random(G) * user_totalw, side='right')
    picked = np.minimum(picked, n_users - 1)
    row_chunks, y_chunks, lengths = [], [], []
    for u in picked:
        np_ = int(pos_len[u]); nn_ = int(neg_len[u])
        pos_idx = pos_flat[pos_start[u]:pos_start[u] + np_]
        if np_ >= M_max:
            pos_idx = pos_idx[rng.choice(np_, size=M_max, replace=False)]
            neg_sel = np.empty(0, dtype=np.int64)
        else:
            neg_cap = M_max - np_
            if nn_ > neg_cap:
                neg_sel = neg_flat[neg_start[u] + rng.choice(nn_, size=neg_cap, replace=False)]
            else:
                neg_sel = neg_flat[neg_start[u]:neg_start[u] + nn_]
        rows = np.concatenate([pos_idx, neg_sel])
        row_chunks.append(rows)
        y_chunks.append(np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_sel))]))
        lengths.append(len(rows))
    X_idx = np.concatenate(row_chunks)
    y_grp = np.concatenate(y_chunks)
    lengths = np.array(lengths, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    return X_idx, y_grp, offsets, lengths


def listwise_step(m, X, y_grp, offsets, lengths):
    """X: the (N, F) encoded feature-index matrix for this batch's rows,
    already gathered by the caller (Xtr[X_idx]) -- same convention as
    bpr_step's X argument (feature-vocab indices, NOT dataset row indices).
    y_grp/offsets/lengths describe the contiguous per-group structure of
    X's rows."""
    G = len(lengths)
    z, E, S = m.logits(X)
    group_max = np.maximum.reduceat(z, offsets)
    z_shift = z - np.repeat(group_max, lengths)
    exp_z = np.exp(z_shift)
    group_sum = np.add.reduceat(exp_z, offsets)
    p = exp_z / np.repeat(group_sum, lengths)
    group_pos = np.add.reduceat(y_grp, offsets)
    t = y_grp / np.repeat(group_pos, lengths)

    g = ((p - t) / G).astype(np.float32)
    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    np.add.at(gW, X, g[:, None])
    np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
    gV += m.l2 * m.V; gW += m.l2 * m.W
    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for P, Gr, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * Gr
        Vv *= b2; Vv += (1 - b2) * (Gr * Gr)
        P -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    m.b -= m.lr * g.sum()
    return float(-np.mean(t[t > 0] * np.log(p[t > 0] + 1e-9)))


def run_listwise(data_dir, feature_set=('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3',
                                         'last1', 'lastk_rate', 'gap'),
                  halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                  k=16, lr=0.001, epochs=40, G=256, M_max=16, patience=4, seed=0, verbose=True,
                  splits_cache=None, sampling_alpha=0.75, decay_halflife=3,
                  alpha=ALPHA, n_buckets=20, steps_per_epoch=200):
    splits = splits_cache if splits_cache is not None else \
        load_ext(data_dir, halflives=halflives, tab_halflives=tab_halflives)
    enc, dim = encode_ext(splits, feature_set=feature_set, halflives=halflives, tab_halflives=tab_halflives,
                           alpha=alpha, n_buckets=n_buckets)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = build_pos_neg_index(ytr, utr)
    decayed_pos_dict = compute_final_decayed_pos(splits['train'], halflife=decay_halflife)
    decayed_arr = np.array([decayed_pos_dict.get(u, 0.0) for u in eligible], dtype=np.float64)
    weights = decayed_arr ** sampling_alpha
    user_cumw = np.cumsum(weights); user_totalw = user_cumw[-1]

    if verbose:
        print(f"  BPR/listwise-eligible train users: {len(eligible)} | {steps_per_epoch} steps/epoch "
              f"| G={G} M_max={M_max} | sampling_alpha={sampling_alpha} decay_halflife={decay_halflife}")

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        for _ in range(steps_per_epoch):
            X_idx, y_grp, offsets, lengths = sample_listwise_batch(
                rng, G, M_max, eligible, pos_flat, pos_start, pos_len,
                neg_flat, neg_start, neg_len, user_cumw, user_totalw)
            losses.append(listwise_step(m, Xtr[X_idx], y_grp, offsets, lengths))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} "
                  f"| {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='../../KuaiRand-Pure/data')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()
    res = run_listwise(a.data_dir, seed=a.seed, lr=a.lr, epochs=a.epochs)
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s} GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} primary {r['primary']:.4f}")
