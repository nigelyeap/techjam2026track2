"""Hybrid multi-task loss: pointwise logloss + activity-weighted BPR pairwise loss,
trained jointly on ONE shared FM instance.

Hypothesis (round 3, iter8): iter1's pointwise loss and iter3's BPR loss learn
complementary things. Pointwise sees every row and does implicit
activity-weighting for free (it just iterates over all training rows each
epoch), which is good for calibration/overall discrimination (GAUC-ish).
BPR (iter3's activity-weighted version) explicitly optimizes within-user pairwise
ordering, which is good for top-of-list precision (nDCG-ish). A joint objective
might get both.

Training procedure per epoch (PASS-BY-PASS, chosen for simplicity/correctness
over step-by-step interleaving — see note below):
  1. One full pointwise pass over the shuffled training set, exactly like
     baseline.py's run_fm (dense minibatches of size `bs`, one Adam step per
     minibatch, ~139 steps/epoch on this dataset).
  2. Then `steps_per_epoch` activity-weighted BPR steps, exactly like
     iter3_bpr_weighted/train.py's run_bpr (same steps_per_epoch formula:
     ceil(pos_len.sum() / bs), ~47 steps/epoch on this dataset), scaled by
     `--bpr_weight`.
Both (1) and (2) update the SAME FM.V/W/b via the SAME shared Adam moment
accumulators (mV/vV/mW/vW) on the FM instance — there is only one set of
weights and one optimizer state; the two losses just take turns writing
gradients into it every epoch.

Why pass-by-pass instead of step-by-step interleaving: pass-by-pass lets us
reuse baseline.py's run_fm inner loop and iter3's run_bpr inner loop
essentially verbatim (each is already correct and tested), so the only new
code is the outer epoch loop and the bpr_weight scaling. Step-by-step
interleaving would require carefully alternating two very differently-shaped
batch constructions (dense-permutation index vs weighted-pair sampling) inside
a single loop with a mixing ratio, which is more bookkeeping for a hypothesis
test that doesn't depend on the interleaving granularity. Since Adam's moment
accumulators are per-parameter EMAs updated every step regardless of which
loss produced the gradient, the two update sources mix into the same
trajectory either way within an epoch; pass-by-pass just batches "when" each
loss gets to write.

`--bpr_weight` scales the BPR loss's per-row gradient magnitude (multiplies
gpos/gneg in bpr_step, equivalent to scaling the BPR loss term by a constant)
relative to the pointwise loss's gradient, since pointwise logloss and BPR
pairwise loss have different natural gradient scales. Default 1.0 (unscaled).

Reuses baseline.FM unmodified (same V/W/b, same Adam optimizer, same
logits()). build_pos_neg_index / sample_pairs / the bpr gradient math are
copied directly from experiments/iter3_bpr_weighted/train.py (only bpr_step
gained a `weight` argument that scales the pairwise gradient before it's
accumulated into gV/gW).
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from data import load, encode, FIELDS
from evaluate import evaluate
from baseline import FM, sigmoid


# ---------------- copied from experiments/iter3_bpr_weighted/train.py ----------------

def build_pos_neg_index(y, users):
    """For each user with >=1 positive AND >=1 negative row, build a CSR-style
    flat index of positive/negative row positions so sampling is vectorizable.
    Users that are all-positive or all-negative are dropped (no pairs to form)."""
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


def sample_pairs(rng, n_users, bs, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
                  user_cumw=None, user_totalw=None):
    """user_cumw/user_totalw: cumulative-sum sampling table over eligible users,
    weighted by activity (positive-row count, iter3's fix). Users are picked
    proportional to their weight via cumsum + searchsorted (vectorized)."""
    if user_cumw is not None:
        picked = np.searchsorted(user_cumw, rng.random(bs) * user_totalw, side='right')
        picked = np.minimum(picked, n_users - 1)
    else:
        picked = rng.integers(0, n_users, size=bs)
    pos_rand = (rng.random(bs) * pos_len[picked]).astype(np.int64)
    neg_rand = (rng.random(bs) * neg_len[picked]).astype(np.int64)
    pos_rows = pos_flat[pos_start[picked] + pos_rand]
    neg_rows = neg_flat[neg_start[picked] + neg_rand]
    return pos_rows, neg_rows


def bpr_step(m, Xpos, Xneg, weight=1.0):
    """Same Adam/FM update math as baseline.FM.step, but the per-row logit
    gradient comes from the BPR pairwise loss instead of pointwise logloss.
    `weight` scales the BPR gradient contribution (== scaling the BPR loss
    term by a constant before differentiating) relative to pointwise steps
    taken elsewhere on the same FM instance/optimizer state."""
    B = len(Xpos)
    zpos, Epos, Spos = m.logits(Xpos)
    zneg, Eneg, Sneg = m.logits(Xneg)
    d = zpos - zneg
    gpos = (weight * (sigmoid(d) - 1) / B).astype(np.float32)
    gneg = -gpos

    X = np.concatenate([Xpos, Xneg], axis=0)
    E = np.concatenate([Epos, Eneg], axis=0)
    S = np.concatenate([Spos, Sneg], axis=0)
    g = np.concatenate([gpos, gneg], axis=0)

    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    np.add.at(gW, X, g[:, None])
    np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
    gV += m.l2 * m.V; gW += m.l2 * m.W
    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for P, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        P -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    m.b -= m.lr * g.sum()
    return float(-np.mean(np.log(sigmoid(d) + 1e-9)))


# ---------------- hybrid training loop (new for iter8) ----------------

def run_hybrid(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True,
               bpr_weight=1.0):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = \
        build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs)))
    user_cumw = np.cumsum(pos_len.astype(np.float64))
    user_totalw = user_cumw[-1]

    if verbose:
        n_pt_batches = int(np.ceil(len(ytr) / bs))
        print(f"  BPR-eligible train users: {n_users} (>=1 pos & >=1 neg) | "
              f"{n_pt_batches} pointwise batches/epoch + {steps_per_epoch} BPR steps/epoch "
              f"| bpr_weight={bpr_weight} | activity-weighted user sampling")

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()

        # (1) full pointwise pass, dense minibatches, exactly like baseline.run_fm
        idx = rng.permutation(len(ytr))
        pt_losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]

        # (2) activity-weighted BPR pass, same steps_per_epoch as iter3, scaled by bpr_weight
        bpr_losses = []
        for _ in range(steps_per_epoch):
            Xpos_rows, Xneg_rows = sample_pairs(
                rng, n_users, bs, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
                user_cumw=user_cumw, user_totalw=user_totalw)
            bpr_losses.append(bpr_step(m, Xtr[Xpos_rows], Xtr[Xneg_rows], weight=bpr_weight))

        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | pt_loss {np.mean(pt_losses):.4f} | bpr_loss {np.mean(bpr_losses):.4f} "
                  f"| valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} "
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
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bpr_weight', type=float, default=1.0,
                    help='scales the BPR gradient contribution relative to the pointwise '
                         'gradient contribution on the shared FM parameters')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = run_hybrid(splits, k=a.k, lr=a.lr, epochs=a.epochs, bs=a.bs,
                      patience=a.patience, seed=a.seed, bpr_weight=a.bpr_weight)
    print(f"\n=== fm+hybrid (bpr_weight={a.bpr_weight}, seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
