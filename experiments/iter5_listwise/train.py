"""Listwise (ListNet-style softmax cross-entropy) training on top of the same
FM architecture as baseline.py.

Motivation: the FM baseline trains on pointwise logloss and iter2's BPR trains
on pairwise ranking loss -- both are still one step removed from the metric
(GAUC / nDCG@5), which score a whole ranked *list* per user at once. This
iteration aligns the loss family with that group structure directly: for each
sampled user we build a softmax distribution over their sampled rows' scores
and cross-entropy it against a uniform target over that user's positive rows
(the standard ListNet listwise loss). A user contributes to a step only if
their sampled group contains >=1 positive row -- mirrors nDCG@5's convention
that a zero-positive group has no defined "correct" ranking to match.

Reuses baseline.FM unmodified (same V/W/b, same Adam optimizer, same
logits()) and copies the mechanical Adam-update pattern from
experiments/iter2_bpr_uniform/train.py's bpr_step -- only the loss and the
per-row gradient it produces are different.
"""
import argparse, sys, time
import numpy as np
sys.path.insert(0, '..')
from data import load, encode, FIELDS
from evaluate import evaluate
from baseline import FM, sigmoid


def build_user_index(y, users):
    """Group train row indices by user. Restrict the sampling pool to users
    with >=1 positive row *somewhere* in their train history -- a user with
    zero positives overall can never pass the per-step "sampled group has
    >=1 positive" filter (step 3 of the spec), so dropping them here is
    equivalent but avoids wasted sampling every single step."""
    by_user = {}
    for i, u in enumerate(users):
        by_user.setdefault(u, []).append(i)
    pos_count = {}
    for i, (yi, u) in enumerate(zip(y, users)):
        if yi == 1:
            pos_count[u] = pos_count.get(u, 0) + 1
    eligible = sorted(u for u in by_user if pos_count.get(u, 0) > 0)
    user_rows = {u: np.array(by_user[u], dtype=np.int64) for u in eligible}
    return eligible, user_rows


def sample_batch_groups(rng, eligible_arr, user_rows, y, users_per_batch, cap):
    """Sample users_per_batch users uniformly (with replacement) from the
    eligible pool. For each, take up to `cap` of their train rows (all if
    fewer, a random subset without replacement if more). Drop any sampled
    group that ends up with zero positive rows (possible even for an
    eligible user if cap << their row count and positives are rare)."""
    picked = rng.choice(eligible_arr, size=users_per_batch, replace=True)
    row_blocks, gid_blocks, gid = [], [], 0
    for u in picked:
        rows = user_rows[u]
        if len(rows) > cap:
            rows = rng.choice(rows, size=cap, replace=False)
        if y[rows].sum() < 1:
            continue
        row_blocks.append(rows)
        gid_blocks.append(np.full(len(rows), gid, dtype=np.int64))
        gid += 1
    if gid == 0:
        return None, None, 0
    return np.concatenate(row_blocks), np.concatenate(gid_blocks), gid


def listwise_step(m, Xrows, gid, n_groups, y):
    """Same Adam/FM update math as baseline.FM.step / iter2's bpr_step, but
    the per-row logit gradient comes from softmax cross-entropy against a
    uniform-over-positives target within each user's group. For softmax CE
    with target summing to 1 per group, d(loss_group)/d(z_i) = softmax_i - target_i.
    We average the accumulated gradient over the number of groups (users) in
    the batch, mirroring how the pointwise/BPR steps average over row/pair
    count -- here the natural "unit" of the loss is one user's list."""
    z, E, S = m.logits(Xrows)

    max_per_group = np.full(n_groups, -np.inf, dtype=np.float32)
    np.maximum.at(max_per_group, gid, z)
    ez = np.exp(z - max_per_group[gid])
    sum_per_group = np.zeros(n_groups, dtype=np.float32)
    np.add.at(sum_per_group, gid, ez)
    sm = ez / sum_per_group[gid]

    pos_count = np.zeros(n_groups, dtype=np.float32)
    np.add.at(pos_count, gid, y)
    target = np.where(y == 1, 1.0 / pos_count[gid], 0.0).astype(np.float32)

    g = ((sm - target) / n_groups).astype(np.float32)

    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    np.add.at(gW, Xrows, g[:, None])
    np.add.at(gV, Xrows, g[:, None, None] * (S[:, None, :] - E))
    gV += m.l2 * m.V; gW += m.l2 * m.W
    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for P, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        P -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    m.b -= m.lr * g.sum()

    loss_per_group = np.zeros(n_groups, dtype=np.float32)
    np.add.at(loss_per_group, gid, -target * np.log(sm + 1e-9))
    return float(loss_per_group.sum() / n_groups)


def run_listwise(splits, k=16, lr=0.001, epochs=40, users_per_batch=256, cap=50,
                  patience=4, seed=0, verbose=True, steps_mult=1):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    eligible, user_rows = build_user_index(ytr, utr)
    eligible_arr = np.array(eligible)
    n_eligible = len(eligible)
    # ~one pass over the eligible user pool per epoch (spec guidance):
    # steps_per_epoch ~= n_eligible_users / users_per_batch
    steps_per_epoch = max(1, int(np.ceil(n_eligible / users_per_batch))) * steps_mult
    if verbose:
        print(f"  listwise-eligible train users: {n_eligible} (>=1 positive) | "
              f"{steps_per_epoch} steps/epoch, users_per_batch={users_per_batch}, cap={cap}")

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        for _ in range(steps_per_epoch):
            rows, gid, n_groups = sample_batch_groups(
                rng, eligible_arr, user_rows, ytr, users_per_batch, cap)
            if n_groups == 0:
                continue
            losses.append(listwise_step(m, Xtr[rows], gid, n_groups, ytr[rows]))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
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
    ap.add_argument('--data_dir', default='../KuaiRand-Pure/data')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--users_per_batch', type=int, default=256)
    ap.add_argument('--cap', type=int, default=50)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps_mult', type=int, default=1)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = run_listwise(splits, k=a.k, lr=a.lr, epochs=a.epochs,
                        users_per_batch=a.users_per_batch, cap=a.cap,
                        patience=a.patience, seed=a.seed, steps_mult=a.steps_mult)
    print(f"\n=== fm+listwise (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
