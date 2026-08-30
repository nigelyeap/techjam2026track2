"""iter31: MULTI-TASK LEARNING -- an organizer-suggested "unexplored
direction" (README section "未探索: headroom 应该在这里", item 3: "多目标.
日志里还有 is_click、is_like、is_follow、is_comment、is_forward、
play_time_ms, 可以做多任务辅助 long_view 主任务") never tried across the
prior 30 iterations.

Starts from iter24's EXACT feature set/pipeline (this dir's data_ext.py is
a verbatim copy of iter24's, plus one addition: `load_aux_labels()` -- see
its docstring there for the leakage-by-construction argument). Everything
below down to `run_bpr_ext` is otherwise a line-for-line copy of iter24's
train.py (itself copied from iter3/iter9/iter16/iter18/iter19/iter20):
same FM class, same activity-weighted BPR sampling, same optimizer.

The ONLY new mechanism is `mtl_bpr_step`, a drop-in replacement for
`bpr_step` (kept, verbatim, alongside it for the aux_weight=0 harness-
fidelity check):

  Design -- "hard parameter sharing" via a LITERALLY shared score, no new
  architecture: the main task's own FM logit z = b + W[X].sum + inter(V)
  (the exact same forward pass BPR already computes) is ALSO used as the
  score for a pointwise logistic (BCE) loss against each of the 5
  auxiliary engagement labels, on the SAME Xpos/Xneg rows already sampled
  for the main BPR pair this step. The auxiliary BCE gradient (averaged
  across the 5 tasks) is scaled by a single scalar `aux_weight` (swept
  0.1-0.3) and added directly into the gradient on V/W/b BEFORE the Adam
  update -- i.e. one shared Adam step per training iteration optimizing
  `L = L_bpr(long_view) + aux_weight * mean_t BCE_t(engagement_t)`.

  This is deliberately the simplest faithful multi-task design given a
  linear/FM model trained via manual numpy gradients: no new parameters,
  no new forward pass, no new sampling -- literally the same batch, same
  score, an extra loss term summed into the same gradient tensors. It
  satisfies the point of MTL (shared representations across tasks) without
  needing a second scoring head.

  Safety/leakage argument (documented per protocol, same shape as iter22/
  23's decay-aware BPR sampling-weight argument): aux_pos/aux_neg are read
  from `load_aux_labels(...)['train']`, indexed by Xpos_rows/Xneg_rows --
  which are themselves indices into the TRAIN split only (they come out of
  `build_pos_neg_index(ytr, utr)`, built exclusively from `enc['train']`).
  The auxiliary labels never touch encode_ext's feature pipeline, are never
  concatenated into any X array, and are never read for valid/test rows
  anywhere in this file. They influence ONLY the gradient computed at each
  training step -- exactly like the existing decay-aware BPR sampling
  weight already does with a different train-only aggregate. No leakage
  path exists by construction: it would require deliberately reading
  aux['valid'] or aux['test'], which this file's evaluation code
  (`evaluate(uva, ...)` / `evaluate(ute, ...)`) never does -- those only
  ever call `m.predict(Xva)` / `m.predict(Xte)`, i.e. the FM's own logits
  on the frozen feature matrix, with zero reference to any aux array.
"""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from evaluate import evaluate           # noqa: E402  official eval, unmodified
from baseline import FM, sigmoid        # noqa: E402  same FM class as iter3/iter9

from data_ext import (load_ext, encode_ext, load_aux_labels, AUX_LABELS,   # noqa: E402
                       BASE_FIELDS, HALFLIVES, TAB_HALFLIVES)


def build_pos_neg_index(y, users):
    """Identical to iter3/iter9/iter16/iter18/iter19/iter20/iter24's train.py -- copied verbatim."""
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
    """Identical to iter3/iter9/iter16/iter18/iter19/iter20/iter24's train.py -- copied verbatim."""
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


def bpr_step(m, Xpos, Xneg):
    """Identical to iter3/iter9/iter16/iter18/iter19/iter20/iter24's train.py -- copied
    verbatim, UNCHANGED. Kept alongside mtl_bpr_step so aux_weight=0 can use this exact
    function for the harness-fidelity check (bit-exact reproduction of iter24)."""
    B = len(Xpos)
    zpos, Epos, Spos = m.logits(Xpos)
    zneg, Eneg, Sneg = m.logits(Xneg)
    d = zpos - zneg
    gpos = ((sigmoid(d) - 1) / B).astype(np.float32)
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
    return float(-np.mean(np.log(sigmoid(d) + 1e-9))), 0.0


def mtl_bpr_step(m, Xpos, Xneg, aux_pos, aux_neg, aux_weight):
    """Multi-task variant of bpr_step. aux_pos/aux_neg: (B, n_aux) float32
    arrays -- the auxiliary engagement labels for the SAME rows already
    sampled for the main BPR pair (Xpos/Xneg), train-only by construction
    (see module docstring). Everything through the `d = zpos - zneg` line is
    identical to bpr_step; the only change is summing an extra pointwise-BCE
    gradient (mean over the n_aux tasks, scaled by aux_weight) into the same
    per-row gradient `g` before it is scattered into gV/gW -- i.e. a single
    shared Adam step per call, not two separate updates.
    """
    B = len(Xpos)
    zpos, Epos, Spos = m.logits(Xpos)
    zneg, Eneg, Sneg = m.logits(Xneg)
    d = zpos - zneg
    gpos_bpr = ((sigmoid(d) - 1) / B).astype(np.float32)
    gneg_bpr = -gpos_bpr

    spos = sigmoid(zpos)                     # (B,)
    sneg = sigmoid(zneg)
    # mean-over-tasks pointwise BCE gradient dL/dz = (s - y); average the
    # n_aux task gradients so aux_weight's scale doesn't depend on n_aux.
    gpos_aux = ((spos[:, None] - aux_pos).mean(axis=1) / B).astype(np.float32)
    gneg_aux = ((sneg[:, None] - aux_neg).mean(axis=1) / B).astype(np.float32)

    gpos = gpos_bpr + aux_weight * gpos_aux
    gneg = gneg_bpr + aux_weight * gneg_aux

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

    bpr_loss = float(-np.mean(np.log(sigmoid(d) + 1e-9)))
    eps_l = 1e-9
    aux_bce_pos = -(aux_pos * np.log(spos[:, None] + eps_l) + (1 - aux_pos) * np.log(1 - spos[:, None] + eps_l))
    aux_bce_neg = -(aux_neg * np.log(sneg[:, None] + eps_l) + (1 - aux_neg) * np.log(1 - sneg[:, None] + eps_l))
    aux_loss = float(np.mean(np.concatenate([aux_bce_pos, aux_bce_neg], axis=0)))
    return bpr_loss, aux_loss


def run_bpr_ext(data_dir, feature_set=('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3',
                                        'last1', 'lastk_rate', 'gap'),
                 halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                 k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True,
                 steps_mult=1, K=5, splits_cache=None,
                 aux_weight=0.0, aux_tasks=AUX_LABELS, aux_cache=None):
    """splits_cache: optional pre-loaded load_ext() result, to avoid recomputing
    the (expensive, 4-traversal) feature pass across many sweep configs.

    aux_weight: multi-task auxiliary loss coefficient (0.0 = exactly iter24,
    uses the unmodified bpr_step for a bit-exact harness-fidelity check).
    aux_tasks: subset/order of AUX_LABELS to use when aux_weight > 0.
    aux_cache: optional pre-loaded load_aux_labels() result (avoids re-reading
    the raw CSVs across sweep configs, same caching pattern as splits_cache).
    """
    splits = splits_cache if splits_cache is not None else \
        load_ext(data_dir, halflives=halflives, tab_halflives=tab_halflives, K=K)
    enc, dim = encode_ext(splits, feature_set=feature_set, halflives=halflives, tab_halflives=tab_halflives)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = \
        build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs))) * steps_mult

    user_cumw = np.cumsum(pos_len.astype(np.float64))
    user_totalw = user_cumw[-1]

    use_mtl = aux_weight != 0.0
    aux_mat = None
    if use_mtl:
        aux = aux_cache if aux_cache is not None else load_aux_labels(data_dir)
        aux_train = aux['train']
        assert len(aux_train[AUX_LABELS[0]]) == len(ytr), \
            "aux label count must match train row count (see data_ext.load_aux_labels)"
        aux_mat = np.stack([aux_train[t] for t in aux_tasks], axis=1).astype(np.float32)  # (Ntr, n_aux)

    if verbose:
        fields = list(BASE_FIELDS) + list(feature_set)
        print(f"  fields={fields} dim={dim}")
        print(f"  BPR-eligible train users: {n_users} (>=1 pos & >=1 neg) | "
              f"{steps_per_epoch} steps/epoch | activity-weighted user sampling")
        if use_mtl:
            print(f"  multi-task: aux_weight={aux_weight} aux_tasks={list(aux_tasks)}")

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        bpr_losses, aux_losses = [], []
        for _ in range(steps_per_epoch):
            Xpos_rows, Xneg_rows = sample_pairs(
                rng, n_users, bs, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
                user_cumw=user_cumw, user_totalw=user_totalw)
            if use_mtl:
                bl, al = mtl_bpr_step(m, Xtr[Xpos_rows], Xtr[Xneg_rows],
                                       aux_mat[Xpos_rows], aux_mat[Xneg_rows], aux_weight)
            else:
                bl, al = bpr_step(m, Xtr[Xpos_rows], Xtr[Xneg_rows])
            bpr_losses.append(bl); aux_losses.append(al)
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            extra = f" | aux_bce {np.mean(aux_losses):.4f}" if use_mtl else ""
            print(f"  epoch {ep:2d} | bpr_loss {np.mean(bpr_losses):.4f}{extra} | valid GAUC {va['GAUC']:.4f} "
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
    ap.add_argument('--data_dir', default='../../KuaiRand-Pure/data')
    ap.add_argument('--features', default='decay_rate_2.5,decay_act_2.5,decay_tab_3,last1,lastk_rate,gap',
                    help="comma-separated subset/order of "
                         "{activity,tab,rate,decay_rate_H,decay_act_H,decay_tab_H,last1,lastk_rate,gap} "
                         "-- default is iter24's exact winning feature set")
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps_mult', type=int, default=1)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--aux_weight', type=float, default=0.0,
                    help="multi-task auxiliary loss coefficient (0.0 = iter24 baseline, "
                         "bit-exact harness-fidelity mode)")
    ap.add_argument('--aux_tasks', default=','.join(AUX_LABELS),
                    help="comma-separated subset of is_click,is_like,is_follow,is_comment,is_forward")
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    feature_set = tuple(f for f in a.features.split(',') if f)
    aux_tasks = tuple(f for f in a.aux_tasks.split(',') if f)
    print(f"loading {a.data_dir} ... extra features={feature_set} aux_weight={a.aux_weight} "
          f"aux_tasks={aux_tasks}")
    res = run_bpr_ext(a.data_dir, feature_set=feature_set, k=a.k, lr=a.lr, epochs=a.epochs,
                       bs=a.bs, patience=a.patience, seed=a.seed, verbose=not a.quiet,
                       steps_mult=a.steps_mult, K=a.K, aux_weight=a.aux_weight, aux_tasks=aux_tasks)
    print(f"\n=== iter31 fm+bpr_weighted+{'+'.join(feature_set)}+mtl(w={a.aux_weight}) (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
    def _clean(d):
        return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}
    print(json.dumps({'seed': a.seed, 'features': list(feature_set), 'aux_weight': a.aux_weight,
                       'aux_tasks': list(aux_tasks),
                       'valid': _clean(res['valid']), 'test': _clean(res['test'])}))
