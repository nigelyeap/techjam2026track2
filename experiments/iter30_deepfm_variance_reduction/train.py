"""iter30: variance-reduction study on iter26's DeepFM deep part
(`deep_h32`: single 32-unit hidden layer, on iter19's exact feature set).

Reuses iter26's forward/backward math and BPR training loop VERBATIM --
`build_pos_neg_index`, `sample_pairs`, and the FM-part gradient block inside
`deepfm_bpr_step` are byte-for-byte copies of `iter26_deepfm/train.py` (which
were themselves byte-for-byte copies of iter19's `train.py`). The only new
code is:

  1. `run_deepfm_bpr` here is `iter26_deepfm.train.run_deepfm_bpr` with the
     model class swapped from `DeepFM` to `model.DeepFMVR` (this dir) and two
     extra passthrough kwargs (`init_scale_mult`, `mlp_seed`) added purely to
     the constructor call -- no training-loop logic changed.
  2. `run_deepfm_ensemble`, a NEW function for lever 4 (ensembling multiple
     deep-part inits): trains K independent DeepFMVR models sharing the same
     `seed` (identical FM V/W init + identical BPR sampling order/schedule)
     but each with its own `mlp_seed` (so only the deep part's initial
     weights differ across members), then averages the K members' raw
     predict() logits at inference before calling evaluate() ONCE on the
     averaged score. Each member is still selected on its own best-epoch
     checkpoint (early stopping is per-member, on that member's own valid
     primary) -- this is standard "train K, average logits" ensembling, not
     a joint objective.

Note training-dynamics subtlety (documented for honesty, not a bug): although
V/W's *gradient formula* never sees the deep part's output (iter26's design
decision, unchanged here), the *value* of the shared BPR gradient signal
g = sigmoid(zpos_fm+zpos_deep - zneg_fm-zneg_deep) - 1 DOES depend on
z_deep. So members with the same `seed` but different `mlp_seed` do NOT have
byte-identical V/W trajectories -- they diverge over training because the
deep part perturbs the loss gradient magnitude at each step. This is exactly
what the dispatch instructions describe ("same data seed, different MLP init
seed") and is still a legitimate ensembling scheme; it does mean the K
members are correlated (not fully independent), which is expected to make
ensembling's variance reduction less than the naive 1/sqrt(K).
"""
import argparse, os, sys, time
import importlib.util
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate           # noqa: E402  official eval, unmodified
from baseline import sigmoid            # noqa: E402


def _load_module(name, rel_path):
    # NOTE: iter26_deepfm/{model,train}.py share basenames with this dir's own
    # {model,train}.py. Always load cross-directory files via importlib under
    # a private module name (never a bare `sys.path.insert` + `import model`/
    # `import train`) to avoid sys.modules collisions between the same-named
    # files in the two directories.
    path = os.path.join(_THIS_DIR, *rel_path.split('/'))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_iter26_train = _load_module('iter26_deepfm_train', '../iter26_deepfm/train.py')
build_pos_neg_index = _iter26_train.build_pos_neg_index  # byte-identical to iter19
sample_pairs = _iter26_train.sample_pairs                # byte-identical to iter19

_this_model = _load_module('iter30_model', 'model.py')
DeepFMVR = _this_model.DeepFMVR         # this dir's own model.py (loads iter26's model.py internally via importlib, see model.py)

_iter19_de = _load_module('iter19_data_ext', '../iter19_decay_momentum/data_ext.py')
load_ext = _iter19_de.load_ext
encode_ext = _iter19_de.encode_ext
BASE_FIELDS = _iter19_de.BASE_FIELDS
HALFLIVES = _iter19_de.HALFLIVES

DEFAULT_FEATURES = ('decay_rate_3', 'decay_act_3', 'tab', 'last1', 'lastk_rate', 'gap')


def deepfm_bpr_step(m, Xpos, Xneg):
    """Byte-for-byte identical to iter26_deepfm.train.deepfm_bpr_step (the
    FM-part gradient block is, transitively, byte-identical to iter19's
    bpr_step). Reproduced here (not imported) only because it closes over
    the model `m`'s own class methods (deep_forward/deep_backward/
    mlp_adam_step), which are inherited unmodified by DeepFMVR."""
    B = len(Xpos)
    zpos_fm, Epos, Spos = m.logits(Xpos)
    zneg_fm, Eneg, Sneg = m.logits(Xneg)
    zpos_deep, in_pos, pre_pos = m.deep_forward(Epos)
    zneg_deep, in_neg, pre_neg = m.deep_forward(Eneg)
    zpos = zpos_fm + zpos_deep
    zneg = zneg_fm + zneg_deep
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

    if m.use_deep:
        inputs = [np.concatenate([ip, iN], axis=0) for ip, iN in zip(in_pos, in_neg)]
        preacts = [np.concatenate([pp, pn], axis=0) for pp, pn in zip(pre_pos, pre_neg)]
        gWd, gbd, _dE_discarded = m.deep_backward(g, inputs, preacts)
        m.mlp_adam_step(gWd, gbd)

    return float(-np.mean(np.log(sigmoid(d) + 1e-9)))


def _prep(data_dir, feature_set, halflives, K, splits_cache):
    splits = splits_cache if splits_cache is not None else load_ext(data_dir, halflives=halflives, K=K)
    enc, dim = encode_ext(splits, feature_set=feature_set, halflives=halflives)
    return splits, enc, dim


def _train_one(m, Xtr, ytr, utr, Xva, yva, uva, bs, epochs, patience, seed, verbose, tag=''):
    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = \
        build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs)))
    user_cumw = np.cumsum(pos_len.astype(np.float64))
    user_totalw = user_cumw[-1]
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    diverged = False
    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        for _ in range(steps_per_epoch):
            Xpos_rows, Xneg_rows = sample_pairs(
                rng, n_users, bs, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
                user_cumw=user_cumw, user_totalw=user_totalw)
            losses.append(deepfm_bpr_step(m, Xtr[Xpos_rows], Xtr[Xneg_rows]))
        mean_loss = float(np.mean(losses))
        if not np.isfinite(mean_loss):
            diverged = True
            if verbose: print(f"  {tag} epoch {ep:2d} | loss NaN/Inf -- diverged, stopping")
            break
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  {tag} epoch {ep:2d} | loss {mean_loss:.4f} | valid primary {va['primary']:.4f} "
                  f"| {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b),
                           [w.copy() for w in m.mlp_W], [b.copy() for b in m.mlp_b])
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  {tag} early stop at epoch {ep}")
                break
    if diverged and best_state is None:
        return None, diverged
    m.V, m.W, m.b, m.mlp_W, m.mlp_b = best_state
    return m, diverged


def run_deepfm_bpr(data_dir, feature_set=DEFAULT_FEATURES, halflives=HALFLIVES,
                    k=16, hidden=(32,), use_deep=True, lr=0.001, mlp_lr=None, l2_mlp=1e-5,
                    init_scale_mult=1.0, mlp_seed=None,
                    epochs=40, bs=8192, patience=4, seed=0, verbose=True,
                    K=5, splits_cache=None):
    """Single-model variant-reduction runner. Same training loop as
    iter26_deepfm.train.run_deepfm_bpr; only the model class (DeepFMVR
    instead of DeepFM) and the two new constructor kwargs are different."""
    splits, enc, dim = _prep(data_dir, feature_set, halflives, K, splits_cache)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    num_fields = Xtr.shape[1]

    if verbose:
        fields = list(BASE_FIELDS) + list(feature_set)
        print(f"  fields={fields} dim={dim} num_fields={num_fields} hidden={hidden} "
              f"mlp_lr={mlp_lr} l2_mlp={l2_mlp} init_scale_mult={init_scale_mult} mlp_seed={mlp_seed}")

    m = DeepFMVR(dim, num_fields=num_fields, k=k, hidden=hidden, lr=lr, mlp_lr=mlp_lr,
                 l2_mlp=l2_mlp, use_deep=use_deep, seed=seed,
                 init_scale_mult=init_scale_mult, mlp_seed=mlp_seed)
    m, diverged = _train_one(m, Xtr, ytr, utr, Xva, yva, uva, bs, epochs, patience, seed, verbose)
    if m is None:
        return {'valid': {'GAUC': 0.0, 'nDCG@5': 0.0, 'primary': 0.0},
                'test': {'GAUC': 0.0, 'nDCG@5': 0.0, 'primary': 0.0}, 'diverged': True}
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte)), 'diverged': diverged}


def run_deepfm_ensemble(data_dir, feature_set=DEFAULT_FEATURES, halflives=HALFLIVES,
                         k=16, hidden=(32,), lr=0.001, mlp_lr=None, l2_mlp=1e-5,
                         init_scale_mult=1.0, n_members=3, mlp_seed_stride=1_000_003,
                         epochs=40, bs=8192, patience=4, seed=0, verbose=True,
                         K=5, splits_cache=None):
    """Lever 4: train `n_members` DeepFMVR models sharing `seed` (identical
    FM V/W init + identical BPR sampling schedule) but each with its own
    `mlp_seed = seed*mlp_seed_stride + member_idx` (so ONLY the deep part's
    initial weights differ), then average the raw predict() logits across
    members and evaluate the average once. use_deep is always True here
    (an ensemble of pure-FM heads would just be one FM model, nothing to
    ensemble over)."""
    splits, enc, dim = _prep(data_dir, feature_set, halflives, K, splits_cache)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    num_fields = Xtr.shape[1]

    if verbose:
        print(f"  [ensemble] n_members={n_members} hidden={hidden} mlp_lr={mlp_lr} "
              f"l2_mlp={l2_mlp} seed={seed}")

    va_scores, te_scores = [], []
    any_diverged = False
    for mi in range(n_members):
        eff_mlp_seed = seed * mlp_seed_stride + mi
        m = DeepFMVR(dim, num_fields=num_fields, k=k, hidden=hidden, lr=lr, mlp_lr=mlp_lr,
                     l2_mlp=l2_mlp, use_deep=True, seed=seed,
                     init_scale_mult=init_scale_mult, mlp_seed=eff_mlp_seed)
        m, diverged = _train_one(m, Xtr, ytr, utr, Xva, yva, uva, bs, epochs, patience, seed,
                                  verbose, tag=f'[member {mi}]')
        any_diverged = any_diverged or diverged
        if m is None:
            continue
        va_scores.append(m.predict(Xva))
        te_scores.append(m.predict(Xte))

    if not va_scores:
        return {'valid': {'GAUC': 0.0, 'nDCG@5': 0.0, 'primary': 0.0},
                'test': {'GAUC': 0.0, 'nDCG@5': 0.0, 'primary': 0.0}, 'diverged': True,
                'n_members_used': 0}

    va_avg = np.mean(np.stack(va_scores, axis=0), axis=0)
    te_avg = np.mean(np.stack(te_scores, axis=0), axis=0)
    return {'valid': evaluate(uva, yva, va_avg), 'test': evaluate(ute, yte, te_avg),
            'diverged': any_diverged, 'n_members_used': len(va_scores)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='../../KuaiRand-Pure/data')
    ap.add_argument('--features', default=','.join(DEFAULT_FEATURES))
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--hidden', default='32')
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--mlp_lr', type=float, default=None)
    ap.add_argument('--l2_mlp', type=float, default=1e-5)
    ap.add_argument('--init_scale_mult', type=float, default=1.0)
    ap.add_argument('--mlp_seed', type=int, default=None)
    ap.add_argument('--ensemble', type=int, default=0, help='if >0, run ensemble with this many members')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    feature_set = tuple(f for f in a.features.split(',') if f)
    hidden = tuple(int(x) for x in a.hidden.split(',') if x)
    if a.ensemble > 0:
        res = run_deepfm_ensemble(a.data_dir, feature_set=feature_set, k=a.k, hidden=hidden,
                                   lr=a.lr, mlp_lr=a.mlp_lr, l2_mlp=a.l2_mlp,
                                   init_scale_mult=a.init_scale_mult, n_members=a.ensemble,
                                   epochs=a.epochs, bs=a.bs, patience=a.patience, seed=a.seed,
                                   verbose=not a.quiet, K=a.K)
    else:
        res = run_deepfm_bpr(a.data_dir, feature_set=feature_set, k=a.k, hidden=hidden,
                              lr=a.lr, mlp_lr=a.mlp_lr, l2_mlp=a.l2_mlp,
                              init_scale_mult=a.init_scale_mult, mlp_seed=a.mlp_seed,
                              epochs=a.epochs, bs=a.bs, patience=a.patience, seed=a.seed,
                              verbose=not a.quiet, K=a.K)
    print(f"\n=== iter30 hidden={hidden} mlp_lr={a.mlp_lr} l2_mlp={a.l2_mlp} "
          f"init_scale_mult={a.init_scale_mult} ensemble={a.ensemble} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
