"""iter13 — stack iter10's ensembling idea ON TOP OF iter9's winning extended
feature set (activity+tab+rate), instead of on top of iter3's plain features.

iter10 found a small but real gain (+0.00095 test) from blending an
independently-trained pointwise FM with an independently-trained
activity-weighted-BPR FM, both fed the PLAIN 5-field encoding (data.py).
iter9 separately found a much larger gain (+0.0090 test) from feeding BPR the
extended (activity+tab+rate causal history) encoding from data_ext.py.

This script asks: does ensembling pointwise + BPR still help once BOTH models
are fed iter9's extended features, or does iter9's feature set already
capture whatever complementary signal the ensemble used to add?

Reuses:
  - experiments/iter9_history_dense/data_ext.py's load_ext/encode_ext
    (imported directly, NOT copied/modified) for the extended encoding.
  - experiments/iter9_history_dense/train.py's build_pos_neg_index/
    sample_pairs/bpr_step helpers (imported directly) for the BPR loop —
    identical to iter3's, copied verbatim there too.
  - baseline.FM / baseline.sigmoid (unmodified) as the model class.
  - evaluate.py's evaluate() (unmodified) as the official metric.

New code in this file (mirrors iter10_ensemble/train.py's structure exactly,
just swapping the plain encoding for the extended one):
  - run_fm_ext_model: pointwise training loop on the EXTENDED encoding
    (mirrors baseline.run_fm / iter10.run_fm_model, but returns the trained
    FM object so we can call .predict() ourselves for ensembling, and uses
    encode_ext's (X,y,users) tuples instead of encode()'s).
  - run_bpr_ext_model: BPR training loop on the EXTENDED encoding (same
    logic as iter9_history_dense/train.py's run_bpr_ext, just also returns
    the model object instead of only the evaluate() dict).
  - ensembling: z-score each model's raw scores per split, blend
    w*pointwise + (1-w)*bpr, sweep w in {0.25,0.5,0.75} like iter10.

Usage:
  python train.py --data_dir ../../KuaiRand-Pure/data --stage sweep   # w-sweep, seeds 0,1,2
  python train.py --data_dir ../../KuaiRand-Pure/data --stage final --w 0.5  # 5-seed final run
  python train.py --data_dir ../../KuaiRand-Pure/data --stage all     # both, auto-picks w
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from evaluate import evaluate            # noqa: E402  official eval, unmodified
from baseline import FM, sigmoid         # noqa: E402  same FM class as iter9/iter10

_IT9_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'iter9_history_dense')
sys.path.insert(0, _IT9_DIR)
from data_ext import load_ext, encode_ext, BASE_FIELDS  # noqa: E402  reused unmodified

# import iter9's train.py as its own module to reuse its BPR-loop helpers
# (identical to iter3's — copied verbatim there too), same pattern iter10
# used for iter3's helpers.
_IT9_TRAIN_PATH = os.path.join(_IT9_DIR, 'train.py')
_spec = importlib.util.spec_from_file_location('iter9_ext_train', _IT9_TRAIN_PATH)
iter9_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iter9_train)
build_pos_neg_index = iter9_train.build_pos_neg_index
sample_pairs = iter9_train.sample_pairs
bpr_step = iter9_train.bpr_step

FEATURE_SET = ('activity', 'tab', 'rate')  # iter9's winning combo


# ---------------- pointwise FM training on EXTENDED features ----------------
def run_fm_ext_model(enc, dim, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad, best_ep = -1, None, 0, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  [pointwise-ext] epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad, best_ep = va['primary'], 0, ep
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  [pointwise-ext] early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return m, best_ep


# ---------------- BPR FM training on EXTENDED features ----------------
def run_bpr_ext_model(enc, dim, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = \
        build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs)))

    user_cumw = np.cumsum(pos_len.astype(np.float64))
    user_totalw = user_cumw[-1]

    if verbose:
        print(f"  [bpr-ext] eligible train users: {n_users} | {steps_per_epoch} steps/epoch | "
              f"activity-weighted user sampling")

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad, best_ep = -1, None, 0, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        for _ in range(steps_per_epoch):
            Xpos_rows, Xneg_rows = sample_pairs(
                rng, n_users, bs, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
                user_cumw=user_cumw, user_totalw=user_totalw)
            losses.append(bpr_step(m, Xtr[Xpos_rows], Xtr[Xneg_rows]))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  [bpr-ext] epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad, best_ep = va['primary'], 0, ep
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  [bpr-ext] early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return m, best_ep


# ---------------- ensembling helpers ----------------
def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd + 1e-9)


def train_seed(enc, dim, seed, k=16, lr=0.001, epochs=40, bs=8192, patience=4, verbose=True):
    """Train both models (pointwise-ext, bpr-ext) for one seed on the same
    encoded extended data. Returns raw scores + standalone evaluate() dicts."""
    Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    t0 = time.time()
    m_pw, ep_pw = run_fm_ext_model(enc, dim, k=k, lr=lr, epochs=epochs, bs=bs, patience=patience,
                                    seed=seed, verbose=verbose)
    t1 = time.time()
    m_bpr, ep_bpr = run_bpr_ext_model(enc, dim, k=k, lr=lr, epochs=epochs, bs=bs, patience=patience,
                                       seed=seed, verbose=verbose)
    t2 = time.time()
    print(f"seed {seed}: pointwise-ext trained in {t1-t0:.1f}s (best ep {ep_pw}), "
          f"bpr-ext trained in {t2-t1:.1f}s (best ep {ep_bpr})")

    sp_va, sp_te = m_pw.predict(Xva), m_pw.predict(Xte)
    sb_va, sb_te = m_bpr.predict(Xva), m_bpr.predict(Xte)

    pw_valid = evaluate(uva, yva, sp_va); pw_test = evaluate(ute, yte, sp_te)
    bpr_valid = evaluate(uva, yva, sb_va); bpr_test = evaluate(ute, yte, sb_te)

    return {
        'seed': seed,
        'raw': {'sp_va': sp_va, 'sp_te': sp_te, 'sb_va': sb_va, 'sb_te': sb_te},
        'pointwise': {'valid': pw_valid, 'test': pw_test, 'best_epoch': ep_pw},
        'bpr': {'valid': bpr_valid, 'test': bpr_test, 'best_epoch': ep_bpr},
    }


def ensemble_eval(seed_result, enc, w):
    """w = weight on pointwise-ext model, (1-w) on bpr-ext model."""
    _, yva, uva = enc['valid']; _, yte, ute = enc['test']
    raw = seed_result['raw']
    va_score = w * zscore(raw['sp_va']) + (1 - w) * zscore(raw['sb_va'])
    te_score = w * zscore(raw['sp_te']) + (1 - w) * zscore(raw['sb_te'])
    return {'valid': evaluate(uva, yva, va_score), 'test': evaluate(ute, yte, te_score)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='../../KuaiRand-Pure/data')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--stage', default='all', choices=['sweep', 'final', 'all'])
    ap.add_argument('--sweep_seeds', default='0,1,2')
    ap.add_argument('--final_seeds', default='0,1,2,3,4')
    ap.add_argument('--ws', default='0.25,0.5,0.75')
    ap.add_argument('--w', type=float, default=None, help='fixed w for --stage final')
    ap.add_argument('--out', default='results.json')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    verbose = not a.quiet

    print(f"loading {a.data_dir} (extended, features={FEATURE_SET}) ...")
    splits = load_ext(a.data_dir)
    enc, dim = encode_ext(splits, feature_set=FEATURE_SET)
    print({k_: len(v) for k_, v in splits.items()},
          f"fields={list(BASE_FIELDS)+list(FEATURE_SET)} dim={dim}")

    ws = [float(x) for x in a.ws.split(',')]
    sweep_seeds = [int(x) for x in a.sweep_seeds.split(',')]
    final_seeds = [int(x) for x in a.final_seeds.split(',')]

    need_seeds = sorted(set(sweep_seeds) | (set(final_seeds) if a.stage in ('final', 'all') else set()))
    if a.stage == 'sweep':
        need_seeds = sorted(set(sweep_seeds))

    trained = {}
    for seed in need_seeds:
        print(f"\n=== training seed {seed} (pointwise-ext + bpr-ext) ===")
        trained[seed] = train_seed(enc, dim, seed, k=a.k, lr=a.lr, epochs=a.epochs,
                                    bs=a.bs, patience=a.patience, verbose=verbose)
        pw, bp = trained[seed]['pointwise'], trained[seed]['bpr']
        print(f"  seed {seed} STANDALONE pointwise-ext: valid {pw['valid']['primary']:.4f} "
              f"test {pw['test']['primary']:.4f} (best ep {pw['best_epoch']})")
        print(f"  seed {seed} STANDALONE bpr-ext:       valid {bp['valid']['primary']:.4f} "
              f"test {bp['test']['primary']:.4f} (best ep {bp['best_epoch']})")

    out = {'sweep': None, 'winning_w': None, 'final': None, 'standalone': {}}

    for seed, res in trained.items():
        out['standalone'][seed] = {
            'pointwise': {'valid': res['pointwise']['valid'], 'test': res['pointwise']['test'],
                          'best_epoch': res['pointwise']['best_epoch']},
            'bpr': {'valid': res['bpr']['valid'], 'test': res['bpr']['test'],
                    'best_epoch': res['bpr']['best_epoch']},
        }
        # save incrementally so partial progress survives interruption
        def to_native(o):
            if isinstance(o, dict):
                return {str(k): to_native(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [to_native(v) for v in o]
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            return o
        with open(a.out, 'w') as fh:
            json.dump(to_native(out), fh, indent=2)

    winning_w = a.w
    if a.stage in ('sweep', 'all'):
        sweep_table = {}
        for w in ws:
            vals = []
            for seed in sweep_seeds:
                r = ensemble_eval(trained[seed], enc, w)
                vals.append(r['valid']['primary'])
            sweep_table[w] = {'per_seed_valid_primary': vals, 'mean_valid_primary': float(np.mean(vals))}
            print(f"w={w}: valid primary per seed {[f'{v:.4f}' for v in vals]} -> mean {np.mean(vals):.5f}")
        out['sweep'] = sweep_table
        winning_w = max(sweep_table, key=lambda w: sweep_table[w]['mean_valid_primary'])
        out['winning_w'] = winning_w
        print(f"\nwinning w (by mean valid primary over seeds {sweep_seeds}): {winning_w}")

    if a.stage in ('final', 'all'):
        if winning_w is None:
            raise SystemExit("must supply --w for --stage final, or run --stage sweep/all first")
        final_table = {}
        for seed in final_seeds:
            if seed not in trained:
                print(f"\n=== training seed {seed} (pointwise-ext + bpr-ext) [final-only] ===")
                trained[seed] = train_seed(enc, dim, seed, k=a.k, lr=a.lr, epochs=a.epochs,
                                            bs=a.bs, patience=a.patience, verbose=verbose)
                out['standalone'][seed] = {
                    'pointwise': {'valid': trained[seed]['pointwise']['valid'],
                                  'test': trained[seed]['pointwise']['test'],
                                  'best_epoch': trained[seed]['pointwise']['best_epoch']},
                    'bpr': {'valid': trained[seed]['bpr']['valid'],
                            'test': trained[seed]['bpr']['test'],
                            'best_epoch': trained[seed]['bpr']['best_epoch']},
                }
            r = ensemble_eval(trained[seed], enc, winning_w)
            final_table[seed] = r
            print(f"seed {seed} ensemble (w={winning_w}): valid primary {r['valid']['primary']:.4f} "
                  f"| test primary {r['test']['primary']:.4f}")
        out['final'] = {'w': winning_w, 'per_seed': final_table}
        valid_vals = [final_table[s]['valid']['primary'] for s in final_seeds]
        test_vals = [final_table[s]['test']['primary'] for s in final_seeds]
        print(f"\n=== ENSEMBLE (w={winning_w}), {len(final_seeds)} seeds ===")
        print(f"valid primary: mean {np.mean(valid_vals):.5f} std {np.std(valid_vals):.5f}")
        print(f"test  primary: mean {np.mean(test_vals):.5f} std {np.std(test_vals):.5f}")

    def to_native(o):
        if isinstance(o, dict):
            return {str(k): to_native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_native(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    with open(a.out, 'w') as fh:
        json.dump(to_native(out), fh, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == '__main__':
    main()
