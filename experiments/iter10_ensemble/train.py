"""iter10 — ensemble of independently-trained pointwise FM and weighted-BPR FM models.

Idea: iter1 (pointwise logloss FM) and iter3 (BPR pairwise loss, activity-weighted
user sampling, same FM architecture) were each evaluated standalone and never
combined. This script trains BOTH on the SAME encoded data / SAME seed, then
blends their raw prediction scores per split via a weighted average:

    score_ensemble = w * zscore(score_pointwise) + (1-w) * zscore(score_bpr)

Each model's scores are z-score normalized independently, per split (valid and
test normalized separately, using that split's own mean/std), before blending,
since the two models' raw logit scales need not match even though they share
the same `FM.logits()` function (different loss landscapes -> different score
magnitudes/calibration).

Training-loop code for both models is a direct copy of the reference
implementations, adapted only to also return the trained `FM` object (not just
the `evaluate()` dict) so we can call `.predict()` on it ourselves after the
fact for ensembling:
  - pointwise: copied from `baseline.run_fm` (baseline.py, repo root)
  - BPR:       copied from `experiments/iter3_bpr_weighted/train.py`'s
               `run_bpr`, reusing its helpers (`build_pos_neg_index`,
               `sample_pairs`, `bpr_step`) via direct import.

Usage:
  python train.py --data_dir ../../KuaiRand-Pure/data --stage sweep   # w-sweep, seeds 0,1,2
  python train.py --data_dir ../../KuaiRand-Pure/data --stage final --w 0.5  # 5-seed final run
  python train.py --data_dir ../../KuaiRand-Pure/data --stage all     # both, auto-picks w
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from data import load, encode, FIELDS
from evaluate import evaluate
from baseline import FM, sigmoid

# --- import experiments/iter3_bpr_weighted/train.py as its own module (not
# reusing the top-level 'train' module name, to avoid clashing with this file
# if something ever imports both) ---
_IT3_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'iter3_bpr_weighted', 'train.py')
_spec = importlib.util.spec_from_file_location('iter3_bpr_train', _IT3_PATH)
iter3_bpr_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iter3_bpr_train)
build_pos_neg_index = iter3_bpr_train.build_pos_neg_index
sample_pairs = iter3_bpr_train.sample_pairs
bpr_step = iter3_bpr_train.bpr_step


# ---------------- pointwise FM training (copy of baseline.run_fm, + returns model) ----------------
def run_fm_model(enc, dim, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad, best_ep = -1, None, 0, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  [pointwise] epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad, best_ep = va['primary'], 0, ep
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  [pointwise] early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return m, best_ep


# ---------------- BPR FM training (copy of iter3_bpr_weighted.train.run_bpr, + returns model) ----------------
def run_bpr_model(enc, dim, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = \
        build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs)))

    user_cumw = np.cumsum(pos_len.astype(np.float64))
    user_totalw = user_cumw[-1]

    if verbose:
        print(f"  [bpr] eligible train users: {n_users} | {steps_per_epoch} steps/epoch | "
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
            print(f"  [bpr] epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad, best_ep = va['primary'], 0, ep
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  [bpr] early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return m, best_ep


# ---------------- ensembling helpers ----------------
def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd + 1e-9)


def train_seed(splits, enc, dim, seed, k=16, lr=0.001, epochs=40, bs=8192, patience=4, verbose=True):
    """Train both models for one seed on the same encoded data. Returns a dict with
    raw scores (valid/test, both models) and each model's standalone evaluate() dict."""
    Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    t0 = time.time()
    m_pw, ep_pw = run_fm_model(enc, dim, k=k, lr=lr, epochs=epochs, bs=bs, patience=patience,
                                seed=seed, verbose=verbose)
    t1 = time.time()
    m_bpr, ep_bpr = run_bpr_model(enc, dim, k=k, lr=lr, epochs=epochs, bs=bs, patience=patience,
                                   seed=seed, verbose=verbose)
    t2 = time.time()
    print(f"seed {seed}: pointwise trained in {t1-t0:.1f}s (best ep {ep_pw}), "
          f"bpr trained in {t2-t1:.1f}s (best ep {ep_bpr})")

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
    """Given a train_seed() result, compute ensembled valid/test evaluate() dicts
    for blend weight w (weight on the pointwise model; 1-w on bpr)."""
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

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    enc, dim = encode(splits)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS} dim={dim}")

    ws = [float(x) for x in a.ws.split(',')]
    sweep_seeds = [int(x) for x in a.sweep_seeds.split(',')]
    final_seeds = [int(x) for x in a.final_seeds.split(',')]

    # union of seeds we actually need to train (avoid retraining seeds shared
    # between sweep and final, e.g. default sweep={0,1,2} subset of final={0..4})
    need_seeds = sorted(set(sweep_seeds) | (set(final_seeds) if a.stage in ('final', 'all') else set()))
    if a.stage == 'sweep':
        need_seeds = sorted(set(sweep_seeds))

    trained = {}
    for seed in need_seeds:
        print(f"\n=== training seed {seed} (pointwise + bpr) ===")
        trained[seed] = train_seed(splits, enc, dim, seed, k=a.k, lr=a.lr, epochs=a.epochs,
                                    bs=a.bs, patience=a.patience, verbose=verbose)
        pw, bp = trained[seed]['pointwise'], trained[seed]['bpr']
        print(f"  seed {seed} STANDALONE pointwise: valid {pw['valid']['primary']:.4f} "
              f"test {pw['test']['primary']:.4f} (best ep {pw['best_epoch']})")
        print(f"  seed {seed} STANDALONE bpr:       valid {bp['valid']['primary']:.4f} "
              f"test {bp['test']['primary']:.4f} (best ep {bp['best_epoch']})")

    out = {'sweep': None, 'winning_w': None, 'final': None, 'standalone': {}}

    # record standalone (pointwise/bpr) metrics for every trained seed
    for seed, res in trained.items():
        out['standalone'][seed] = {
            'pointwise': {'valid': res['pointwise']['valid'], 'test': res['pointwise']['test'],
                          'best_epoch': res['pointwise']['best_epoch']},
            'bpr': {'valid': res['bpr']['valid'], 'test': res['bpr']['test'],
                    'best_epoch': res['bpr']['best_epoch']},
        }

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
                print(f"\n=== training seed {seed} (pointwise + bpr) [final-only] ===")
                trained[seed] = train_seed(splits, enc, dim, seed, k=a.k, lr=a.lr, epochs=a.epochs,
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
