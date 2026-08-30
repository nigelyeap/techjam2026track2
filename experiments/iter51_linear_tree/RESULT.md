# iter51 — LightGBM `linear_tree=True` at num_leaves=2

## Motivation

At `num_leaves=2`, every tree in iter44's GBM makes exactly one split and
predicts a flat constant on each side — a piecewise-*constant* step
function per tree. LightGBM's `linear_tree=True` option instead fits a
(regularized) linear regression per leaf, so with only 2 leaves the tree
becomes a genuine piecewise-*linear* function: one split, but each side
gets its own linear model over the continuous features instead of a flat
constant. This is a structural change to what a "split" buys the model —
distinct from every hyperparameter (iter44/46), boosting-algorithm
(iter50), feature (iter48), and constraint (iter49) variant tried in Round
15, and a natural fit for the un-bucketed continuous features iter44's
native encoding already exposes (`duration_ms`, the three decay features,
`lastk_rate`, `gap`).

## Method

Single-axis swap on iter44's exact pipeline/hyperparameters
(`num_leaves=2, learning_rate=0.05, n_estimators=500,
min_child_samples=200, reg_lambda=1.0`): add `linear_tree=True` to the
`LGBMRanker` constructor, nothing else changed. Harness-checked with
`linear_tree=False` first to confirm exact reproduction of iter44's
0.66135/0.64794 before testing `linear_tree=True`.

## Result — standalone GBM

Single run (seed=0) cleared both the 0.0003 look-threshold and the 0.001
confirmed-gain threshold immediately (+0.00797 valid), triggering the
mandated 5-seed confirmation:

| seed | valid | test |
|---|---|---|
| 0 | 0.66932 | 0.65146 |
| 1 | 0.66922 | 0.65149 |
| 2 | 0.66943 | 0.65133 |
| 3 | 0.66917 | 0.65136 |
| 4 | 0.66915 | 0.65135 |
| **mean** | **0.66926** | **0.65140** |
| range | 0.66915–0.66943 | 0.65133–0.65149 |

Baseline (`linear_tree=False`, i.e. iter44 standalone GBM): valid=0.66135,
test=0.64794.

**Gain: +0.0079 valid, tighter across seeds (range 0.00028) than iter44's
own seed variance.** This is the first genuine standalone gain over
iter44's GBM found across the entire "own track" (iter45–51).
**Verdict: PROMOTE** (standalone GBM).

## Result — blend with FM ensemble

Question: does this new GBM also blend better with the FM 5-seed ensemble
than iter44's original (constant-leaf) GBM did? Reused the exact
FM-training and alpha-sweep pattern from iter44/iter47 (FM trained via
`iter47/stack.py`'s `train_one_fm`, alpha swept over `[0.0, 0.40]` step
0.02, selected on valid).

| | valid | test |
|---|---|---|
| GBM standalone (`linear_tree=True`, seed=0) | 0.66932 | 0.65146 |
| FM 5-seed ensemble standalone | 0.63988 | 0.64187 |
| **iter51 blend (best alpha=0.08)** | **0.67297** | **0.65643** |
| iter44 blend (old GBM, alpha=0.10) — current final submission | 0.66473 | 0.65197 |

**Gain over the current final submission: +0.00824 valid, +0.00446 test.**
The optimal alpha shifted slightly (0.10 → 0.08, still low FM weight,
consistent with iter44's finding that the GBM dominates the blend), and
the improved GBM alone accounts for the entire gain — FM's standalone
score is unchanged from prior runs.

**Verdict: PROMOTE** (blend). This is a new best result, pending the
user's explicit go-ahead to promote it into the actual submission
deliverables (`SUBMISSION.md`, `make_submission.py`, `submission.csv`) —
not done unilaterally as part of this experiment.

## Files

- `train.py` — standalone GBM harness-check + 5-seed confirmation.
- `blend.py` — FM ensemble training + alpha sweep + blend evaluation.
- `run.log` — full standalone 5-seed output.
- `blend_results.json` — machine-readable blend result.

## Notes

`blend.py`'s first run crashed on a `train` module-name collision (fixed
in `train.py` by loading iter44's module via
`importlib.util.spec_from_file_location` under a globally-unique name,
rather than a plain `import train as gbm44` that could resolve to a
different experiment directory's `train.py` already cached in
`sys.modules`). See `LEDGER.md` for detail if this pattern recurs
elsewhere.
