# iter36 — multi-task learning v2: per-task linear head (fixes iter31's diagnosed flaw)

## Idea

iter31 (first multi-task attempt) shared the LITERAL raw FM score `z`
between the rank-invariant BPR loss and 5 base-rate-calibrated pointwise
BCE losses (`is_click`/`is_like`/`is_follow`/`is_comment`/`is_forward`) and
regressed monotonically at every weight tested (0.1→0.3), worst at the
highest weight. iter31's own diagnosis named the likely fix: give each
auxiliary task its own linear head (own bias + own linear weight row),
sharing ONLY the FM embedding matrix `V` — decoupling each task's
calibration/scale from the main task's `b`/`W`. This iteration builds and
tests that fix on top of iter27 (current best: features+decay-aware
sampling+formula constants fused).

## Method

`experiments/iter36_multitask_v2/{data_ext.py,train.py,driver.py}`:
`data_ext.py` = iter27's, verbatim, plus iter31's `load_aux_labels`/
`AUX_LABELS` appended unmodified. `train.py` = iter27's, verbatim, plus a
new `mtl2_step`: each aux task `t` gets its own `Waux[t]`/`baux[t]` (own
Adam moments), and its logit `z_aux_t = baux[t] + Waux[t]·x + inter(V)`
reuses the SAME pairwise interaction term (`inter`, built from the shared
`E`/`S` already computed for the main BPR forward pass — no extra forward
pass). Backward: each task's BCE gradient scatters into its own
`Waux[t]`/`baux[t]` only; `V`'s gradient gets the main-task contribution
plus the mean-over-tasks aux contribution (so `aux_weight`'s scale doesn't
depend on `n_aux`); `W`/`b` (main task) get ONLY the main-task contribution
— this is the sole structural change from `bpr_step`/iter31's shared-score
design. `bpr_step` itself is untouched (used when `aux_weight=0`).

**Gradient correctness, verified before any real run**: a standalone
finite-difference check (`scratchpad/grad_check_iter36.py`, tiny synthetic
FM, `dim=8,k=3,B=5,n_aux=2`) initially caught a real bug — `gWaux`/`gbaux`
were missing a `1/n_aux` factor (each task's own gradient must be scaled
consistently with the `mean-over-tasks` convention used for `V`'s
contribution, since both derive from one scalar total loss). After adding
the missing factor, all 5 gradients (`gV`,`gW`,`gb`,`gWaux`,`gbaux`) matched
numerical gradients to ≤3e-10 max abs error. Fixed formula is what actually
ran (see `train.py`'s `mtl2_step` comment).

Leakage argument: identical to iter31's (aux labels indexed only by
`Xpos_rows`/`Xneg_rows`, which are themselves indices into `enc['train']`
only; never concatenated into any feature matrix; never read anywhere in
the `evaluate()` call path on valid/test).

## Harness-fidelity check

`aux_weight=0`, 5 seeds: bit-exact match to iter27's published
`fusion_sampling_alpha0.75` numbers (max abs Δ = 0.0 on both valid and
test, all 5 seeds) — independently re-verified against `results.json`.

## Sweep (3 seeds, valid-only selection)

| aux_weight | valid mean (3-seed) | test mean (3-seed) | Δ valid vs iter27 (0.63816) |
|---|---|---|---|
| 0.01 | 0.63671 | 0.63838 | **−0.00145** |
| 0.03 | 0.63546 | 0.63786 | **−0.00269** |
| 0.10 | 0.63299 | 0.63576 | **−0.00517** |

Monotonic regression, both splits, at every weight tested, worst at the
highest weight — the same qualitative pattern iter31 found. Not extended
to 5 seeds (protocol threshold is for a candidate *beating* the reference).
Orchestrator independently hand-verified all 4 tags' means directly from
`results.json` (bit-exact match to driver output, including the
harness-fidelity row).

## Diagnosis

The architectural fix (per-task linear heads, decoupled calibration) does
**not** rescue multi-task learning here — regression persists even at
`aux_weight=0.01`, an order of magnitude below where iter31 saw damage.
This localizes the failure mode one level deeper than iter31's diagnosis:
it is not merely that sharing the raw score conflates two different
calibration scales — it is that pushing *any* gradient signal from
base-rate-calibrated pointwise engagement losses into the SHARED embedding
matrix `V` conflicts with the purely rank-invariant BPR objective, because
`V` is where all of the model's representational capacity for
interactions lives. Decoupling the linear head removes one source of
conflict (scale) but not the deeper one (the embeddings themselves get
pulled toward optimizing for absolute engagement probability rather than
relative long-view ranking). A design that avoided touching `V` at all —
e.g. task-specific embeddings, or freezing `V` during the auxiliary
update — would be the natural next step, but is out of scope given the
now twice-confirmed direction (both the organizer-named "multi-task"
avenue's two most natural designs have been tried and both regress).

## Verdict: REJECT

iter27 remains current best. This closes the multi-task learning direction
for this model family: two independently-designed, gradient-verified
attempts (shared score, per-task head) both regress monotonically, with a
diagnosed common root cause (embedding-space conflict between rank-
invariant and pointwise-calibrated objectives) rather than an
implementation artifact in either.

## Code

`experiments/iter36_multitask_v2/{data_ext.py,train.py,driver.py}`, raw
results in `experiments/iter36_multitask_v2/results.json`, gradient check
in `scratchpad/grad_check_iter36.py` (not part of the experiment, kept for
reference at
`/private/tmp/claude-501/-Users-nigelyeap/0b355fc9-bbc3-4b4d-88ee-ee430c3f4e64/scratchpad/grad_check_iter36.py`).
