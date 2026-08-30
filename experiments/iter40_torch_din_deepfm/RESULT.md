# iter40: end-to-end differentiable DIN attention + DeepFM tower (PyTorch)

## Hypothesis

iter32/34 (numpy) tested user-history attention as a **frozen, pretrained,
feature-ized** signal and found only a small gain that didn't compose with
the current best config (overlapping information with the existing
recency-decay features). The natural follow-up, flagged as untried in the
handover doc's own "unexplored directions" list, is a **genuinely
end-to-end-trained** attention mechanism — gradients flow from the BPR loss
straight into the shared video/author embedding table, rather than freezing
it. This requires autodiff, so this iteration also pivots the stack from
numpy to PyTorch (confirmed in-scope: the handover doc explicitly allows
"any open-source library/framework"; the numpy-only convention was
self-imposed by prior iterations, not a competition rule).

Combined with this: a DeepFM-style deep MLP tower over the concatenated
field embeddings, to also pick up the organizer-suggested "model swap"
direction in the same iteration.

## Method

- `data_prep.py`: causal, leak-free per-row history sequences (video_id,
  author_id, label, mask), reusing iter27's exact proven feature set and the
  established `(time_ms, orig_idx)` causal-ordering discipline. Verified with
  a brute-force causality spot-check (30 random rows, 0 mismatches).
- `model.py` (`FMDeepDIN`): FM wide term (exact `baseline.py` formula) +
  three optional additions, individually toggleable:
  - `use_attention`: DIN-style local-activation-unit attention (small MLP
    scorer over `[hist, target, hist*target, hist-target, hist_label]`,
    masked softmax, explicit empty-history zero-handling).
  - `use_deep`: MLP tower over all field embeddings (+ attention output when
    enabled).
  - `use_attn_direct`: a **low-capacity alternative** to routing attention
    through the deep tower — a single learned linear layer over the
    elementwise product of the attention-pooled user-interest vector and the
    target representation, added directly to the FM logit. Zero-initialized
    so it starts as a no-op.
- `train.py`: BPR loss (`-logsigmoid(pos-neg)`), reusing iter27's exact
  decay-weighted user-sampling infrastructure (`build_pos_neg_index`,
  `sample_pairs`) verbatim. Adam optimizer, MPS (Apple GPU) device, ~1-9s/epoch
  depending on config.

## Harness-fidelity check (mandatory before trusting anything new)

`use_attention=False, use_deep=False` (plain FM+BPR, torch) vs. the known-good
numpy iter27 reference:

| | valid primary | test primary |
|---|---|---|
| numpy iter27 reference (5-seed mean) | ~0.638 | 0.63889 |
| torch reimplementation (seed 0) | 0.63892 | 0.63972 |

Matches closely. **Harness confirmed trustworthy.**

## Results

All single-seed unless noted; same feature set, same sampling scheme,
20 epochs w/ early stopping (patience 4), differing only in the
architecture flags under test.

| Config | valid primary | test primary |
|---|---|---|
| Plain FM (no attention, no deep) — reference | 0.6389 | 0.6397 |
| Attention feeding **full deep MLP** (128,64), dropout 0.1 | 0.6354 | 0.6352 |
| Attention feeding smaller deep MLP (64,), dropout 0.4 | 0.6373 | 0.6353 |
| Attention feeding smaller deep MLP (32,), dropout 0.3 | 0.6369 | 0.6372 |
| Deep MLP alone, no attention | 0.6347 | 0.6300 |
| **Direct attention head** (bypasses deep MLP entirely), L=40 | 0.6396 | 0.6399 |
| Direct attention head + deep MLP (64,), dropout 0.3 | 0.6373 | 0.6338 |
| Direct attention head, L=100 (longer window) | 0.6397 / 0.6373 (2 seeds) | 0.6401 / 0.6383 |

**Every deep-MLP variant underperforms plain FM.** This mirrors the earlier
multi-task-learning finding in this project (SUBMISSION.md): pushing a
second, higher-capacity gradient path through the *same* shared embedding
table that the FM's own bilinear term depends on conflicts with the
rank-invariant BPR objective, regardless of regularization strength (tried
dropout up to 0.4, weight_decay up to 1e-4, smaller hidden layers down to
32 units — all still below the plain-FM baseline).

**5-seed confirmation of the direct-attention head** (the only variant that
edged out the baseline on a single seed):

| | valid mean | valid std | test mean | test std |
|---|---|---|---|---|
| Baseline FM (no attention) | 0.63767 | 0.00069 | 0.63862 | 0.00063 |
| Direct attention head | 0.63801 | 0.00086 | 0.63896 | 0.00056 |

Gain: **+0.00034 valid, +0.00034 test** — below the project's established
0.001 promotion threshold, and smaller than one standard deviation of either
config's own seed-to-seed noise. **Not a real gain.**

A longer attention window (L=100 vs. the original L=40, matched to iter32's
winning window) produced no meaningful difference (2-seed spot check:
0.6397/0.6373 at L=100 vs. 0.6396/0.6372 at L=40) — ruling out window length
as the limiting factor.

## Diagnosis

This is the **third independent test** of user-history sequence modeling in
this project (iter32: frozen dot-product attention feature; iter34: fusing
it with the current best; iter40: two forms of genuinely end-to-end
differentiable attention, one deep-MLP-routed and one direct-bilinear). All
three land in the same place: a small, noise-level gain that does not
compose additively with the existing recency-decay features (`decay_rate`,
`decay_act`, `decay_tab`, `last1`, `lastk_rate`, `gap`). The most likely
explanation, consistent across all three attempts: those decay features
already summarize "how has this user been behaving recently" well enough
that a learned attention mechanism over the same underlying (video, author,
label) history has little new information left to extract, regardless of
whether the attention weights are frozen or fully trained end-to-end.

## Verdict: **REJECT**

Both the DIN-attention and DeepFM-tower directions are closed for this
feature set — attention modeling has now been tried in 3 forms (frozen
feature, differentiable+deep, differentiable+direct) with consistent
noise-level results, and the deep tower actively hurts regardless of
regularization. iter27+iter38 (FM+BPR+decay features+ensembling, test
primary 0.64187) remains the best model.

## What this implies for the 0.70 target

Every materially different architecture tried across this project's history
— FM+BPR variants, DeepFM, multi-task learning (×2), frozen attention
feature, and now end-to-end differentiable DIN attention (×2 forms) — lands
in the same 0.636-0.642 test-primary band. This is evidence of a real
plateau for this feature representation + FM-family model class, not a
string of implementation misses (every one of these was harness-verified
and, where relevant, gradient-verified before being trusted). Closing the
remaining ~0.06 gap to 0.70 likely needs a lever that hasn't been tried yet
and is structurally different from "add a mechanism on top of FM," e.g.
hard-negative BPR mining (sharpen the pairwise objective itself) or a
non-FM-family model. Continuing to vary attention/deep-tower hyperparameters
further is not expected to move the needle based on this evidence.

## Code

[`data_prep.py`](data_prep.py), [`model.py`](model.py), [`train.py`](train.py)
