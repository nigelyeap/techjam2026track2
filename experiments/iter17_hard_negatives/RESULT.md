# iter17 — Informed / hard negative sampling for BPR

## What "negative" actually meant before this iteration

Before writing any new code, I read `iter9_history_dense/train.py` and
`iter3_bpr_weighted/train.py` in full to pin down exactly what the existing
BPR negative-sampling mechanism does, since the iteration brief's framing
("the negative video has always been drawn uniformly at random from the
full video vocabulary") turns out to be **not what the code does**.

The actual mechanism (`build_pos_neg_index` + `sample_pairs`, unchanged
since iter2/iter3, still used verbatim by iter9): for each user `u`, two
row-index lists are built from the TRAIN split — `by_user_pos[u]` (rows
where `u`'s label==1) and `by_user_neg[u]` (rows where `u`'s label==0).
Each BPR step samples a user (activity/pos_len-weighted since iter3), then
samples ONE ROW uniformly from **that same user's own** positive list and
ONE ROW uniformly from **that same user's own** negative list. So the
"negative" has always been *a video this exact user was actually exposed to
and did not long-view* — a verified, observed negative label — drawn
uniformly from that user's own negative history. It has never been an
arbitrary/unrelated video sampled from the global vocabulary independent of
what the user actually saw. This distinction turns out to be the whole
story of this iteration's result (see Discussion below).

## What was implemented

`iter17_hard_negatives/train.py` (+ `driver.py` for an efficient shared-data
sweep) adds a `negative_mode` axis on top of iter9's exact feature set
(`activity`, `tab`(`tab_pos`), `rate`) and exact activity-weighted user
sampling — nothing else changes. `data_ext.py` re-exports
`iter9_history_dense/data_ext.py`'s `compute_causal_features`/`load_ext`/
`encode_ext` unmodified (loaded via `importlib` to avoid a module-name
collision with this directory's own `data_ext.py`).

Four modes, all built to be O(1)-per-sample using structures precomputed
**once** from the train split (`build_negative_pools`):

- **`uniform`** (baseline / parity check): exactly iter9/iter3's mechanism,
  unchanged — negative row from the sampled user's own `label==0` rows.
- **`same_tab`**: negative's item-side columns (`video_id`, `author_id`,
  `tab`, `dur_bucket`) come from a uniformly-random **unique video** shown
  (to *any* user, anywhere in train) under the *same* `tab` as the sampled
  positive row. One representative row per unique video per tab is
  precomputed so every unique video in the tab has equal probability
  (i.e. NOT popularity-weighted). User-side columns (`user_id` +
  activity/tab_pos/rate) are copied from the positive row (same user, same
  point in time). Falls back to a global uniform-over-all-unique-videos pool
  when a tab's candidate pool has fewer than `min_tab_pool=20` unique
  videos.
- **`pop_weighted`**: negative's item-side columns come from a **uniformly
  random TRAIN ROW** (any tab/user). This needs zero extra bookkeeping,
  because sampling a row uniformly makes a video's selection probability
  exactly proportional to its row-count in train (its exposure/popularity) —
  a video with N rows is N× more likely to be hit by a uniform row draw than
  a video with 1 row. User-side columns copied from the positive row as
  above.
- **`same_tab_pop_weighted`** (the optional 4th variant): same idea as
  `same_tab` but sampling a uniformly-random ROW within the tab's full
  row-pool (not deduplicated to unique videos), making video selection
  popularity-weighted *within* the tab. Same fallback mechanism as
  `same_tab`.

Known simplification (documented, not hidden): for all three non-uniform
modes, the `tab_pos` causal feature — itself defined per (user, tab) — is
carried over from the positive row's own tab even when the candidate video's
tab differs from the positive's. This only actually happens in
`pop_weighted` mode (both `same_tab*` modes match tabs by construction), and
avoiding it would require a full per-(user, tab, date) re-derivation at
sample time, which was judged not worth the added per-step cost for a
mode that (see below) was already the worst performer.

## Fallback-rate stats (same-tab modes)

Precomputed per-tab unique-video pool sizes (train split, 15 tabs total):

```
7231, 6063, 584, 502, 4951, 4322, 213, 318, 405, 296, 166, 231, 54, 240, 11
```

Only one tab (11 unique videos) falls below `min_tab_pool=20`. Measured
fallback rate across the full 3-seed × ~10-epoch sweep for both `same_tab`
and `same_tab_pop_weighted`: **0.0%** (exactly zero fallback events counted)
— that smallest tab's rows never happened to be the sampled positive row's
tab in this run. Fallback logic is implemented and verified working (spot
tested by temporarily setting `min_tab_pool=10000`, which forces every draw
to fall back and runs without error), it just never fired at the default
threshold.

## Sweep results (3 seeds: 0, 1, 2; iter9's exact hyperparams — k=16, lr=0.001,
bs=8192, epochs≤40, patience=4)

| negative_mode          | valid primary (mean, std) | test primary (mean, std) | epochs to convergence |
|-------------------------|---------------------------|---------------------------|------------------------|
| `uniform` (baseline)    | **0.6102**, 0.00013        | **0.6057**, 0.00010        | 10–12 |
| `same_tab`               | 0.5600, 0.00033             | 0.5518, 0.00069             | 9–11 |
| `pop_weighted`            | 0.5809, 0.00148             | 0.5731, 0.00157             | 5 (all 3 seeds) |
| `same_tab_pop_weighted`  | 0.5627, 0.00027             | 0.5562, 0.00069             | 6 (all 3 seeds) |

**Parity check**: `uniform` mode's 3-seed valid mean (0.6102) matches iter9's
published 5-seed valid mean (0.61013, std 0.00027) essentially exactly —
this run's harness is a faithful re-derivation of iter9. Per-seed values for
`uniform` (0.6103, 0.6104, 0.6101) also individually match iter9's own
per-seed log (0.6103, 0.6104, 0.6101, 0.6096, 0.6103) for seeds 0–2.

All three hard-negative variants are **dramatically** worse than the
baseline — a gap of 0.03–0.05 in absolute primary score, roughly 100–400×
each variant's own seed-to-seed std, and far larger than any gap seen
anywhere else in this entire 17-iteration run (the next-largest effect,
iter9's feature win, was +0.0090). This is not a borderline call requiring a
5-seed extension; the sweep is already fully conclusive with 3 seeds.
`pop_weighted` also converges suspiciously fast (early-stops at epoch 5
every seed, vs. 10-12 for the baseline) — consistent with the model
quickly overfitting to a biased/noisy signal rather than learning a stable
ranking function.

## Discussion — why harder negatives hurt this badly

The magnitude here (a 5-8% *relative* collapse in primary score) is too
large to be explained by "harder negatives create noisier gradients, making
optimization slower" (iter4 already ruled out plain gradient-variance
effects for BPR, and here the loss curves show fast, confident convergence
to a *worse* plateau, not slow/noisy convergence to the same one). The more
likely explanation, given what "negative" turned out to mean in this
codebase (see top of doc):

The `uniform` baseline's negatives are **verified negative labels** — videos
the specific user was actually shown and did *not* long-view. Every one of
these is a trustworthy, observed data point. The `same_tab` and
`pop_weighted` schemes instead assume that *any* video sharing the
positive's tab, or any popular video, is a valid negative for that user —
but these candidate videos were **never actually shown to that user**, so we
have zero ground truth about whether the user would have long-viewed them.
Same-tab and popular videos are, if anything, exactly the videos a user is
statistically *more* likely to have enjoyed (that's why they're popular /
why the user is active in that tab) — so this scheme is systematically
mislabeling plausible true positives as negatives, injecting heavy,
directionally-biased label noise into every single training step. This is
the opposite of typical "hard negative mining" in retrieval settings (where
negatives are drawn from a candidate pool with verified relevance judgments
or filtered by an auxiliary model) — here there is no such verification
available, so "harder-looking" negatives are actually far *more* likely to
be silently mislabeled than a uniform-videos-the-user-actually-rejected
negative is. `same_tab_pop_weighted`, which combines both biases, is not
even the worst performer — `pop_weighted` (unrestricted by tab) is, which
fits this story: popularity alone is the single strongest confound with
"the user would probably have liked this," so weighting toward popular
candidate videos does the most damage.

## Verdict: REJECT

None of `same_tab`, `pop_weighted`, or `same_tab_pop_weighted` come close to
iter9's baseline — all three are massively worse, not within noise, and
in the wrong direction relative to the iteration's hypothesis. No 5-seed
confirmation run is warranted (the 3-seed sweep is already unambiguous by a
huge margin). **iter9 remains the current best**; do not promote.

Residual finding for future rounds: this suggests any future "hard
negative" idea for this dataset needs an independent way to verify a
candidate video is a true negative for the user (e.g. restricting candidates
to videos the user was actually exposed to elsewhere in the log but never
positively engaged with across ALL their appearances, rather than videos
merely resembling ones they liked) — sampling distribution changes over an
*unverified* candidate pool are actively harmful here, unlike the
user-sampling-distribution changes (iter3) which operated over already
fully-verified per-user positive/negative rows.

## Code

- `experiments/iter17_hard_negatives/data_ext.py` — re-exports iter9's
  feature computation unmodified.
- `experiments/iter17_hard_negatives/train.py` — `negative_mode` param
  (`uniform`/`same_tab`/`pop_weighted`/`same_tab_pop_weighted`), standalone
  CLI entry point.
- `experiments/iter17_hard_negatives/driver.py` — sweep driver, loads/
  encodes data once and shares it across all mode×seed runs, writes
  `results.json` incrementally.
- `experiments/iter17_hard_negatives/results.json` — raw 12-run sweep
  output (4 modes × 3 seeds).
