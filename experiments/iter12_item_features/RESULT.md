# iter12 — ITEM-side causal history features (video_pop, author_rate)

## Idea
iter9's causal history features are all USER-side (activity, tab_pos, rate — "how
active/experienced is this user, what's their historical positive rate"). This
iteration asks the complementary question: does knowing how popular/well-received
the VIDEO or AUTHOR has been so far (causally) add anything on top of iter9's
user-side set?

Two new candidate features, computed via the exact same strict-causal (`<`, never
`<=`) two-phase date-grouped traversal pattern iter6/iter9 validated — see
`data_ext.py::compute_causal_features`, which extends iter9's traversal to also
track per-video and per-author counters in the same single pass:

- **`video_pop`**: count of this `video_id`'s prior positive (`long_view`==1)
  rows, anywhere (any user), before this row's date. Pure item-popularity signal.
- **`author_rate`**: this `author_id`'s Laplace-smoothed prior positive rate,
  aggregated over ALL of that author's prior rows from ANY user —
  `(prior_pos+1)/(prior_total+2)`, same smoothing formula as iter9's user `rate`.
  Explicitly NOT iter6's per-(user, author) affinity (0.70% coverage, rejected)
  — this is the author's own aggregate rate across the whole user population.

## Coverage
| feature | nonzero coverage | rows |
|---|---|---|
| `video_pop` | **78.70%** | 1,130,624 / 1,436,609 |
| `author_rate` (author has any prior history) | **84.70%** | 1,216,867 / 1,436,609 |
| `author_rate` (author has any prior positive) | 80.41% | 1,155,110 / 1,436,609 |

Both far above the ~5% sparsity-flag threshold, and much denser than iter6's
per-(user,author) affinity (0.70%) — since a video/author accumulates history
from many different users, not just one. Sparsity is NOT a concern here.

## Causality verification
`data_ext.py`'s `__main__` block brute-force spot-checks `video_pop` and
`author_rate`'s `author_prior_pos`/`author_prior_total` against manual O(n)
recounts over the full combined train+valid+test timeline, including same-date-
pair edge cases (a video and an author each found with >=2 same-date positive
rows; confirmed same-date rows show IDENTICAL feature values, i.e. never count
each other, matching a strictly-earlier-date-only manual recount). All asserts
passed:
```
video_pop: all spot-checks passed.
author_rate: all spot-checks passed.
same-date-pair edge case passed: same-date positives do not count each other.
same-date-pair edge case passed for author_rate.
All causal spot-checks passed. No same-date or future leakage detected.
```
No leakage detected. Full output preserved by re-running `python3 data_ext.py`.

As an additional sanity check, the `user_only_iter9` combo below (features
`activity,tab,rate`, i.e. a byte-for-byte re-derivation of iter9's winning set
through this directory's own `data_ext.py`/`train.py`) reproduces iter9's
published per-seed numbers almost exactly (seed 0: valid 0.61028 here vs
0.61028 in iter9's RESULT.md; seed 1: 0.61038 vs 0.61038; seed 2: 0.61006 vs
0.61006) — confirming this iteration's training loop is a faithful copy of
iter9's and isolates the feature-set variable cleanly.

## Sweep (3 seeds: 0,1,2)
| combo | valid mean | valid std | test mean | test std | Δ valid vs user-only |
|---|---|---|---|---|---|
| user-only (iter9 ref: activity,tab,rate) | 0.61024 | 0.00013 | 0.60572 | 0.00010 | — |
| user + video_pop | 0.61026 | 0.00047 | 0.60573 | 0.00035 | +0.00002 |
| user + author_rate | 0.61017 | 0.00033 | 0.60564 | 0.00055 | -0.00007 |
| user + video_pop + author_rate | 0.61037 | 0.00033 | 0.60558 | 0.00056 | +0.00013 |
| item-only (video_pop, author_rate; no user-side) | 0.60274 | 0.00055 | 0.59688 | 0.00026 | -0.00750 |

Every combo that adds item-side features to iter9's user-side set lands within
±0.0002 of the user-only reference — smaller than iter9's own 5-seed std
(0.00027) and far below the ~0.001-0.002 margin needed to be considered a real
signal rather than noise. None of the three combined combos comes close to
warranting a 5-seed confirmation run.

The item-only baseline (no user-side features at all) reaches valid 0.60274 /
test 0.59688 — meaningfully above iter1's plain pointwise baseline (test
0.5946) and roughly matching iter3's plain activity-weighted BPR with no causal
features at all (test 0.59658). So item-side features do carry real, non-trivial
signal on their own — they're just almost entirely redundant with what the
user-side `rate`/`tab_pos`/`activity` features (plus the FM's own learned
video_id/author_id embeddings) already capture, which is why stacking them on
top of the user-side set adds ~nothing.

## Verdict: **REJECTED** — item-side features do not add value over iter9

No combo beat iter9's valid mean (0.61013 5-seed / 0.61024 3-seed
re-derivation here) by anything close to the ~0.001-0.002 threshold; all
deltas are noise-level. Per the task's stated criteria, no combo qualifies for
extension to the remaining seeds (3,4) or a test-set check — iter9 remains the
current best.

**Why item-side didn't add value** (not a coverage/leakage problem — both are
clean, as verified above): the FM model already includes `video_id` and
`author_id` as raw categorical fields, so it can already learn a per-video/
per-author "average propensity" bias term directly from the embedding table
whenever there's enough training data for that id — which is essentially what
`video_pop`/`author_rate` encode, just pre-computed and bucketed rather than
learned. The causal user-side features (especially `rate`) added a genuinely
new axis of information the base 5-field model couldn't otherwise represent
(a *user's* personal historical tendency, which has no dedicated field). Item
popularity, by contrast, is largely already reachable through the existing
video_id/author_id embeddings, so making it explicit is mostly redundant
re-encoding of information the model could already extract, hence the flat
result.

## Code
`experiments/iter12_item_features/{data_ext.py,train.py,sweep.py}`,
raw sweep results in `experiments/iter12_item_features/results.json`.
