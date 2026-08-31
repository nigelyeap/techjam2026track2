# iter79 — FM-side `decay_tab_rate_3` (parity test with iter63's GBM win)

## Provenance

Not another GBM-feature variant. After 10 consecutive closed/rejected
GBM-side iterations (iter68-78), tested whether the FM half of the current
ensemble benefits from the same insight that produced iter63's real GBM
gain: swapping a decayed (user, tab) positive-row COUNT for a
Laplace-smoothed decayed RATE. The FM's live promoted feature set
(`iter27_triple_fusion/driver.py`'s `ITER24_FEATS`, used by iter38's 5-seed
ensemble — the FM half of the current final submission) still uses
`decay_tab_3` (count), unchanged since Round 7/iter24 — it has never
received this treatment, unlike the GBM side. Genuinely new-in-kind: the
first change to the FM's own feature encoding since Round 18.

## Implementation

`experiments/iter79_fm_decay_tab_rate/data_ext.py` = `iter27_triple_fusion/
data_ext.py` (has `encode_ext`/`compute_final_decayed_pos`, which iter63's
file lacks) + iter63's already-verified `compute_decay_tab_features`/
`load_ext` diff (adds `decayed_tab_total` tracking alongside
`decayed_tab_pos`, same lazy-decay mechanism, causality already established
in iter63/iter77) + a new `decay_tab_rate` kind added to `encode_ext`'s
three dispatch points (`parse_feat`, edge-building, `extra_val`), computing
`(pos + α)/(total + 2α)` and quantile-bucketing it exactly like every other
bucketed feature in this pipeline. `train.py` = byte-for-byte copy of
`iter27_triple_fusion/train.py` (local import now resolves to this
directory's own `data_ext.py`). No re-derivation of the count/total
causality proof was needed (already established twice); only a
harness-fidelity check was required before trusting any new number.

Harness-fidelity check reproduced iter27's own published seed-0 result for
the exact promoted config (`sampling_mode=decay, sampling_alpha=0.75,
decay_halflife=3, alpha=0.5, n_buckets=20`, features `decay_rate_2.5,
decay_act_2.5, decay_tab_3, last1, lastk_rate, gap`) from
`iter27_triple_fusion/results.json`: expect valid=0.63894/test=0.63989, got
valid=0.63892/test=0.63982 (both well within 1e-4) — PASS. Then swapped
`decay_tab_3` → `decay_tab_rate_3` in the feature set, single seed=0
(exploratory, per project convention).

## Result (seed 0)

| variant | valid | test |
|---|---|---|
| baseline (`decay_tab_3`, count) | 0.63892 | 0.63982 |
| `decay_tab_rate_3` | 0.63534 | 0.63321 |
| **delta** | **-0.00358** | **-0.00660** |

Consistent-direction, clear regression on both splits at a single seed.

## Diagnosis

Opposite sign from iter63's GBM result, and the mechanism explains why:
iter63's GBM feeds the rate as a *numeric* column into `linear_tree`'s
per-leaf linear regression, where L2 regularization can extract genuine
per-user rate signal while shrinking noisy/sparse cells toward zero
contribution. The FM instead quantile-*buckets* the rate into a categorical
field and looks up a trained embedding per bucket — this discards the
numeric ordering that makes a rate useful and instead asks every bucket to
learn its own embedding from BPR gradients. Many (user, tab) cells have a
small `decayed_tab_total` (thin exposure history), so their rate is
high-variance and dominated by the α=0.5 Laplace prior; bucketing such a
noisy quantity produces buckets with inconsistent membership across
users/eras, so their embeddings get undertrained/noisy compared to the
count feature's buckets (which correlate more directly and monotonically
with raw recent engagement, regardless of exposure denominator). In short:
a numeric-regression model can safely absorb a noisy rate; a
bucketed-embedding model cannot — this is the reverse of iter77/iter70's
finding that FM-style tricks don't transfer to `linear_tree`, now shown
symmetrically the other way.

## Verdict: REJECT (real, not promotable — regression)

Magnitude (-0.00358 valid, -0.00660 test) is far outside this project's
single-seed noise floor (iter27's own 5-seed valid std was 0.00075 on this
exact config — the observed delta is ~4.8σ valid / ~8.8σ test), so this is
unambiguously a real effect, not noise, and 5-seed confirmation would only
narrow error bars on a result whose direction and promotion decision are
already unambiguous (a clear negative, mirroring iter77/iter70's
exact-zero/clean-reject precedent for skipping confirmation). iter38's FM
ensemble (with `decay_tab_3` count, unchanged) remains part of the current
best/submitted model. This closes off "porting the GBM-side count→rate
insight to the FM" as a productive direction — the two model families need
opposite feature representations for this particular signal.
