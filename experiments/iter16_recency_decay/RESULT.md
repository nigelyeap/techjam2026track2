# iter16 — recency-decayed (exponential half-life) causal history features

## Idea
iter9's `activity`/`tab_pos`/`rate` are FLAT cumulative counters — a positive
row from 3 weeks ago counts exactly as much as one from yesterday. This
iteration asks whether TIME-DECAYED versions of `rate` (and `activity`)
carry more signal, since recent behavior may be more predictive of a user's
*current* interest than old behavior, even within KuaiRand-Pure's short
(~3-week) log window.

Decay formulation (half-life parameterized), computed via the exact same
strict-causal (`<`, never `<=`) two-phase date-grouped traversal iter9/iter6
validated:
- `decayed_pos(u,d)   = Σ 0.5 ** ((d - date(r)) / halflife)` over prior positive rows `r`
- `decayed_total(u,d) = Σ 0.5 ** ((d - date(r)) / halflife)` over ALL prior rows
- `decayed_rate = (decayed_pos + 1) / (decayed_total + 2)` — same Laplace shape as iter9's `rate`
- `decayed_activity` (`decay_act`) = `decayed_total` itself

Implementation note: rather than recomputing pairwise date diffs, the decay
is computed with an exact (not approximate) lazy-decay running-state trick —
per user we keep `(last_update_date, pos_value, total_value)` already decayed
to `last_update_date`; reading at a later date `d` is a single multiply by
`0.5 ** ((d - last_update_date)/halflife)`, which composes exactly because
exponential decay is multiplicative over elapsed time regardless of chunking.
All 5 half-lives are computed together in one traversal pass for efficiency.
See `data_ext.py::compute_decay_features`.

## Causality verification
`data_ext.py`'s `__main__` block:
- Brute-force spot-checks 25 random rows × 3 half-lives (1d/7d/30d) against a
  manual O(n) recount summing `0.5**(gap/halflife)` over every actual prior
  row — max abs error `4.26e-14` (floating-point noise only). **PASSED.**
- Confirms zero-activity rows (no prior history at all) get exactly
  `decayed_pos = decayed_total = 0.0` for every half-life. **PASSED.**
- Same-date-pair edge case: a user with 3 same-date positive rows — all 3
  rows show *identical* `decayed_pos`/`decayed_total` (each is blind to the
  other same-date rows, per the strict `<` semantics). **PASSED.**

No leakage detected. Full output:
```
25 random rows x 3 halflives: all decayed_pos/decayed_total match brute force
(max abs err 4.26e-14). No leakage detected.
zero-activity rows (5 checked): decayed_pos/total correctly 0.0.
halflife=7d, user/date pair with 3 same-date positives:
  user=1 date=20220412 decayed_pos=0.8203 decayed_total=2.6318  (x3, identical)
  -> identical decayed_pos/decayed_total across the same-date pair, as expected.
```

**Sanity check on the harness itself**: `flat_rate` (iter9's `rate`, alone,
recomputed through this iteration's own code path) gave valid `0.60905` /
test `0.60348` — an *exact* match to iter11's independently-run `rate`-alone
number. Confirms the encoding/training pipeline is a faithful match to
iter9/iter11, so the gains below are attributable to the decay mechanism, not
an implementation drift.

## Phase 1 — half-life sweep, `decay_rate` alone (3 seeds: 0,1,2)
| feature | valid primary (mean, std) | test primary (mean, std) |
|---|---|---|
| flat `rate` (iter9, baseline) | 0.60905 (0.00033) | 0.60348 (0.00060) |
| `decay_rate_1` (halflife=1d) | 0.61561 (0.00037) | 0.60887 (0.00024) |
| **`decay_rate_3` (halflife=3d)** | **0.61802 (0.00026)** | **0.61621 (0.00052)** |
| `decay_rate_7` (halflife=7d) | 0.61373 (0.00009) | 0.61202 (0.00036) |
| `decay_rate_14` (halflife=14d) | 0.61103 (0.00034) | 0.60756 (0.00014) |
| `decay_rate_30` (halflife=30d, ~flat sanity check) | 0.60980 (0.00042) | 0.60557 (0.00010) |

Key findings:
- **Every single decayed-rate variant, on its own, already beats flat `rate`**,
  and `decay_rate_3`/`decay_rate_7` alone (a *single* field) beat iter9's full
  3-feature combo (`activity+tab+rate` @ 0.61013 valid / 0.60560 test).
- The halflife axis has a clear, non-monotonic peak around **3 days** — not
  the shortest (1d, too noisy/local) nor the longest (30d, ~reproduces flat
  `rate` as expected — the sanity-check assumption held: 0.60980 valid is
  within noise of flat `rate`'s 0.60905).
- This directly answers the iteration's question: recency *does* carry real
  additional signal beyond flat cumulative counting, even within a 3-week
  window — user interest apparently drifts fast enough that a ~3-day
  half-life captures it much better than a ~3-week (effectively flat) one.

## Phase 2 — feature combos around the winning half-lives (3d, 7d) (3 seeds: 0,1,2)
| feature set | valid primary (mean, std) | test primary (mean, std) |
|---|---|---|
| `decay_rate_3` alone | 0.61802 (0.00026) | 0.61621 (0.00052) |
| `decay_rate_3` + `tab` | 0.61873 (0.00020) | 0.61726 (0.00038) |
| **`decay_rate_3` + `decay_act_3` + `tab`** | **0.62028 (0.00061)** | **0.61727 (0.00190)** |
| `decay_rate_3` + flat `rate` | 0.61092 (0.00073) | 0.60768 (0.00046) |
| `decay_rate_7` alone | 0.61373 (0.00009) | 0.61202 (0.00036) |
| `decay_rate_7` + `tab` | 0.61513 (0.00019) | 0.61352 (0.00039) |
| `decay_rate_7` + `decay_act_7` + `tab` | 0.61626 (0.00036) | 0.61500 (0.00025) |
| `decay_rate_7` + flat `rate` | 0.61122 (0.00041) | 0.60740 (0.00079) |

Findings:
- 3-day half-life dominates 7-day at every combo depth — confirms 3d as the
  right operating point, not just a lucky single-feature result.
- Adding `tab` (flat, iter9-style) on top of `decay_rate` gives a small,
  consistent lift; adding `decay_act` (decayed activity, same half-life) on
  top of that gives a further, consistent lift — both real, both additive.
- **Combining decayed rate WITH flat rate (`decay_rate_3+flat_rate`) clearly
  HURTS** relative to `decay_rate_3` alone (0.61092 vs 0.61802 valid) — this
  contradicts the "complementary short-term vs long-term signal" hypothesis
  from the task brief. The two fields are highly correlated (same underlying
  positive-history counts, just decayed differently), and giving the FM two
  separate quantile-bucketed categorical encodings of near-duplicate
  information seems to dilute rather than help, consistent with iter12's
  finding that redundant-but-different-shaped signal doesn't stack cleanly
  in this FM.

## 5-seed confirmation, top 2 combos (seeds 0–4)
### `decay_rate_3` + `decay_act_3` + `tab` (overall winner on valid)
| seed | valid primary | test primary |
|---|---|---|
| 0 | 0.61943 | 0.61761 |
| 1 | 0.62064 | 0.61479 |
| 2 | 0.62078 | 0.61941 |
| 3 | 0.62013 | 0.61482 |
| 4 | 0.62050 | 0.61826 |
| **mean** | **0.62030** | **0.61698** |
| **std** | 0.00048 | 0.00187 |

### `decay_rate_3` + `tab` (simpler 2-field runner-up)
| seed | valid primary | test primary |
|---|---|---|
| 0 | 0.61854 | 0.61727 |
| 1 | 0.61864 | 0.61679 |
| 2 | 0.61901 | 0.61772 |
| 3 | 0.61846 | 0.61518 |
| 4 | 0.61926 | 0.61590 |
| **mean** | **0.61878** | **0.61657** |
| **std** | 0.00030 | 0.00092 |

The 3-feature combo beats the 2-feature one on valid **consistently in all 5
seeds** (gap +0.00152, ~3x the runner-up's own std) — a systematic, not
noise-driven, win. Per protocol (selection on valid only), `decay_rate_3 +
decay_act_3 + tab` is the winning config.

## Verdict vs iter9 (current best: valid 0.61013 / test 0.60560, 5-seed)
**Δ = +0.01017 valid, +0.01138 test.** This is ~20x iter9's own valid std
(0.00027) and, even accounting for this config's own noisier test std
(0.00187 vs iter9's 0.00020), the *worst* of the 5 test seeds (0.61479) is
still +0.0092 above iter9's mean — the gain is not fragile to any single
seed.

## Status: **PROMOTE — new best, replacing iter9**

Config: FM + activity-weighted BPR (identical loss/sampling to iter3/iter9),
fed 3 causal features via the strict-`<` date-grouped traversal:
- `decay_rate_3`: Laplace-smoothed exponentially-decayed positive rate, half-life 3 days
- `decay_act_3`: exponentially-decayed total-row count (activity), half-life 3 days
- `tab`: iter9's flat per-tab positive-row count (unchanged)

Final numbers (5-seed): **valid primary mean 0.62030 (std 0.00048)**, **test
primary mean 0.61698 (std 0.00187)**. Beats iter9 by a wide, consistent
margin on both splits — the largest single gain since iter9 itself replaced
iter3. Confirms the hypothesis: flat cumulative history-counting was leaving
real signal on the table; recency matters even within a ~3-week log window,
and a short (~3-day) half-life captures it best. The straightforward
"combine flat and decayed" idea does NOT work (they're redundant, not
complementary) — the win comes purely from replacing the flat features with
better (decayed) ones, plus a small additional lift from also decaying
`activity`.

## Code
`experiments/iter16_recency_decay/{data_ext.py,train.py,driver.py}`,
raw sweep results in `experiments/iter16_recency_decay/results.json`.
