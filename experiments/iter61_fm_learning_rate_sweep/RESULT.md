# iter61 — FM learning_rate resweep

## Motivation

The FM's `lr=0.001` (Adam) has been unchanged since iter38 and was never
resystematically resweept — the FM-side analogue of the GBM `learning_rate`
finding (iter55) that produced the current best blend.

## Method

Coarse sweep `{0.0002..0.005}`, fine sweep `{0.0003..0.0008}` around the
coarse winner, 5-seed confirmation at the fine-sweep winner, then a full
blend re-run against the unchanged iter55 GBM (single seed=0).

## Standalone results

Coarse sweep (single seed=0):

| lr | valid | test |
|---|---|---|
| 0.0002 | 0.63086 | 0.63384 |
| 0.0005 | 0.63951 | 0.64194 |
| 0.0007 | 0.63929 | 0.64078 |
| **0.0010 (current)** | 0.63894 | 0.63989 |
| 0.0015 | 0.63774 | 0.63995 |
| 0.0020 | 0.63769 | 0.63844 |
| 0.0030 | 0.63457 | 0.63664 |
| 0.0050 | 0.63068 | 0.63030 |

Fine sweep around 0.0005 — a genuine plateau, not a spike, +0.0005-0.0006
above baseline across the whole 0.0003-0.0008 neighborhood:

| lr | valid | test |
|---|---|---|
| 0.0003 | 0.63952 | 0.64130 |
| 0.0004 | 0.63927 | 0.64066 |
| **0.0005** | 0.63951 | 0.64194 |
| 0.0006 | 0.63957 | 0.64073 |
| 0.0007 | 0.63929 | 0.64078 |
| 0.0008 | 0.63809 | 0.63938 |

5-seed confirmation, `lr=0.0005` vs `lr=0.001`:

| seed | valid (0.0005) | valid (0.001) | delta |
|---|---|---|---|
| 0 | 0.63951 | 0.63894 | +0.00058 |
| 1 | 0.63884 | 0.63868 | +0.00016 |
| 2 | 0.63908 | 0.63685 | +0.00223 |
| 3 | 0.63853 | 0.63747 | +0.00105 |
| 4 | 0.63909 | 0.63768 | +0.00140 |

**5-seed valid: mean=0.63901 (std=0.00032) vs mean=0.63792 (std=0.00077),
mean delta=+0.00108, wins=5/5.** This is a real, standalone gain — it
clears the 0.001 "unambiguously real" bar and also *reduces* seed-to-seed
variance (0.00032 vs 0.00077), unlike the GBM's learning_rate finding
which didn't change GBM variance.

## Blend result

| | valid | test |
|---|---|---|
| iter55 blend (FM lr=0.001, alpha=0.10) | **0.67451** | **0.65832** |
| iter61 blend (FM lr=0.0005, alpha=0.12) | 0.67475 | 0.65804 |

The blend-level valid gain (+0.00024) is well below the 0.0003 look
threshold that gated every other promotion this session, and — more
importantly — **test moves the wrong way** (-0.00028). **Verdict: REJECT
for promotion, but flag the standalone finding.** The FM's real, confirmed
standalone improvement (+0.00108 valid, 5/5 seeds) does not propagate
meaningfully to the blend: the GBM already captures much of what a
better-fit FM adds here, so the blend's marginal information gain from a
sharper FM component is small, while the alpha search compensating with
slightly more FM weight (0.12 vs 0.10) is enough to tip test the other
way. Per the model-selection protocol (valid-only), a config that clears
the standalone bar but not the blend-level look-threshold — and regresses
test — is not itself a good promotion candidate; the current iter55 blend
(GBM lr=0.10, FM lr=0.001, alpha=0.10) remains the best-known submission
candidate.
