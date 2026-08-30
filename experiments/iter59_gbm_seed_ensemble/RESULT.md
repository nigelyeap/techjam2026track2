# iter59 — GBM 5-seed prediction-averaged ensemble, blended with FM

## Motivation

Every prior blend (iter44, iter51, iter55) used a single GBM seed (seed=0)
while the FM side has always been a genuine 5-seed prediction-averaged
ensemble. iter55's own 5-seed confirmation showed real but small GBM
seed-to-seed valid std (0.00021). This checks whether averaging the GBM's
raw prediction scores across 5 seeds — the same ensembling treatment
already given to the FM side — reduces that noise and beats using a single
arbitrary GBM seed.

## Method

Train the GBM (`linear_tree=True, learning_rate=0.10`, all else at iter55's
config) at seeds `{0,1,2,3,4}`, min-max normalize each seed's raw
prediction scores, average across seeds, then alpha-sweep the blend
against the unchanged iter38 FM 5-seed ensemble exactly as iter55's
`blend.py` did.

## Result

| | valid | test |
|---|---|---|
| GBM single seed=0 (iter55) | 0.67052 | 0.65277 |
| GBM 5-seed ensemble (this) | 0.67016 | 0.65228 |
| FM 5-seed ensemble (unchanged) | 0.63988 | 0.64187 |
| iter55 blend (alpha=0.10) | **0.67451** | **0.65832** |
| iter59 blend (alpha=0.12) | 0.67424 | 0.65818 |

The GBM 5-seed ensemble standalone is *worse* than the single seed=0 run
(0.67016 vs 0.67052) — of the 5 seeds, only seed=0 sits at the top of the
tight variance band (per iter55's 5-seed table: seeds 0/4 at 0.67008-0.67052,
seeds 2/3 at 0.66993), so averaging pulls the mean down slightly rather
than denoising upward. The resulting blend is correspondingly slightly
worse than iter55's current blend (-0.00027 valid, -0.00014 test).
**Verdict: REJECT** — prediction-averaging the GBM across seeds does not
help here; unlike the FM (whose seed-to-seed variance is apparently large
enough that averaging denoises a real gain), the GBM's seed variance is
tight enough that the single already-selected seed=0 is, at worst, a wash
and here is mildly ahead of the ensemble mean. Not worth pursuing further;
iter55's single-seed GBM blend remains the best-known candidate.
