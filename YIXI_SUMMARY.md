# Yixi experiment summary

## 6a — XGBoost-native ranker and three-model blend: PROMOTE BLEND

The native XGBoost objective/depth sweep selected `rank:ndcg`, depth 1 at
0.65863872 valid / 0.64512122 test, which is weaker than iter44 LightGBM and
therefore rejected as a standalone replacement. Its errors were complementary,
however: the validation-only blend sweep selected 16% FM / 8% LightGBM / 76%
XGBoost at **0.67006499 valid / 0.65645623 test**, confirmed identically across
five XGBoost seeds. **PROMOTE the three-model blend only; standalone XGBoost is
REJECT.**

## 6b — native-LightGBM feature depth: REJECT

Additional user/tab decay horizons, causal author/video popularity, and three
cross-features were tested independently on fixed iter44 LightGBM. The best
seed-0 feature, `decay_rate_2.5 * log1p(decay_act_2.5)`, scored 0.66478848 valid
/ 0.64850 test, but its five-seed mean valid gain was only +0.00091 and three
of five paired seed deltas were negative. **REJECT:** no 6b feature met the
required +0.001 validation gain across five seeds.

## 6c — lightweight attention pool for native XGBoost: REJECT

Frozen k=8 train-only item embeddings were used for scaled-dot-product
attention over each user's strictly causal last 20 or 40 interactions, adding
one raw pooled-label scalar to YIXI4's fixed h5 XGBoost component. The exact h5
reference reproduced at 0.66649055 valid / 0.65197098 test. K=20 fell to
0.61597961 valid; the selected K=40 candidate scored **0.65841424 valid /
0.64911228 test**, deltas of -0.00807631 / -0.00285870. Neither window cleared
the +0.0003 revisit gate, so no seed confirmation, union, or attention blend
was permitted. **REJECT:** the feature takes substantial split gain but is
highly redundant with h5 decay, `lastk_rate`, and the long-window mean.

## Additional 6d transfer result — component finding only

Retesting the unchanged 6b families on fixed XGBoost showed that replacing the
2.5-day user-decay pair with the 5-day pair improves the standalone component
from 0.65863872 to **0.66649055 valid** and from 0.64512122 to **0.65197098
test**, with identical paired validation deltas across five seeds. This is a
confirmed XGBoost feature improvement, but it has not been shown to beat the
project-best 6a three-model blend at 0.67006499 valid; it should not be
described as a new overall project promotion without a validation-only reblend.
