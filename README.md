# KuaiRand-Pure Starter Kit

## Dependencies

Python 3.9+ and numpy. **Nothing else.** No torch, pandas, or sklearn needed.

## Data

Download from https://kuairand.com (direct Zenodo link, no registration needed):

```bash
# Run from the Starter Kit directory; extracts to ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Run

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`; pass it explicitly if your data lives elsewhere.

`--model` accepts `fm` (official baseline) / `pop` (trivial baseline) / `random` (lower bound, for
sanity-checking the eval harness).
FM takes about 40 seconds end-to-end (CPU, single core).

## Task definition (fixed — do not change)

| | |
|---|---|
| Task | **Within-user ranking** — each user's own exposures in the eval set are ranked against each other; this is not full-catalog retrieval |
| Relevance label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary = mean of the two** |
| Data split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Zero-positive users | nDCG is recorded as 0.0 and counted in the average; GAUC only counts users with `0 < positives < exposures`, weighted by positive count |
| nDCG gain | `2^rel − 1` (equivalent to identity under binary labels) |

Implementation in `evaluate.py`; every convention is documented in the file header comment.

## Baseline ladder

Scores on the test set. **The line to beat is FM.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound, sanity check) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ The real range of the metric: nDCG@5's ceiling is 0.729, not 1.0

Across the 23,875 users in the test set:

| | share | effect on the metric |
|---|---|---|
| All-negative users (none of their exposures are long_view) | **27.1%** | nDCG is always **0**, no model can fix this; excluded from GAUC |
| All-positive users | **9.2%** | nDCG is always **1**; excluded from GAUC |
| Users with actual discrimination | **63.7%** | the real sample GAUC is computed over |

So even using the true labels as the prediction score (oracle, perfect ranking) only reaches:

| | random | FM baseline | **oracle ceiling** | headroom FM has already captured |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Measure progress against the oracle, not 1.0.** Seeing 0.5946 and assuming "far from a perfect
1.0 score" is a misread — the baseline has already captured roughly three-tenths of the available
range; the remaining headroom is 0.27, not 0.41.

FM's std across 5 random seeds is **0.0008**. The convergence rule is set accordingly at
**ε = 0.002 (≈2.5σ), N = 3**: stop when 3 consecutive iterations each improve the validation
primary score by no more than 0.002.

> Sanity check: if running your eval code with `--model random` doesn't give primary ≈ 0.475
> (±0.001), something is wrong with the harness — fix that first.

## Submission format

CSV with a header row, one row per row of the eval set:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| field | description |
|---|---|
| `row_id` | 0-indexed, contiguous, matching the row order of `data.load()[split]` (deterministic: `log_standard_4_08_to_4_21_pure.csv` is read first, then `log_standard_4_22_to_5_08_pure.csv`, filtered by date while preserving original file order) |
| `user_id` / `video_id` | redundant fields, used only to validate alignment |
| `score` | your model's score for that row, any real number, only relative order matters; NaN/Inf not allowed |

> **Why `row_id` is required:** `(user_id, video_id)` is **not unique** in the eval set —
> 3.06% of test-set pairs are duplicated, up to 12 times. So it can't serve as a primary key.

Generate and validate:

```bash
python3 submit.py --make  --split test  submission.csv    # generate a sample submission using the official FM baseline
python3 submit.py --check --split test  submission.csv    # validate format and alignment
python3 submit.py --score --split valid submission.csv    # validate and score (usable locally on valid)
```

`--check` rejects: wrong header, wrong row count, gaps in `row_id`, `user_id`/`video_id` misaligned
with the eval set, non-numeric or NaN/Inf `score`. **Run `--check` yourself before submitting.**

## Where to start improving

The ordering below is **empirically tested**, not a guess. Dead ends the organizers already tried
are marked explicitly — don't repeat them.

### Already tested: no gain from these, don't spend iterations here

| Tried | Result |
|---|---|
| **Adding static features** — wiring in all 13 CWM feature domains (+`music_id`/`video_type`/`upload_type` + 6 coarse user-side buckets) | primary **0.5940** vs **0.5950** with 5 domains — no difference beyond noise, if anything slightly worse |
| **Adding model capacity** — embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887, barely moves |

Why: the `user_id × video_id` cross already captures most of the learnable signal. Coarse buckets
like `follow_user_num_range` are redundant once `user_id` is present; and 1.14M rows of data can't
support much more capacity either. **The bottleneck isn't features or capacity.**

⚠️ Also note: **pure user-side features' first-order term always contributes 0 to the score.**
Because ranking happens within each user, any term that's constant within a user doesn't change the
in-group order (empirically: `item_pop × user_bias` and plain `item_pop` scored identically, bit
for bit). User-side features can only matter through **cross terms with item-side features**.

### Unexplored: this is where the headroom should be

Ranked by our best guess at likely payoff (**the organizers have not tested these — they're left
for you**):

1. **Change the loss function.** Currently pointwise logloss, but the metrics (GAUC / nDCG) are
   **ranking metrics**. Switching to pairwise (BPR) or listwise (softmax over a user's own
   exposures) — aligning the training objective with the eval metric — is what we consider the
   most likely lever to work.
2. **User history sequences.** The current features **make no use of behavior sequences at all**.
   Each KuaiRand user has hundreds to thousands of interactions in train — DIN/SIM-style interest
   modeling is a completely untouched direction.
3. **Multi-task learning.** The logs also carry `is_click`, `is_like`, `is_follow`, `is_comment`,
   `is_forward`, `play_time_ms` — these could serve as auxiliary tasks alongside the main
   `long_view` task.
4. **Modeling watch duration.** This is exactly what [CWM](https://github.com/hyz20/CWM)
   contributes: it models watch time as **censored regression** (true watch time is truncated
   when a video finishes playing, so it uses a one-sided loss rather than squared error). A
   research-depth direction.
5. **Changing the model.** DeepFM / DCN / xDeepFM. Given that capacity empirically isn't the
   bottleneck, **prioritize this after 1-4**.
6. **Time features and distribution drift.** `hourmin`, `date`, and drift between train and test.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is a randomly-exposed
   log (1.18M rows) that can serve as an additional unbiased validation set, to check whether a
   model is just overfitting to biased traffic.

## Using your own model (including CWM)

`evaluate.py` is fully decoupled from any model — it only needs three equal-length arrays:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores can come from any model
```

- `user_ids`: the user_id of each row in the eval set
- `labels`: that row's `long_view` (0/1)
- `scores`: your model's score for that row (any real number, only relative order matters)

So you don't have to use `baseline.py` at all — swap in PyTorch, LightGBM, or CWM's xDeepFM,
as long as you hand the final `scores` to `evaluate()`. **`evaluate.py` is the sole source of
truth for scoring.**

> A note on using CWM: it depends on `torch==1.6.0` (a 2020-era release, likely won't install
> cleanly on newer GPUs), and its loss optimizes counterfactual watch time, with its own
> reconstructed `long_view2` label rather than this task's `long_view`. It's research code from a
> watch-time debiasing paper — useful as an **advanced reference**, not recommended as a starting
> point.

## Files

| | |
|---|---|
| `evaluate.py` | Metric implementation + all scoring conventions. **Do not modify.** |
| `data.py` | Data loading, the official split, feature encoding. Add features here. |
| `baseline.py` | The three baselines. FM is the one to beat. |
| `baseline_scores.json` | Officially published scores + seed variance + convergence parameters. |
| `submit.py` | Generate / validate submission files. |
| `ablation_features.py` | Feature ablation experiments; reproduces the "no gain from adding features" numbers. |
