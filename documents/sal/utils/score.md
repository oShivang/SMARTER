# `sal/utils/score.py` — Confidence Scoring, Score Aggregation, and Voting

## What Is This File?

This file is the **scoring engine** of the SMARTER pipeline. It provides:
1. `calculate_confidence_score()` — computes multiple token-level confidence metrics from log-probabilities.
2. `aggregate_scores()` — reduces a list of per-step PRM scores to a single scalar.
3. `needs_correction()` — determines whether a confidence score is "bad enough" to warrant LLM intervention.
4. `score()` — a dataset-level function that computes majority-vote and weighted predictions across subsets of N completions.

---

## Input Format

### `calculate_confidence_score(answer_tokens_logprobs_list)`
| Argument | Type | Description |
|---|---|---|
| `answer_tokens_logprobs_list` | `list[dict]` | Per-token dicts from vLLM's `output.logprobs`. Each dict maps token_id → `Logprob` object with `.logprob` attribute |

### `aggregate_scores(scores, agg_strategy)`
| Argument | Type | Description |
|---|---|---|
| `scores` | `list[float]` | Per-step PRM scores for one candidate |
| `agg_strategy` | `"min" \| "prod" \| "last"` | Aggregation method |

### `needs_correction(score, threshold, strategy)`
| Argument | Type | Description |
|---|---|---|
| `score` | `float` | The computed confidence metric value for a step |
| `threshold` | `float` | The configured threshold from `config.threshold` |
| `strategy` | `str` | The `conf_strategy` name |

### `score(dataset, config)`
| Argument | Type | Description |
|---|---|---|
| `dataset` | `datasets.Dataset` | Must have `"completions"`, `"agg_scores"`, `"answer"` columns |
| `config` | `Config` | Used for `n`, `num_proc` |

---

## Output Format

### `calculate_confidence_score()` → `list[float]` of length 7
```python
[likelihood_score, likelihood_mean_score, probs_mean_score,
 entropy_score, top_2_diff_score, mean_least_3_score, probs_min_score]
```

### `aggregate_scores()` → `float`
A single scalar summarizing the entire multi-step solution's PRM quality.

### `needs_correction()` → `bool`
`True` if this step's confidence warrants LLM correction.

### `score()` → `datasets.Dataset`
An augmented dataset with new columns `pred_weighted@N`, `pred_maj@N`, `pred_naive@N` for each power-of-2 subset size up to `config.n`.

---

## What It Does and How (In Detail)

### `calculate_confidence_score()`

Takes the per-token logprob dicts produced by vLLM and computes **7 metrics** reflecting how confident the model was during this generation step:

| Index | Metric | Formula | Interpretation |
|---|---|---|---|
| 0 | `likelihood` | `exp(Σ log p_t)` | Overall joint probability of the step text |
| 1 | `likelihood_mean` | `exp(Σ log p_t / T)` | Geometric mean per-token probability |
| 2 | `probs_mean` | `mean(exp(log p_t))` | Arithmetic mean token probability |
| 3 | `entropy` | `mean(-Σ p * log p)` | Average per-token entropy (measures uncertainty) |
| 4 | `top_2_diff` | `mean(p1 - p2)` | Mean margin between top-2 token probabilities |
| 5 | `mean_least_3` | `mean(3 smallest p_t)` | Average of the 3 lowest probability tokens |
| 6 | `probs_min` | `min(exp(log p_t))` | The most uncertain single token in the step |

For entropy and mean_least_3, **higher values = more uncertain**; for all others, **lower values = less confident**.

### `needs_correction(score, threshold, strategy)`

Acts as a **direction-aware comparison**:
```python
if strategy in ["entropy", "mean_least_3"]:
    return score > threshold   # high entropy = bad
elif strategy == "top_2_diff":
    return score < threshold   # low margin = ambiguous
else:
    return score < threshold   # low probability = bad
```

### `aggregate_scores(scores, agg_strategy)`

Three aggregation strategies, each reflecting a different philosophical choice:
- **`"last"`**: Trust only the final step's PRM score (used in most experiments — PRM scores tend to be cumulative).
- **`"min"`**: Pessimistic — the whole solution is as good as its worst step.
- **`"prod"`**: Product of all step scores — penalizes solutions with any weak steps strongly.

### `score(dataset, config)`

A pipeline-level aggregation pass used for offline evaluation. For each power-of-2 subset size (1, 2, 4, ..., N):
1. `subsample_completions()` — takes the first `n` completions.
2. `extract_completion_answers()` — extracts LaTeX answers from each completion using the Qwen math parser.
3. `compute_weighted_pred()` — finds the answer cluster with the highest total PRM score.
4. `compute_maj_pred()` — finds the majority-vote answer.
5. `compute_naive_pred()` — takes the highest-scored individual completion.
6. Temporary columns are removed after each subset to keep the dataset lean.

---

## Key Design Notes

- The `STRATEGY_MAP` dict maps strategy names to their indices in the 7-tuple, making it easy for search files to select the right metric with `calculate_confidence_score()[strategy_idx]`.
- `calculate_confidence_score()` handles the **empty list case** by returning zeros (no-confidence defaults), preventing crashes on truncated outputs.
- The `score()` function is used for **post-hoc evaluation** only — it's not called during live inference.
