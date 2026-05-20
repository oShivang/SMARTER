# `sal/search/best_of_n_conf.py` — Best-of-N with Token Confidence Scoring

## What Is This File?

This is a **confidence-based variant of Best-of-N** that replaces the PRM (Process Reward Model) scorer with token-level probability metrics derived from the model's own output log-probabilities. It does not require a separate reward model, making it lighter-weight and faster.

---

## Input Format

### `best_of_n_conf(x, config, llm, prm)`
| Argument | Type | Description |
|---|---|---|
| `x` | `dict` | Dataset batch; must have key `"problem"` → `list[str]` |
| `config` | `Config` | Pipeline configuration (`conf_strategy`, `n`, etc.) |
| `llm` | `vllm.LLM` | The generation model |
| `prm` | `PRM` | **Unused** in this file — present for API compatibility |

---

## Output Format

Same structure as `best_of_n.py`:
```python
{
  "problem": [...],
  "completions": [["completion_1a", "completion_1b", ...], ...],
  "pred": ["best_completion_per_problem", ...],
}
```

---

## What It Does

Generates `N` complete solutions per problem using the SLM, scores each using **token log-probability statistics** (not a PRM), and returns the best-scoring one.

The confidence scoring strategy is controlled by `config.conf_strategy`, which can be:
| Strategy | Description |
|---|---|
| `"probs_mean"` | Mean token probability across all output tokens |
| `"likelihood"` | Total log-likelihood (exp of summed log-probs) |
| `"likelihood_mean"` | Geometric mean of per-token probabilities |
| `"entropy"` | Mean per-token entropy (lower = more confident) |
| `"top_2_diff"` | Mean difference between top-1 and top-2 token probability |
| `"mean_least_3"` | Mean of the 3 least probable tokens (uncertainty signal) |

---

## How It Does It (In Detail)

### Step 1 — Conversation Formatting and Duplication
Identical to `best_of_n.py`: problems are wrapped into chat format, then duplicated `N` times for efficient batching.

### Step 2 — Generation with Log-Probs
The key difference from `best_of_n.py` is the `SamplingParams`:
```python
SamplingParams(..., logprobs=10)
```
Setting `logprobs=10` tells vLLM to return the log-probabilities of the top-10 tokens at each generation step, which is what `calculate_confidence_score()` needs for entropy, top-2-diff, and similar metrics.

### Step 3 — Confidence Score Computation
After generation, for each response:
```python
calculate_confidence_score(output.logprobs)[strategy_idx]
```
`calculate_confidence_score()` (from `sal/utils/score.py`) takes the list of per-token logprob dicts and computes all metrics (likelihood, entropy, etc.) at once, returning a tuple. The `strategy_idx` selects which metric to use based on `config.conf_strategy`.

### Step 4 — Prediction Selection
- For **maximization metrics** (`probs_mean`, `likelihood`, etc.): `argmax` is used.
- For **uncertainty metrics** (`entropy`, `mean_least_3`): `argmin` is used (lower = more confident).

```python
pred_indices = [np.argmax(s) if config.conf_strategy not in ["entropy", "mean_least_3"]
                else np.argmin(s) for s in agg_scores]
```

---

## Key Design Notes

- This file **does not call `prm.score()`** at all. The `prm` argument is a no-op, kept only for API uniformity with the other search functions.
- Unlike PRM scoring, confidence scoring is **self-contained**: the model evaluates its own certainty using its own logits, with no external reward model.
- `logprobs=10` is a vLLM parameter that returns the top-10 token logprobs at each position. The extra tokens are needed for entropy and top-2-diff calculations.
