# `sal/search/beam_search_conf.py` — Beam Search with Confidence-Based Scoring

## What Is This File?

This file implements **Beam Search using token confidence** (log-probability statistics) instead of a PRM for step scoring. Beam pruning decisions are made based on the model's own token probability distribution rather than an external reward signal.

---

## Input Format

### `beam_search_conf(examples, config, llm, prm)`
| Argument | Type | Description |
|---|---|---|
| `examples` | `dict` | Dataset batch with `"problem"` → `list[str]` |
| `config` | `Config` | Config with `conf_strategy`, `n`, `beam_width`, etc. |
| `llm` | `vllm.LLM` | The generation model |
| `prm` | `PRM` | Used **only** for final PRM scoring of completed beams (re-scoring pass) |

---

## Output Format

Same as `beam_search.py`:
```python
{
  "completions": [["sol1", "sol2", ...], ...],
  "pred": ["best_solution_per_problem", ...],
}
```

The final `pred` is chosen by the accumulated **confidence score** (`all_scores`) using `aggregate_scores()`.

---

## What It Does

This runs iterative beam search where at each step, beams are ranked by **self-assessed token-level confidence** (not PRM). The beam with higher mean token probability (or lower entropy, depending on `conf_strategy`) is kept.

At the end, the PRM is called **once** on all completed beams for final score recording (but confidence scores are used for `pred` selection).

---

## How It Does It (In Detail)

### Setup
`SamplingParams` is created with `logprobs=True` to enable log-probability collection. A flag `logprobs=True` is passed at every generation step.

### Main Iteration Loop

Steps 1–3 (activation, conversation building, step generation) are identical to `beam_search.py`, with one key difference: `generate_k_steps_with_responses()` is used instead of `generate_k_steps()`. This variant **returns the raw vLLM response objects** alongside the Beam list, enabling access to `output.logprobs`.

### Confidence Score Extraction
After each step's generation:
```python
conf_scores = [calculate_confidence_score(output.logprobs) for output in responses]
```
The selected strategy's score is extracted:
```python
conf_agg_scores = [[score[0][-1]] for score in conf_scores]  # probs_min_score (index -1)
```

> **Note:** In the current implementation, the hardcoded index `[-1]` selects `probs_min_score` (the minimum token probability in the step), not the strategy specified by `config.conf_strategy`. This is a minor inconsistency in the code.

Each beam's `all_scores` list is updated with this step's confidence value.

### Beam Pruning
`top_indices` are computed by `np.argsort(conf_agg_scores)` — beams with the highest confidence survive:
```python
top_indices = np.argsort(np.array(conf_agg_scores).flatten())[-(config.n // config.beam_width):]
```

### Final Re-Scoring with PRM
After the loop, `prm.score()` is called on all completed beams to record their PRM scores (used for analysis, not for `pred` selection). The prediction is still determined by the accumulated confidence scores.

---

## Key Design Notes

- **Dual scoring:** Confidence scores drive beam pruning decisions; PRM scores are only computed at the end for completeness/analysis.
- The `_with_responses` generation variant is needed specifically to expose `output.logprobs`, which vLLM only populates when `SamplingParams.logprobs=True`.
- **No LLM intervention** occurs in this file — it's a pure single-model search, unlike the `_smart` variants.
- The `conf_agg_scores` index is currently hardcoded to `probs_min_score` — a potential area for improvement to properly respect `config.conf_strategy`.
