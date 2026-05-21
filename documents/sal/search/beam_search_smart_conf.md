# `sal/search/beam_search_smart_conf.py` — SMART Beam Search with Confidence-Based LLM Intervention

## What Is This File?

This is the **SMART Beam Search variant that uses token confidence** (instead of PRM scores) both to prune beams and to decide when the LLM should intervene. This is the primary "Surgical Scaffolding" algorithm for beam-based search, combining SLM speed with LLM precision triggered by self-assessed confidence.

---

## Input Format

### `smart_beam_search_conf(examples, config, slm, llm, prm=None)`
| Argument | Type | Description |
|---|---|---|
| `examples` | `dict` | Dataset batch with `"problem"` → `list[str]` |
| `config` | `Config` | Config with `conf_strategy`, `threshold`, `n`, `beam_width`, etc. |
| `slm` | `vllm.LLM` | Small Language Model |
| `llm` | `AutoModelForCausalLM` | Large Language Model (correction oracle) |
| `prm` | `PRM` | **Unused** — present for API uniformity |

---

## Output Format

```python
{
  "completions": [["sol1", "sol2", ...], ...],
  "pred": ["best_solution_per_problem", ...],
}
```

Completed beams contain SMART metadata (`smart_step`, `gen_update`, `llm_tokens`). Note: `prm_update` is left empty because no PRM is used.

---

## What It Does

Combines the pruning logic of `beam_search_conf.py` with the LLM intervention logic of `beam_search_smart.py`, but using **confidence scores** as the intervention trigger instead of PRM scores.

At each step:
1. The SLM generates one step per active beam (with `logprobs=10`).
2. Beam confidence scores are computed and used for pruning.
3. Among the surviving beams, those whose confidence is below threshold trigger LLM regeneration of their most recent step.

---

## How It Does It (In Detail)

### Setup
Same as `beam_search_smart.py` but:
- `SamplingParams` includes `logprobs=10` (not just `True`) for richer entropy/top-2 metric support.
- The LLM tokenizer is cached using `slm._llm_tokenizer` (lazy loading pattern): the tokenizer is loaded once from disk and reused across iterations.

### Main Iteration Loop

#### SLM Step Generation
`generate_k_steps_with_responses()` is used (like in `beam_search_conf.py`) to get both the `Beam` list and the raw vLLM responses for log-prob access.

#### Confidence Scoring
```python
conf_scores = []
for output in [o for r in responses for o in r.outputs]:
    conf_scores.append([calculate_confidence_score(output.logprobs)])
strategy_idx = STRATEGY_MAP.get(config.conf_strategy, 2)
conf_agg_scores = [[score[0][strategy_idx]] for score in conf_scores]
```
The strategy index maps `config.conf_strategy` to a position in the confidence metric tuple.

#### Beam Pruning (by Confidence)
Top beams by confidence score survive. Same argsort-based pruning as `beam_search_conf.py`.

#### SMART Intervention Trigger
The `needs_correction()` function is used (instead of a simple threshold comparison):
```python
re_indices = [top_idx for top_idx in top_indices
              if needs_correction(conf_agg_scores[top_idx][0], config.threshold, config.conf_strategy)]
```
`needs_correction()` handles the **direction** of the threshold correctly:
- For `probs_mean`, `likelihood`: `score < threshold` means correction needed (low confidence)
- For `entropy`, `mean_least_3`: `score > threshold` means correction needed (high uncertainty)
- For `top_2_diff`: `score < threshold` means correction needed (ambiguous top tokens)

#### LLM Regeneration
For flagged beams, the state *before* the current SLM step (`prev_active_beams[idx]`) is used to construct the prompt:
1. **Meta-Prompt Formatting**: The problem and current correct steps (`current_text`) are structured using a dedicated `build_meta_conv` function instructing the LLM that it is helping a smaller model and must wrap the single next step inside `\boxed{}`.
2. **Fresh Assistant Turn**: Prompt template uses `add_generation_prompt=True` and `continue_final_message=False` to start a new assistant turn.
3. **Step Generation**: `generate_k_steps_for_llm()` generates a replacement step using HuggingFace's `model.generate()` with `StopStringCriteria`.
4. **Boxed Step Extraction**: The output is parsed using an inline `find_box()` utility to isolate the correct step from any surrounding conversational text, falling back to the raw response if no LaTeX box is found.

#### Metadata Logging
Per corrected beam:
```python
beam.smart_step.append(iterate_idx)
beam.gen_update.append((slm_step, llm_step))
beam.llm_tokens.append(token_count)
beam.all_scores = active_beams[re_idx].all_scores  # reuse original confidence
```
Note: confidence scores from the *original SLM step* are kept (not recomputed from LLM output), because the LLM doesn't return logprobs in this flow.

### Sentinel Values
If no SMART intervention occurred in the entire run (`smart_done=False`), all beams' SMART fields are set to sentinel `[-1]`.

---

## Key Design Notes

- **Tokenizer caching** (`slm._llm_tokenizer`) avoids repeated `AutoTokenizer.from_pretrained()` calls across iterations — critical for performance on long benchmarks.
- `needs_correction()` makes this file **strategy-aware**: the intervention direction correctly flips for entropy/uncertainty-based metrics.
- This file is the **confidence-based equivalent** of `beam_search_smart.py`. Both share the same core loop structure; they differ only in the scoring metric (PRM vs. confidence) and the intervention trigger function.
- `prm_update` is intentionally left empty since confidence scores are stored in `all_scores` directly.
