# `sal/search/best_of_n_smart_conf.py` — SMART Best-of-N with Confidence-Based Intervention

## What Is This File?

This file implements the most advanced Best-of-N variant: **SMART BoN with confidence-based step evaluation**. It replaces the PRM scorer with **token log-probability confidence metrics**, and uses a **post-hoc iterative loop** that repeatedly runs the SLM until all candidates are completed — with the LLM stepping in at each low-confidence bottleneck step.

---

## Input Format

### `smart_best_of_n_conf(x, config, slm, llm, prm=None)`
| Argument | Type | Description |
|---|---|---|
| `x` | `dict` | Dataset batch with key `"problem"` → `list[str]` |
| `config` | `Config` | Config with `conf_strategy`, `threshold`, `n`, `num_iterations`, etc. |
| `slm` | `vllm.LLM` | Small/draft language model |
| `llm` | `AutoModelForCausalLM` | Large/teacher language model |
| `prm` | `PRM` | **Unused** — present for API compatibility |

---

## Output Format

```python
{
  "problem": [...],
  "completions": [["solution_1a", "solution_1b", ...], ...],
  "pred": ["best_solution_per_problem", ...],
}
```

---

## What It Does

This implements a **Post-Hoc SMART loop**: instead of generating all steps first and then correcting, this file runs an **iterative generation cycle**:
1. SLM generates full remaining text.
2. The output is **chunked step-by-step** by `\n\n` boundaries.
3. The confidence of each step is evaluated on the fly.
4. The first step with **low confidence** (the "bottleneck") triggers an LLM correction.
5. The LLM's replacement step is accepted and the trajectory continues into the next iteration.
6. This repeats until all trajectories are marked `completed`.

---

## How It Does It (In Detail)

### Initialization

Two tokenizers are set up:
- `slm_tokenizer`: from the vLLM engine's built-in tokenizer
- `llm_tokenizer`: loaded with `AutoTokenizer.from_pretrained(config.model_path)` and **cached** in a module-level `_TOKENIZER_CACHE` dict to avoid repeated disk loads.

A list of trajectory dicts is created — one per `(problem, n)` pair — tracking `current_text`, `history`, `all_scores`, `completed`, `smart_step`, and `gen_update`.

### Iterative Post-Hoc Loop

For each iteration (`0` to `config.num_iterations`):

#### SLM Full Generation
- Active (not-yet-completed) trajectories are formatted into conversations.
- `slm.generate(..., logprobs=10)` generates the **full remaining text** for each trajectory. The `logprobs=10` parameter collects per-token log-probability data needed for confidence scoring.

#### Step-Level Chunking
The SLM output is chunked token-by-token until a `\n\n` boundary is detected:
```python
text = slm_tokenizer.decode(step_token_ids)
if text.endswith("\n\n"):
    steps.append((text, step_logprobs))
    step_token_ids, step_logprobs = [], []
```
This preserves the per-step logprob lists needed for confidence scoring.

#### Bottleneck Detection
For each step, `calculate_confidence_score(logprob_list)` computes a vector of metrics. The configured strategy's metric is extracted and checked against `config.threshold` using `needs_correction()`:
```python
if needs_correction(score, config.threshold, config.conf_strategy):
    first_bad_step_idx = step_idx
    break
```
All steps *before* the first bad one are accepted and added to the trajectory.

#### LLM Surgical Intervention (Batched)
If any trajectories hit a bottleneck, all their correction requests are batched:
1. Each trajectory's current good-step text is templated into an LLM conversation.
2. All prompts are left-padded and tokenized together.
3. A single `llm.generate(...)` call with `StopStringCriteria` generates exactly one replacement step per trajectory.
4. The LLM's step replaces the SLM's bad step in the trajectory's history.

#### No-Bottleneck Case
If no bad step is found in a trajectory, all generated steps are accepted and the trajectory is marked `completed = True`.

### Output Formatting
After all iterations, the results are grouped by problem, and the trajectory with the highest aggregate confidence score is selected as `pred`.

---

## Key Design Notes

- This is the **"Surgical Scaffolding"** algorithm from the BTP research — the most elaborate variant, designed to minimize LLM token usage while maximizing step quality.
- The **tokenizer cache** (`_TOKENIZER_CACHE`) is critical for performance: loading the LLM tokenizer from disk on every call would add seconds of overhead per batch.
- The `prm_update` field (used by PRM-based SMART variants) is **not populated** here — confidence scores are stored in `all_scores` instead.
- `config.num_iterations` acts as a **safety cap**: even if not all trajectories are done, the loop exits to prevent infinite generation on degenerate inputs.
