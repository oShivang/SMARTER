# `sal/search/best_of_n_smart.py` — SMART Best-of-N with PRM-Based Intervention

## What Is This File?

This file implements **SMART Best-of-N** — a speculative decoding variant where a **Small Language Model (SLM)** generates all `N` candidate solutions, and a **Large Language Model (LLM)** surgically corrects any reasoning step that fails a PRM quality threshold. The key idea is: use the fast/cheap SLM for most work, and only invoke the expensive LLM when the PRM flags a step as poor quality.

---

## Input Format

### `smart_best_of_n(x, config, slm, prm, llm)`
| Argument | Type | Description |
|---|---|---|
| `x` | `dict` | Dataset batch with key `"problem"` → `list[str]` |
| `config` | `Config` | Pipeline config (`prm_threshold`, `n`, `temperature`, etc.) |
| `slm` | `vllm.LLM` | Small/draft language model (fast, cheap) |
| `prm` | `PRM` | Process Reward Model to evaluate step quality |
| `llm` | `AutoModelForCausalLM` | Large/teacher language model (slow, expensive) |

---

## Output Format

Returns the same `x` dict with new keys:
```python
{
  "problem": [...],
  "completions": [["fixed_solution_1a", ...], ...],
  "pred": ["best_completion_per_problem", ...],
}
```

---

## What It Does

This is a **two-phase** search approach:

1. **Draft Phase:** The SLM generates `N` full solutions for every problem.
2. **Surgical Correction Phase:** The PRM scores all candidate solutions step-by-step. For any candidate where a step falls below `config.prm_threshold`, the LLM regenerates that one failing step (and only that step). The SLM then picks up and continues from where the LLM left off.

---

## How It Does It (In Detail)

### Phase 1 — SLM Draft Generation

1. Each problem is templated into a full chat format using `convert_to_chat_template()`.
2. Each prompt is duplicated `N` times for parallel batching.
3. `slm.generate()` produces one complete, multi-step solution per slot.
4. Responses are unpacked into `completions[problem_idx][candidate_idx]` — a 2D list of strings.

### Phase 2 — PRM Scoring and Bottleneck Detection

`prm.score(x["problem"], completions)` returns scores with shape `[problem][candidate][step]`.

For each problem/candidate, the first step that falls below `config.prm_threshold` is found:
```python
for step_idx, step_score in enumerate(score):
    if step_score < config.prm_threshold:
        fixes_needed.append((problem_idx, candidate_idx, step_idx))
        break  # Only fix the first failing step
```

### Phase 3 — LLM Surgical Intervention (Batched)

For all flagged `(problem, candidate, step)` tuples:
1. The completion is split into steps by `\n\n`.
2. Only the valid steps *before* the failing one are kept as `partial_completion`.
3. The LLM is invoked (batched) to regenerate the failing step given the valid prefix.
4. A `StopStringCriteria` on `\n\n` ensures the LLM generates exactly one step.

```python
new_ids = llm.generate(
    input_ids["input_ids"], attention_mask=...,
    stopping_criteria=[stopping_criteria],
    max_new_tokens=config.max_tokens,
)
```

### Phase 4 — SLM Continuation

For each fixed candidate, the new partial completion (original prefix + LLM-generated step) is fed back to the SLM to generate the remainder of the solution.

### Phase 5 — Final Re-Scoring and Selection

`prm.score()` is called again on the updated completions. The candidate with the highest aggregate PRM score is selected as `pred`.

---

## Key Design Notes

- The LLM intervention is **targeted**: only the first failing step per candidate triggers an LLM call. This minimizes LLM usage.
- **Batched LLM generation:** All failing steps across all problems and candidates are processed in a single `llm.generate()` call for maximum GPU utilization.
- The `prm_threshold` hyperparameter (from `config`) controls the aggressiveness of intervention — lower threshold = fewer corrections, higher = more frequent LLM invocations.
- The `llm` here is a **HuggingFace `AutoModelForCausalLM`** (not vLLM), because it needs fine-grained `stopping_criteria` control.
