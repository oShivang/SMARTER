# `sal/search/best_of_n.py` — Best-of-N Sampling with PRM Scoring

## What Is This File?

This file implements the **Best-of-N (BoN)** baseline search strategy using a single language model. It generates `N` independent complete solutions for each problem and uses a Process Reward Model (PRM) to score each one, then picks the top-scoring answer.

---

## Input Format

### `best_of_n(x, config, llm, prm)`
| Argument | Type | Description |
|---|---|---|
| `x` | `dict` | A HuggingFace `Dataset` batch; must have key `"problem"` → `list[str]` |
| `config` | `Config` | Global pipeline configuration (see `sal/config.py`) |
| `llm` | `vllm.LLM` | A loaded vLLM model instance (the policy/generation model) |
| `prm` | `PRM` | A loaded PRM scorer (see `sal/models/reward_models.py`) |

---

## Output Format

Returns the same `x` dict with two new keys added:

```python
{
  "problem": ["..."],       # original (unchanged)
  "completions": [          # list[list[str]], N completions per problem
    ["step1\n\nstep2...", "step1b\n\nstep2b...", ...],
    ...
  ],
  "pred": ["..."],          # list[str], the single best completion per problem
  ...
}
```

---

## What It Does

Best-of-N is the simplest search approach: generate many independent answers, score them all, and pick the best. No iterative pruning or inter-step decisions are made.

**High-level flow:**
1. Format each problem into a chat conversation with the system prompt.
2. Duplicate each conversation `N` times to use vLLM's continuous batching efficiently.
3. Generate one completion per duplicated prompt.
4. Score all completions with the PRM.
5. Aggregate per-step PRM scores into a single scalar per completion using `agg_strategy`.
6. Select the completion with the highest aggregate score as `pred`.

---

## How It Does It (In Detail)

### Step 1 — Conversation Formatting
Each problem string from `x["problem"]` is wrapped in a 3-turn conversation:
```python
[{"role": "system", "content": config.system_prompt},
 {"role": "user", "content": problem}]
```
The tokenizer's `apply_chat_template()` converts this to a single prompt string. If `config.custom_chat_template` is set, that Jinja2 template overrides the tokenizer's default.

### Step 2 — Prompt Duplication
To avoid running `N` separate batches, each prompt is repeated `N` times in the input list. For example, with 3 problems and N=2:
```
[p1, p1, p2, p2, p3, p3]
```
This allows vLLM to batch all 3×N generations in a single continuous-batching call, maximizing GPU utilization.

### Step 3 — Generation
`llm.generate()` is called once with:
- `temperature`, `top_p`, `max_tokens` from config
- `n=1` (one output per prompt slot; duplication already handles the N factor)

### Step 4 — Completion Extraction
The flat list of `N * num_problems` responses is sliced back into groups of N:
```python
completions[i] = [r.outputs[0].text for r in responses[i*N : (i+1)*N]]
```
Token counts are tracked for telemetry.

### Step 5 — PRM Scoring
`prm.score(questions, completions)` returns per-step scores with shape `[num_problems][N][num_steps]`.

### Step 6 — Score Aggregation and Selection
`aggregate_scores(step_scores, config.agg_strategy)` reduces the per-step list to a scalar:
- `"last"` → take the final step's score
- `"min"` → take the minimum step score (pessimistic)
- `"prod"` → product of all step scores

The completion with the maximum aggregate score is returned as `pred`.

---

## Key Design Notes

- This is the **reference baseline** — no speculative correction, no iterative search, just raw sampling + PRM selection.
- The entire function is designed to be used as a `dataset.map()` callback with `batched=True`, making it easy to parallelize over a dataset.
- `config.n` controls the number of candidates. Larger N improves quality at linear compute cost.
