# `sal/search/utils.py` — Core Search Primitives and Data Structures

## What Is This File?

This file is the **shared foundation** for all search algorithms in the project. It defines the `Beam` and `GenResult` dataclasses (the core data containers), conversation-building helpers, and four generation functions that wrap LLM calls with step-level stopping logic.

---

## Input Format

### `build_conv(prompt, response, system_prompt)`
| Argument | Type | Description |
|---|---|---|
| `prompt` | `str` | The math problem text |
| `response` | `str \| None` | Partial completion so far (may be empty string) |
| `system_prompt` | `str` | System instruction from `Config.system_prompt` |

### `generate_k_steps(templated_convs, lookahead_steps, llm, sampling_params, beam_width)`
| Argument | Type | Description |
|---|---|---|
| `templated_convs` | `list[str]` | Pre-formatted chat strings (ready for vLLM) |
| `lookahead_steps` | `int` | How many extra greedy steps to generate beyond the first |
| `llm` | `vllm.LLM` | A loaded vLLM model instance |
| `sampling_params` | `vllm.SamplingParams` | Temperature, top_p, max_tokens, stop strings |
| `beam_width` | `int` | Number of parallel candidates per prompt |

The `generate_k_steps_for_llm` variant replaces `vllm.LLM` with a HuggingFace `AutoModelForCausalLM` and adds a `Config` argument for sampling configuration.

---

## Output Format

All `generate_k_steps_*` functions return `list[Beam]` (or `(list[Beam], extra_data)` for the `_with_responses` variants):

```python
[
  Beam(
    prompt="...",
    index=0,
    current_text="",         # accumulated text so far
    next_texts=["step text"],# the newly generated step text(s)
    lookahead_texts=["..."], # next_text + extra greedy steps
    stop_reasons=["\\n\\n"], # why generation stopped
    completion_tokens=[42],  # token counts
    best_scores=[0.0],
    all_scores=[],
    history=[],
    ...
  ),
  ...
]
```

---

## What It Does

### Data Structures

#### `Beam`
The central stateful object that tracks a single hypothesis through the search tree:
- `current_text`: the complete solution text accumulated so far
- `history`: list of individual steps added (for token-counting checks)
- `all_scores`: PRM or confidence scores for each completed step
- `best_scores`: running aggregate scores used for pruning
- `pruned`, `completed`: status flags
- SMART-specific fields: `smart_step`, `prm_update`, `gen_update`, `llm_tokens` (all for logging LLM interventions)

#### `GenResult`
A lightweight intermediate object holding raw generation output for one candidate:
- `first_step_text`: the text generated up to the first `\n\n` stop
- `lookahead_text`: the accumulated text including extra lookahead steps
- `stop_reason`: `"\\n\\n"`, `"EOS"`, or `"length"`
- `completion_tokens`: number of tokens generated

### Helper Functions

- **`build_conv()`**: Assembles a 3-turn conversation `[system, user, assistant]`. If `response` is an empty string, the assistant turn is omitted.
- **`last(x)` / `list_mean(x)`**: Simple aggregation helpers with guard against empty lists.

### Generation Functions

Four variants exist along two axes:
- **Backend:** `vllm.LLM` vs. HuggingFace `AutoModelForCausalLM`
- **Return:** basic `list[Beam]` vs. `(list[Beam], raw_responses)` (the `_with_responses` variants also return raw vLLM outputs for log-prob extraction)

---

## How It Does It (In Detail)

### `generate_k_steps` (vLLM version)

1. **Initialize GenResults:** For each input conversation and each `beam_width` copy, create an empty `GenResult`.
2. **Iterative generation loop (0 to lookahead_steps):**
   - Filter out any `GenResult` that already stopped at `"EOS"`.
   - Concatenate `initial_prompt + lookahead_text` as the current prompt.
   - Call `llm.generate()` on all active prompts in one vLLM batch.
   - Store first-step text on iteration 0; append to `lookahead_text` on all iterations.
   - After iteration 0, switch temperature to 0.0 (greedy decoding for lookahead steps).
3. **Reshape into Beams:** Group GenResults back by their originating prompt index and pack them into `Beam` objects with `next_texts` and `lookahead_texts` lists.

### `generate_k_steps_for_llm` (HuggingFace version)

Same logic but uses HuggingFace's `model.generate()` with:
- `StopStringCriteria` to stop at `\n\n`
- `GenerationConfig` for sampling parameters
- Left-padded batch tokenization for GPU efficiency
- Manual stop reason inference: checks if output ends with `\n\n`, exceeds `max_tokens`, or hit EOS

### `_with_responses` variants

These additionally return the raw vLLM response objects or per-token log-probabilities, enabling confidence-based scoring in the `_conf` search files (e.g., `beam_search_conf.py`, `best_of_n_conf.py`).
