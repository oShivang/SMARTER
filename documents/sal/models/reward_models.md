# `sal/models/reward_models.py` — Process Reward Model (PRM) Wrappers

## What Is This File?

This file implements the **Process Reward Model (PRM)** abstraction layer. A PRM evaluates the quality of each reasoning *step* in a multi-step mathematical solution, rather than evaluating only the final answer. The scores it produces are used by all search algorithms (Best-of-N, Beam Search, SMART variants) to rank and prune candidate solutions.

---

## Input Format

### `batched_math_shepherd_inference(model, tokenizer, inputs, batch_size)`
| Argument | Type | Description |
|---|---|---|
| `model` | `PreTrainedModel` | A loaded HuggingFace causal LM |
| `tokenizer` | `PreTrainedTokenizer` | Corresponding tokenizer |
| `inputs` | `list[str]` | Flat list of formatted prompt+completion strings with `ки` step tags |
| `batch_size` | `int` | How many strings to process per GPU forward pass |

### `PRM.score(questions, outputs, batch_size)`
| Argument | Type | Description |
|---|---|---|
| `questions` | `list[str]` | Raw problem/question strings |
| `outputs` | `list[list[str]]` | For each question, a list of N candidate completions (multi-step text) |
| `batch_size` | `int` | Batch size for inference |

---

## Output Format

All `score()` methods return:
```python
list[list[list[float]]]  # shape: [num_questions][num_candidates][num_steps]
```
- **Outer list:** one entry per question
- **Middle list:** one entry per candidate completion
- **Inner list:** one float per reasoning step, representing the PRM's confidence that this step is correct

---

## What It Does

The file provides three classes:

| Class | Model Used | Step Tag | Method |
|---|---|---|---|
| `PRM` | Abstract base | — | Interface definition |
| `MathShepherd` | `peiyi9979/math-shepherd-mistral-7b-prm` | `ки` (Cyrillic) | Token-level logit at tag positions |
| `RLHFFlow` | `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data` | `+` / `ки` (mask) | Dialogue-style, single or batched |

A factory function `load_prm(config)` selects and returns the right PRM based on `config.prm_path`.

---

## How It Does It (In Detail)

### `MathShepherd`

1. **Formatting:** Each completion is reformatted by inserting `ки` (a special Cyrillic token, ID `12902`) after every `\n\n` separator. This marks each step boundary.
2. **Forward pass:** The full string (prompt + specially formatted completion) is tokenized and fed to the model. The logits at positions where `input_ids == STEP_TAG_ID (12902)` are extracted.
3. **Score extraction:** At those positions, a softmax is applied over only two logits — token IDs `648` (good) and `387` (bad). The probability of the "good" token is taken as the step score.
4. **Reshaping:** Since inputs are batched flat, the per-step scores are then split back into sublists matching the original `[question][candidate]` structure.

### `RLHFFlow`

This PRM was trained on a dialogue format where each step is presented as a user message and the PRM responds with `+` (good) or `-` (bad).

**Single mode (`_score_single`):**
- Builds a multi-turn conversation for each answer, appending `{"role": "assistant", "content": "+"}` after each step.
- Scores are read from the `-3` token position in the logits (the model predicts the `+`/`-` token at a fixed offset before the EOS).

**Batched mode (`_score_batched`):**
- Constructs two parallel conversations: one with real `+` tokens and one with `ки` (a mask/dummy token) at assistant positions.
- The `ки` conversation is used as a mask to locate step positions in the padded batch tensor without confusion from padding.
- Both conversations are tokenized and padded together.
- The `+`-probability at each `ки` position (shifted by one) is extracted as the step score.
- Scores are reshaped back to `[question][candidate][step]`.

---

## Key Design Notes

- **GPU memory management:** `torch.cuda.empty_cache()` is called after every batch to prevent OOM errors on long benchmarks.
- **Batching strategy:** Math Shepherd batches flat inputs; RLHFFlow batches conversations. Both respect a configurable `batch_size` to control memory usage.
- **The `load_prm()` factory** makes it straightforward to swap PRM implementations by changing `config.prm_path`.
