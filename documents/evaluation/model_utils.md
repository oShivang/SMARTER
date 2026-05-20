# `evaluation/model_utils.py` — Model Loading and Inference Utilities

## What Is This File?

This file contains utility functions for **loading Hugging Face models and tokenizers**, configuring parameters (like quantization and half precision), managing **batch inference**, and enforcing **custom sequence-level stopping criteria** during generation.

---

## Main Entry Points

### 1. `load_hf_lm_and_tokenizer`
Loads causal language models and tokenizers from Hugging Face or local weight directories.
* **Arguments**:
  * `model_name_or_path` (`str`): Weights directory or repository identifier.
  * `tokenizer_name_or_path` (`str | None`): Custom tokenizer path (defaults to model path if `None`).
  * `load_in_8bit` (`bool`): Enables 8-bit quantization mapping.
  * `load_in_half` (`bool`): Loads weights in `float16` precision.
  * `gptq_model` (`bool`): Toggles AutoGPTQ loading wrapper.
  * `padding_side` (`str`): Alignment configuration (defaults to `"left"` for decoder generation).
* **Returns**: `(AutoModelForCausalLM, AutoTokenizer)`

### 2. `generate_completions`
Executes token generation over input prompts in batches with active token tracking and post-generation pruning.
* **Arguments**:
  * `model` (`PreTrainedModel`): Loaded model instance.
  * `tokenizer` (`PreTrainedTokenizer`): Associated tokenizer.
  * `prompts` (`list[str]`): Batched list of raw text prompts.
  * `batch_size` (`int`): Batch width per forward pass.
  * `stop_id_sequences` (`list[str]`): Text sequences indicating termination.
  * `**generation_kwargs`: Forwarded generation arguments (e.g. `max_new_tokens`, `temperature`).
* **Returns**: `list[str]` (generated outputs with prompts and stop strings stripped out).

---

## Detailed Stop Criteria Logic

Standard Hugging Face models support stopping generation only on specific tokens (like `<|endoftext|>`). However, mathematical and programming models require halting on arbitrary string boundaries (such as a newline `\n`, separator `---`, or code result tag ```` ```output ````). 

To achieve this, `model_utils.py` defines three classes extending PyTorch's `StoppingCriteria`:

1. **`KeywordsStoppingCriteria` (Primary)**:
   * Decodes current generated token sequences back into text strings on each step.
   * If any of the target string boundaries (`stop_id_sequences`) are detected in the decoded sequence, it reports the sequence as finished.
2. **`KeyWordsCriteria` (Token-based)**:
   * Compares the trailing token ID integers against the target token ID sequences directly without decoding them back to text strings.
3. **`KeyWordsCriteriaTrunc`**:
   * Evaluates token IDs starting specifically from the prompt boundary index to identify matching stops.

### Batch-Level Cleanup:
Because Hugging Face's batch generation evaluates criteria at the batch level (the loop runs until *all* inputs in the batch hit a stop condition), some sequences may generate tokens past their local stop points. To fix this, `generate_completions()` applies a post-generation sweep:
```python
for idx, prediction in enumerate(batch_generations):
    for stop_sequence in stop_id_sequences:
        batch_generations[idx] = prediction.split(stop_sequence)[0]
```

