# `sal/config.py` — Global Configuration Dataclass

## What Is This File?

This file defines the **central configuration object** (`Config`) for the entire SMARTER pipeline. Every script, search algorithm, and evaluation routine reads its hyperparameters and runtime settings from an instance of this dataclass. It is the single source of truth for how a run is configured.

---

## Input Format

This file does **not** read inputs at runtime directly. Instead, a `Config` object is instantiated either:

- **Programmatically** — in Python by passing keyword arguments to `Config(...)`.
- **Via a YAML file** — parsed by `H4ArgumentParser` (defined in `sal/utils/parser.py`) and deserialized into a `Config` instance.
- **Via CLI flags** — overrides on top of a YAML file, e.g. `--temperature=0.9`.

**Example YAML input:**
```yaml
approach: beam_search
model_path: meta-llama/Llama-3.2-1B-Instruct
prm_path: RLHFlow/Llama3.1-8B-PRM-Deepseek-Data
n: 16
beam_width: 4
num_iterations: 40
dataset_name: HuggingFaceH4/MATH-500
output_dir: ./output
```

---

## Output Format

Produces a `Config` **dataclass instance** that is passed as a first-class argument to all major functions in the pipeline. It is never serialized on its own, but it drives all downstream behavior.

---

## What It Does

The `Config` dataclass bundles **all configurable knobs** of the inference pipeline into one object:

| Category | Key Fields |
|---|---|
| **Search Algorithm** | `approach`, `n`, `beam_width`, `num_iterations`, `lookahead` |
| **Models** | `model_path` (draft/SLM), `draft_model_path`, `prm_path` |
| **Scoring** | `score_method`, `agg_strategy`, `conf_strategy`, `threshold`, `prm_threshold` |
| **Sampling** | `temperature`, `top_p`, `max_tokens`, `seed` |
| **Dataset** | `dataset_name`, `dataset_split`, `dataset_start`, `dataset_end`, `num_samples` |
| **Output** | `output_dir`, `push_to_hub`, `hub_dataset_id` |
| **Misc** | `gpu_memory_utilization`, `load_in_4bit`, `apply_voting`, `measure_flops` |

---

## How It Does It (In Detail)

### 1. Dataclass Declaration
`Config` is a `@dataclass` with Python type hints. Default values are provided for every field, so creating `Config()` with no arguments gives a sensible baseline configuration.

### 2. Post-Init Validation (`__post_init__`)
After instantiation, the `__post_init__` method runs automatically and performs:

- **DVTS check:** If `approach == "dvts"`, it verifies that `n` is divisible by `beam_width` and precomputes `n_beams = n // beam_width`.
- **Beam search batch check:** If `approach == "beam_search"` and `search_batch_size != 1`, it emits a warning and resets it to 1 (batched beam search is not yet supported).
- **Hub setup:** If `push_to_hub=True`, it auto-generates a `revision` string encoding all key hyperparameters (temperature, top_p, n, beam_width, etc.) so that each experiment gets a unique branch on the HuggingFace Hub. It also calls `get_dataset_revisions()` to check if that revision already exists, and exits early if `overwrite_hub_revision=False`.

### 3. System Prompt
The `system_prompt` field contains a full instruction string injected at the start of every model conversation. It instructs the model to:
- Separate reasoning steps with exactly two newlines (`\n\n`)
- Use step-by-step formatting for complex problems
- Always end with `Therefore, the final answer is: $\boxed{answer}$`

### 4. Custom Chat Template
The `custom_chat_template` field holds a Jinja2 template string for formatting conversations in the LLaMA-3 chat format, complete with `<|start_header_id|>` tokens. This is applied to the tokenizer at generation time.

---

## Key Design Notes

- The `score_method` field (`'prm'` or `'conf'`) switches between PRM-based scoring and confidence-based scoring throughout the pipeline.
- The `smart_search` flag (`bool`) indicates whether SMART (Speculative Mediation At Reasoning Time) interventions are active.
- The `threshold` is used by SMART variants to decide when a step's score is "low enough" to trigger a large-LLM correction.
