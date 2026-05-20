# `evaluation/math_eval.py` — Main Math Evaluation Runner

## What Is This File?

This file is the main entry point to evaluate LLMs on mathematical reasoning benchmarks. It configures the runtime environment, runs standard or tool-use inferences (CoT, PAL/PoT, and Multi-turn Tool-Integrated Reasoning), extracts predictions, parses outcomes, and outputs benchmark performance reports.

---

## Input Format

### Command Line Interface (CLI) Arguments
Runs via shell with flags:
- `--data_names`: Comma-separated dataset list (e.g. `gsm8k,math`).
- `--model_name_or_path`: Model path/identifier (Hugging Face repo or local folder).
- `--prompt_type`: Reasoning style, e.g. `cot` (Chain of Thought), `pal` (Program-Aided Language), `tool-integrated` (interleaved math and code execution).
- `--use_vllm`: Boolean flag to toggle vLLM engine acceleration.
- `--n_sampling`: Number of completions to sample per question (e.g. for majority voting).
- `--temperature` / `--top_p`: Sampling hyper-parameters.

---

## Output Format

1. **`{out_file}.jsonl`**: Stores prompt inputs, outputs, extracted predictions, and execution reports for every test sample.
2. **`{out_file}_metrics.json`**: Standard summary metrics containing:
   - Evaluated accuracy (`acc`) per dataset and average (`avg`).
   - Timing measurements (`time_use_in_second`, `time_use_in_minite`).

---

## What It Does

`math_eval.py` orchestrates the entire offline benchmark pipeline end-to-end through 5 key stages:

1. **Load Data**: Calls **`data_loader.py`** to load and cache the target dataset splits.
2. **Build Prompts**: Generates formatting templates and prepends few-shot demonstration exemplars (loaded from **`examples.py`** and formatted using helper functions in **`utils.py`**).
3. **Generate & Execute (Interactive Loop)**:
   * Boots up the model (via high-throughput `vllm` or Hugging Face Transformers).
   * Orchestrates the multi-turn generation loop.
   * If a model outputs python code (in PAL/ToRA mode), `math_eval` intercepts the block, runs it locally via **`PythonExecutor`**, appends the execution output back into the prompt conversation history, and prompts the model again to continue (running up to 4 epochs/turns).
4. **Orchestrate Evaluation**: After all generations are complete, it passes the full trajectories to **`evaluate.py`** which leverages **`parser.py`** (to extract boxed answers) and **`grader.py`** (to evaluate mathematical or text equivalence).
5. **Output Metrics**: Saves the final predictions to `.jsonl` and compiles aggregate performance statistics into a metrics JSON file.

---

## How It Does It (In Detail)

### 1. Unified Model Initialization (`setup`)

- If `--use_vllm` is enabled: Initializes vLLM's `LLM` class. Distributes weights across available GPUs according to pipeline/tensor parallel args.
- Otherwise: Calls `load_hf_lm_and_tokenizer` to load weights in float16 half precision.

### 2. Multi-turn Tool-Use Loop (`main`)

For interactive prompting strategies, the model writes markdown code blocks:
- **Inference Epochs**: Up to `max_func_call` turns (typically 4).
- **Output Parsing**:
  - In each turn, the script extracts python snippets using `extract_program()` from the trajectory.
  - The script executes the code using `PythonExecutor.batch_apply()`.
  - The execution output is formatted back as a ```` ```output\n...\n``` ```` block and appended to the generation prompt.
  - The model continues generating from that point.
- **Answer Extraction**: Uses `run_execute()` to parse final values from the text and code executions.

### 3. Evaluation and Reporting

- After completing generation, it runs clean-ups (e.g. choice extraction, normalizing characters).
- Runs `evaluate()` to compute accuracy metrics.
- Caches outputs to disk and writes a summary metrics file.
