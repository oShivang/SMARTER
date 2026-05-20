# `evaluation/utils.py` — General Evaluation and Prompt Construction Utilities

## What Is This File?

This file contains general utility functions for setting seeds, loading and saving files, and building formatted few-shot prompts using pre-defined templates.

---

## Input Format

### `construct_prompt(example, data_name, args)`
| Argument | Type | Description |
|---|---|---|
| `example` | `dict[str, Any]` | Dictionary representing a dataset sample (must contain `"question"`). |
| `data_name` | `str` | Name of the target dataset. |
| `args` | `Namespace` | CLI argument configurations (e.g. `prompt_type`, `num_shots`, `adapt_few_shot`). |

---

## Output Format

- **`construct_prompt()`**: Returns a formatted prompt string ready to be fed into the model.

---

## What It Does

1. **Deterministic Seeds**: Initializes random number generators (`random`, `numpy.random`) and environment variables to ensure reproducible outputs.
2. **File I/O Helpers**:
   - `load_jsonl()`: Reads files line-by-line using generator expressions.
   - `save_jsonl()`: Automatically creates directories and writes list outputs to disk.
3. **Structured Prompt Synthesis**:
   - Matches datasets to their corresponding prompt templates (e.g. `cot`, `pal`, `tora`, `deepseek-math`, `numina`).
   - Retrieves few-shot demonstrations from `examples.py`.
   - Concatenates the demonstrations with the current question to form a unified prompt string.

---

## How It Does It (In Detail)

### 1. File Handling

- **`load_jsonl`**: Uses Python's `yield` generator to stream lines from JSONL files, parsing each line with `json.loads`.
- **`save_jsonl`**: Standardizes dictionary lists, creates output directories, and saves files using UTF-8 encoding.

### 2. Prompt Construction Pipeline (`construct_prompt`)

- **Demonstration Selection**:
  - Checks if `adapt_few_shot` is enabled for multiple-choice questions (loading 5 shots for Chinese Gaokao, or 0 shots otherwise).
  - Retrieves few-shot examples from `examples.py` via `load_prompt()`.
- **Template Application**:
  - Looks up the template format in `PROMPT_TEMPLATES`.
  - Splits templates into user prompt, assistant response, and separator components.
  - Formats each few-shot demonstration using the matched template:
    `input_template.format(input=question) + output_template.format(output=answer)`
- **Special Template Rules**:
  - For `qwen25-math-cot`: Places all few-shot demonstrations into a single turn rather than multi-turn sequences.
  - For `tora` (Tool-integrated reasoning): Prepends guidelines instructing the model to write parameterless functions, use LaTeX formatting, and leverage sympy mathematical representations.
