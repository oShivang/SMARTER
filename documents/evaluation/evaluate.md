# `evaluation/evaluate.py` — Benchmark Evaluation Script

## What Is This File?

This file is the **offline evaluation module** for the SMARTER pipeline. Given a JSONL file of model completions and predictions, it computes accuracy scores by comparing extracted model answers against ground-truth answers using symbolic math equality checking.

---

## Input Format

### `evaluate(data_name, prompt_type, samples, file_path, max_num_samples, execute, pred_keys)`
| Argument | Type | Description |
|---|---|---|
| `data_name` | `str` | Dataset name (e.g. `"math"`, `"gsm8k"`) — controls parsing heuristics |
| `prompt_type` | `str` | Prompt format type (e.g. `"tool-integrated"`) — passed to parsers |
| `samples` | `list \| None` | Pre-loaded list of sample dicts (alternative to `file_path`) |
| `file_path` | `str \| None` | Path to a JSONL file with model outputs |
| `max_num_samples` | `int \| None` | Cap on number of samples to evaluate |
| `execute` | `bool` | Whether to execute Python code in completions (for tool-integrated prompts) |
| `pred_keys` | `list[str]` | Column names to use as final prediction keys (e.g. `["pred"]`) |

**JSONL sample format:**
```json
{
  "idx": 0,
  "problem": "What is 2 + 2?",
  "answer": "4",
  "pred": "Therefore, the final answer is: $\\boxed{4}$.",
  "completions": ["...solution1...", "...solution2..."]
}
```

**CLI usage:**
```bash
python evaluate.py --data_name math --file_path output/results.jsonl
```

---

## Output Format

### `evaluate()` → `(list[dict], dict)`
- **`samples`**: The input sample list, augmented with:
  - `"gt"`: parsed ground truth answer
  - `"preds"`: list of extracted predicted answers (one per `pred_key`)
  - `"correct"`: list of booleans (one per pred_key, True = correct)
  - `"pred_completions"`: all completion texts extracted as answers
  - `"correct_completions"`: correctness per completion

- **`result_json`**: Summary statistics:
  ```python
  {
    "num_samples": 500,
    "num_scores": 500,
    "timeout_samples": 2,
    "empty_samples": 3,
    "acc": {"pred": 72.4}
  }
  ```

### `get_result()` → `dict`
A simplified score dict keyed by `pred_key`:
```python
{"pred": 72.4, "pred_weighted@16": 75.1, ...}
```

---

## What It Does

The `evaluate()` function runs the full **answer extraction and grading pipeline** on a set of model outputs:

1. **Load samples** from a JSONL file or pre-loaded list; index by `"idx"` if present.
2. **Parse ground truth** using `parse_ground_truth()` (from `evaluation/parser.py`).
3. **Extract predictions** using `extract_answer()` for each `pred_key` column.
4. **Grade predictions** against ground truth using `math_equal_process()` (from `evaluation/grader.py`), which checks mathematical equivalence.
5. **Grade all completions** (not just the final pred) for oracle / pass@k analysis.
6. **Aggregate** correctness into per-column accuracy percentages.

---

## How It Does It (In Detail)

### Ground Truth Parsing
`parse_ground_truth(sample, data_name)` extracts the canonical correct answer from the dataset's native format (handles differences between MATH, GSM8K, BoolQ, etc.).

### Answer Extraction
`extract_answer(sample[pred_key], data_name)` is called for each `pred_key` to pull the final boxed answer from the model's output text.

### Math Equality Checking
Each `(pred, gt)` pair is passed to `math_equal_process()`, which wraps the `grader.py` equality checker with a timeout. Grading can fail with `TimeoutError` (e.g., sympy hangs on complex expressions), in which case the sample is scored as incorrect and `timeout_cnt` is incremented.

The evaluation loop uses **sequential processing** (not the `pebble.ProcessPool` import that's present in the code, but a simple for-loop with individual exception handling).

### Score Matrix Construction
All sample correctness lists are collected into `score_mat` (a list of lists). The columns may have different lengths (some samples have more completions than others), so shorter rows are **padded** by repeating the last value:
```python
score_mat[i] = s + [s[-1]] * (max_len - len(s))
```
Column means of `score_mat` give accuracy per `pred_key`.

---

## Key Design Notes

- The `"completions"` scoring provides an **oracle analysis**: if *any* of the N completions is correct, the sample gets credit — useful for measuring the theoretical ceiling of the search approach.
- `timeout_cnt` and `empty_samples` statistics help diagnose systematic failure modes (e.g., model refusing to answer, or very slow sympy equivalence checks).
- The `execute` flag (for Python code execution) is plumbed through but not fully implemented in this script — it is carried over from the original evaluation framework where tool-integrated reasoning was evaluated by executing generated Python code.
