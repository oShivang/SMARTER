# `evaluation/rm_maj_eval.py` — Reward Model and Majority Voting Evaluation

## What Is This File?

This file contains functions to calculate offline metrics like `maj@k` (majority voting across $k$ candidate completions) and `rm@k` (selecting the best solution according to reward model scores out of $k$ candidates) on generated samples.

---

## Input Format

### `eval_maj_k_metrics(data_path, k)` / `eval_rm_k_metrics(data_path, k)`
| Argument | Type | Description |
|---|---|---|
| `data_path` | `str` | Path to a JSONL file containing predictions, reward model scores, and correct/incorrect indicators. |
| `k` | `int` | Number of sampled candidate generations to consider per sample (e.g. 8). |

---

## Output Format

- **Console Log / Return Value**: Prints the count of evaluated samples and returns the calculated accuracy percentage (float).

---

## What It Does

1. **Majority Voting (`maj@k`)**: Groups the first $k$ predictions for a question, identifies the most common answer (majority), and checks if it matches the ground truth.
2. **Reward Model Selection (`rm@k`)**: Selects the prediction with the highest reward model score among the first $k$ candidates and checks if it matches the ground truth.
3. **Symbolic Answer Grouping**: Performs answer clustering using mathematical equivalence tests with timeouts, ensuring that expressions like `x + 1` and `1 + x` are grouped together during voting.

---

## How It Does It (In Detail)

### 1. Answer Grouping Heuristics (`group_pred`)

Groups predictions to identify the majority vote:
- **String Matching Mode (`use_symbol=False`)**: Optionally strips units and whitespaces using `strip_string()`, counts frequencies using `collections.Counter`, and selects the most frequent string.
- **Symbolic Grouping Mode (`use_symbol=True`)**:
  - Compares candidate answers pairwise using `math_equal_timeout()` (a wrapper around `math_equal` with a 5-second timeout).
  - If two answers are mathematically equivalent, they are placed in the same group.
  - Returns the group structure and the representative answer for the largest group.

### 2. Majority Voting Evaluation (`eval_maj_k_metrics`)

- Loads generated samples from the JSONL file.
- For each sample, groups the first $k$ predictions and selects the majority answer.
- Increments the correct count if the selected majority answer is correct.

### 3. Reward Model Scoring Evaluation (`eval_rm_k_metrics`)

- For each sample, retrieves the first $k$ reward model scores (from `pred_score`).
- Finds the index of the highest score: `max_index = rm_score.index(max(rm_score))`.
- Checks if the prediction at `max_index` is correct (from the `score` boolean list).
- Computes and prints the accuracy percentage.
