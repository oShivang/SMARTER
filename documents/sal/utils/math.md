# `sal/utils/math.py` — Mathematical Answer Processing and Voting

## What Is This File?

This file provides the **mathematical answer processing layer** for the pipeline. It converts raw model completions into canonical symbolic forms so that mathematically equivalent answers (e.g., `\frac{1}{2}` and `0.5`) can be recognized as the same answer. It also implements the three voting strategies used to pick the best prediction from a set of N completions.

---

## Input Format

Most functions take dataset batch dicts (`x`) or raw answer strings. Key inputs:

### `memoized_canonical_form(expression, timeout_seconds=3)`
| Argument | Type | Description |
|---|---|---|
| `expression` | `str` | A LaTeX math expression string (e.g., `"\\frac{3}{4}"`) |
| `timeout_seconds` | `int` | Max seconds before falling back to `strip_string()` |

### `extract_completion_answers(x, n)`
| Argument | Type | Description |
|---|---|---|
| `x` | `dict` | Dataset row with `"completions"` or `"completions@n"` key |
| `n` | `int \| None` | Subset size; `None` uses all completions |

### `compute_weighted_pred(x, n)` / `compute_maj_pred(x, n)` / `compute_naive_pred(x, n)`
| Argument | Type | Description |
|---|---|---|
| `x` | `dict` | Row with `"preds@n"` and `"agg_scores@n"` |
| `n` | `int` | Subset size |

---

## Output Format

| Function | Return Type | Description |
|---|---|---|
| `memoized_canonical_form()` | `str` | Simplified LaTeX form or stripped fallback |
| `extract_completion_answers()` | `dict` with `"preds"` or `"preds@n"` | List of extracted answer strings |
| `compute_weighted_pred()` | `dict` with `"pred_weighted@n"` | `\boxed{answer}` format |
| `compute_maj_pred()` | `dict` with `"pred_maj@n"` | `\boxed{answer}` format |
| `compute_naive_pred()` | `dict` with `"pred_naive@n"` | `\boxed{answer}` format |
| `pass_at_k()` | `float` | Unbiased pass@k probability estimate |
| `compute_pass_at_k()` | `dict` with `"pass@k"` | Keyed result |

---

## What It Does

### Canonical Form Computation (`memoized_canonical_form`)
Converts a LaTeX expression to a **sympy-simplified canonical form** so that equivalent math expressions map to the same string. Uses a **shared multiprocessing cache** (`manager.dict()`) so results computed in one process are reused by all worker processes (important for parallel `dataset.map()` calls).

**Fallback chain:**
1. Check the shared cache → return immediately if found.
2. Parse with `latex2sympy(expression)` → simplify with `sympy.simplify()` → convert back to LaTeX.
3. On `TimeoutException` (via SIGALRM) → fall back to `strip_string()` (Qwen math parser's string normalizer).
4. On any other exception → fall back to `strip_string()`.

### Answer Extraction (`extract_completion_answers`)
Calls `extract_answer(completion, "math")` (from `sal/utils/qwen_math_parser.py`) on each completion to pull the boxed final answer. Returns a list of extracted strings.

### Voting Strategies

#### `compute_weighted_pred()` (Score-Weighted Voting)
Groups answers by their canonical form, sums the PRM aggregate scores for each group, and picks the group with the highest total score. This rewards not just the most common answer but the one the model was most *confident* about across its completions.

#### `compute_maj_pred()` (Majority Vote)
Groups answers by canonical form, counts how many completions fall in each group, and picks the largest group. Ties are broken by first occurrence.

#### `compute_naive_pred()` (Best Single Completion)
Picks the completion with the single highest PRM aggregate score and returns its answer. No grouping — just pure score maximization.

### Pass@K Metric (`pass_at_k`, `compute_pass_at_k`)
Implements the **numerically stable pass@k** estimator from OpenAI's Codex paper:
```python
pass@k = 1 - ∏_{i=n-c+1}^{n} (1 - k/i)
```
where `n` = total samples, `c` = correct samples, `k` = k cutoff. This is used to measure coverage probability rather than average accuracy.

### Difficulty Level Assignment (`compute_level`)
Maps a problem's metric value (e.g., `pass@1`) to a difficulty level 1–5 using pre-computed quintiles. Level 1 = easiest (highest metric), Level 5 = hardest.

---

## How It Does It (In Detail)

### Shared Cache Implementation
```python
if multiprocessing.current_process().name == 'MainProcess':
    manager = Manager()
    shared_cache = manager.dict()  # process-safe dict
else:
    shared_cache = {}  # local fallback in worker processes
```
The `Manager().dict()` is backed by a manager process and accessible across forks. Worker processes use a local dict as a fallback (they can't access the manager dict directly after forking in some configurations).

### Timeout with SIGALRM
`signal.alarm(timeout_seconds)` sets a Unix alarm. If `latex2sympy()` takes longer than `timeout_seconds` (default 3s), `SIGALRM` fires and raises `TimeoutException`, triggering the `strip_string()` fallback. The alarm is always cancelled in the `finally` block.

### `silence_stderr()` Context Manager
`latex2sympy` prints verbose warnings to `stderr`. The file wraps the import in a custom context manager that redirects `sys.stderr` to `/dev/null` during the import, suppressing these noise messages without affecting other parts of the program.

---

## Key Design Notes

- **Canonical form caching** is critical for performance: `memoized_canonical_form()` may be called thousands of times during evaluation; without caching, sympy simplification would dominate runtime.
- The **SIGALRM timeout** is Unix-only (not available on Windows), but this project targets Linux GPU servers where this is fine.
- The `strip_string()` fallback ensures that even if sympy fails (malformed LaTeX, complex expressions), the answer comparison still works on a best-effort basis.
