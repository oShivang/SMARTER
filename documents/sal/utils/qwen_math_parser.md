# `sal/utils/qwen_math_parser.py` — Math Answer Extraction and String Normalization

## What Is This File?

This file is a comprehensive **mathematical answer extraction and normalization library**, adapted from the Qwen Math evaluation codebase. It handles the messy reality of extracting final answers from long, multi-step LLM outputs — dealing with inconsistent LaTeX formatting, mixed notation, and various edge cases.

---

## Input Format

### `extract_answer(pred_str, data_name)`
| Argument | Type | Description |
|---|---|---|
| `pred_str` | `str` | Full model output text (may contain many steps, markdown, LaTeX, etc.) |
| `data_name` | `str` | Dataset name hint, e.g. `"math"`, `"gsm8k"`, `"boolq"` — controls extraction heuristics |

### `strip_string(string)`
| Argument | Type | Description |
|---|---|---|
| `string` | `str` | A LaTeX math string to normalize |

---

## Output Format

### `extract_answer()` → `str`
A cleaned, normalized answer string extracted from the completion. May be:
- A LaTeX expression: `"\\frac{1}{2}"`, `"3x+2"`, `"42"`
- A plain number: `"42"`, `"3.14"`
- A word answer: `"yes"`, `"no"`, `"True"`
- An empty string `""` if extraction fails

### `strip_string()` → `str`
A normalized LaTeX string with consistent formatting for comparison.

---

## What It Does

This file provides the **last mile of evaluation**: given the raw text output from a language model, reliably extract the final answer so it can be compared to the ground truth.

Key functions:

| Function | Purpose |
|---|---|
| `extract_answer()` | Main entry point; dispatches based on `data_name` |
| `strip_string()` | Normalizes LaTeX for canonical comparison |
| `extract_boxed_answers()` | Finds all `\boxed{...}` occurrences |
| `find_box_match()` | Locates the innermost matched `\boxed` content |
| `remove_boxed()` | Strips the `\boxed{}` wrapper from a string |
| `is_equiv()` | Checks if two expressions are mathematically equivalent |

---

## How It Does It (In Detail)

### `extract_answer()` — Extraction Logic

The function uses a **priority waterfall** of extraction strategies:

1. **Dataset-specific shortcuts:**
   - For `gsm8k`: look for `"####"` marker and take the number after it.
   - For `boolq`: scan for `yes`/`no` keywords at the end of the answer.

2. **`\boxed{}`  extraction (primary method):**
   - Calls `extract_boxed_answers()` to find all `\boxed{...}` groups.
   - Takes the **last** boxed answer (since the final answer is at the end).
   - Handles nested braces correctly via `find_box_match()`.

3. **Pattern-based fallback:**
   - Searches for phrases like `"The answer is"`, `"Therefore,"`, `"final answer:"`.
   - Tries regex patterns for LaTeX expressions and numbers.
   - Handles multiple-choice answers (A/B/C/D).
   - Falls back to extracting the last standalone number in the text.

4. **Cleaning:** Removes trailing units (e.g., `" cm"`, `" dollars"`), strips text labels.

### `strip_string()` — Normalization

Applies 20+ normalization rules to make LaTeX strings comparable:
- Remove `$` signs, `\text{}` wrappers, `\left`/`\right` size commands
- Normalize fractions: `a/b` → `\frac{a}{b}`
- Handle units: strip common math units
- Normalize spacing and Unicode characters
- Replace `%` with `\%` for consistent LaTeX
- Remove trailing zeros in decimals
- Strip `\approx`, `=`, inequality prefixes

### `is_equiv()` — Mathematical Equivalence
Compares two cleaned answer strings. Falls back to sympy simplification via `memoized_canonical_form()` from `sal/utils/math.py`.

---

## Key Design Notes

- This file is **not written from scratch** — it is adapted from the Qwen Math benchmark evaluation codebase, which in turn draws from prior math evaluation repos. The normalization rules encode years of accumulated edge cases from math benchmark evaluation.
- The `extract_boxed_answers()` function handles **nested braces** properly (e.g., `\boxed{\frac{a}{b}}`) using a character-by-character brace-counting approach, not regex — because regex cannot match nested balanced brackets.
- The extraction is deliberately **forgiving**: it tries many strategies before returning `""`, minimizing false negatives where the model gave the correct answer in an unexpected format.
