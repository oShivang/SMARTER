# Design Decision: Reusing the Core Math Grader for BoolQ

## Overview

In the SMARTER evaluation suite, all model predictions are graded against ground-truth references using the main math equivalence engine (**`src/evaluation/grader.py`**). Rather than developing or importing a separate verification module for the binary reading comprehension dataset (**BoolQ**), the pipeline routes BoolQ predictions through `grader.py`.

---

## Rationale for Reusing the Math Grader

### 1. String Equality Fast-Path
In both `grader.py` and the standalone helper `math_utils.py`, the core comparison functions begin with an exact string match check:
* **In `grader.py` (`math_equal`)**:
  ```python
  if str(prediction.strip().lower()) == str(reference.strip().lower()):
      return True
  ```
* **In `math_utils.py` (`compare_ans`)**:
  ```python
  if ans_p_str.replace(" ", "") == ans_l_str.replace(" ", ""):
      return True
  ```
Because BoolQ answers parse into binary text strings (`"yes"` or `"no"`), their evaluations trigger these fast-path matches and return a correct boolean result immediately. This makes separate logic or a custom verification engine for BoolQ redundant.

### 2. Bypassing Complex Symbolic Compilations
Since the exact string matching matches the binary inputs first, the evaluation process never falls through to:
* **Symbolic computations** (using `SymPy` or `latex2sympy`).
* **Numerical float calculations** (using tolerance intervals).
This guarantees that evaluating BoolQ has zero computational or library-level overhead.

### 3. Unified Runner Interface
By reusing `math_equal_process`, the offline evaluation script (**`evaluate.py`**) can use a single, clean loop structure to verify all benchmarks (GSM8K, MATH, and BoolQ) without needing conditional branches or custom evaluation functions.
