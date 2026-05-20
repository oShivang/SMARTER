# Design Decision: Status of `math_utils.py`

## Overview

The file **`src/evaluation/math_utils.py`** is a standalone mathematical helper library. It contains a complete LaTeX cleaning, parsing, and SymPy comparison engine. 

---

## Current Status

* **Status**: **Not actively imported or used** by the main evaluation runner (`math_eval.py`), orchestrator (`evaluate.py`), or active grading dispatcher (`grader.py`).
* **Active Alternative**: The pipeline utilizes the parser and grading logic embedded directly inside **`grader.py`** (`math_equal`, `symbolic_equal`, etc.) for all dataset evaluations.

---

## Rationale: Why It Exists and Why It Is Not Used

### 1. Unified Integration in `grader.py`
The active runner is standardized around `grader.py` because its math grading logic is fully self-contained. It contains custom choice cleaning and numeric validation functions designed specifically for Hendricks MATH and GSM8K, mapped directly into the multi-process wrapper `math_equal_process`.

### 2. Legacy / Alternative Utility
`math_utils.py` exists as a **legacy utility module** or **reference implementation**. During development and research iterations, multiple parsing strategies (such as strict LaTeX symbols vs. soft string normalization) are often coded. `math_utils.py` preserves a robust SymPy comparison engine (`compare_ans`) and additional utility functions (like `vote()` for majority voting and `percentage_to_fraction()`) that can be used for custom or future test scripts.

---

## How It Handles BoolQ (If Utilized)

If a developer decides to swap the active grader to use `math_utils.py`'s `compare_ans()` function, it would evaluate the binary **BoolQ** dataset successfully:
* At the beginning of `compare_ans()`, it checks for exact string matches:
  ```python
  if ans_p_str.replace(" ", "") == ans_l_str.replace(" ", ""):
      return True
  ```
* Because BoolQ answers are binary `"yes"` or `"no"`, they trigger this fast-path immediately.
* This ensures that even under the alternative utility, BoolQ evaluations completely bypass SymPy compiler dependencies and float operations.
