# `evaluation/examples.py` — Few-Shot Demonstrations Database

## What Is This File?

This file serves as a static database containing few-shot demonstration examples (question-answer pairs) for various datasets and reasoning styles.

---

## Input Format

- **`get_examples()`**: Requires no arguments.

---

## Output Format

- **Demonstration Dictionary**: Returns a dictionary mapping dataset identifiers (e.g. `gsm8k`, `math`, `aqua`, `sat_math`, `mmlu_mathematics`, etc.) to a list of tuples: `list[tuple[str, str]]`. Each tuple represents a few-shot demonstration: `(Question, Answer/Chain-of-Thought)`.

---

## What It Does

This file acts as a centralized repository for prompt templates, ensuring that the same reference demonstrations are used across different evaluation runs. It contains:
- Chain-of-Thought (CoT) examples.
- Program-Aided Language (PAL) examples containing executable Python functions.
- Tool-integrated (ToRA) examples containing interleaved Python code blocks and text reasoning.
- Subject-specific examples for MMLU (STEM, physics, chemistry, biology, computer science) and Chinese Gaokao exams.

---

## How It Does It (In Detail)

The file defines `get_examples()`, which initializes a dictionary and populates it with lists of tuples for each dataset:

### 1. GSM8K Variations
- **`gsm8k`**: Contains 8 standard arithmetic word problem demonstrations with natural language solutions.
- **`gsm8k-pal`**: Contains Python solution definitions (`def solution():`) that return computed integers.
- **`gsm8k-tora`**: Contains interleaved code blocks and natural language outputs showing execution results:
  ````markdown
  ```python
  def money_left():
      ...
  ```
  ```output
  8
  ```
  Olivia has $\\boxed{8}$ dollars left.
  ````

### 2. MATH Variations
- **`math`**: Includes 5 complex algebra, geometry, and calculus problems with detailed LaTeX steps.
- **`math_pal`**: Provides programmatic solutions to geometry and vector math problems using `sympy` and `numpy`.
- **`math-tora`**: Combines symbolic solver code (using `sympy` interval solvers) with text descriptions.

### 3. Multiple Choice and Subject Datasets
- **MMLU / Aqua / Sat Math**: Provides multiple-choice questions with explanation paths that conclude with the selected option (e.g., `The answer is (D).`).
- **Gaokao**: Contains Chinese college entrance examination questions with step-by-step reasoning steps.
- **BoolQ**: Contains 5 reading comprehension few-shot examples with detailed rationale and yes/no answers wrapped in standard LaTeX `\boxed{}` tags.
