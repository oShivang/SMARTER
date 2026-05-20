# `evaluation/grader.py` — Math Grading Decision Engine

## What Is This File?

This file serves as the core math evaluation and grading engine. It contains deep heuristics (based on Hendrycks' MATH, DeepSeek-Math, and other popular benchmarks) to judge mathematical equivalence between predicted math solutions and ground truth labels.

---

## Input Format

### `math_equal(prediction, reference, include_percentage, is_close, timeout)`
| Argument | Type | Description |
|---|---|---|
| `prediction` | `str \| float \| int` | The model's extracted prediction string |
| `reference` | `str \| float \| int` | The dataset's official label/target |
| `include_percentage`| `bool` | Whether to divide values by 100 if they contain '%' |
| `is_close` | `bool` | Whether to perform float proximity checks |
| `timeout` | `float` | Max grading time per check in seconds (default: 1.0) |

---

## Output Format

- **`math_equal()`**: Returns a `bool` (`True` if the prediction is judged correct relative to the reference, `False` otherwise).

---

## What It Does

1. **Multi-Type Coercion**: Handles numerical, symbolic, multiple-choice, matrix-based, and set-based answers.
2. **Robust String Matching**: Normalizes variables, decimal formats, systems of equations, lists, and coordinate tuples.
3. **Symbolic Fallback via Separate Process**: Performs sympy expression parsing and matrix math operations, running verification in an isolated background thread/process to handle timeouts safely.

---

## How It Does It (In Detail)

### 1. Digits & Multiple-Choice Cleaners

- **`parse_digits`**: Converts standard text fractions, percentages, or comma-separated numbers into raw floating-point values.
- **`choice_answer_clean`**: Sanitizes standard multiple-choice brackets (e.g. `(A)` to `A`).

### 2. Equivalency Checklist in `math_equal`

The function runs the following checks in order:
1. **Exact String Match**: Checks if inputs match exactly.
2. **Boolean / String Identifiers**: Compares lowercase values directly (e.g. `true`, `false`, `yes`, `no`).
3. **Floating-point Proximity**: If both parse to numbers, it checks proximity with a relative tolerance of `1e-4`.
4. **Symbolic Verification (`symbolic_equal`)**:
   - Parses expressions to SymPy objects using `latex2sympy` or `sympy.parsing`.
   - Simplifies the difference `a - b`.
   - If the difference is zero, they are equivalent.
5. **System of Equations/Matrices**: Compares lists/sets of solutions and matrices (e.g. checks shape and element equivalence).

### 3. Hanging Guard (`call_with_timeout`)

SymPy simplification can hit complex exponential terms or integrals that hang or run out of memory. To prevent this, `grader.py` spawns symbolic checks inside a `multiprocessing.Process`. It communicates back using a `multiprocessing.Queue` and terminates the worker process if it exceeds the `timeout` parameter, returning `False`.
