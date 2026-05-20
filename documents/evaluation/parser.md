# `evaluation/parser.py` — Math Output Parsing and Sanitization

## What Is This File?

This file contains string clean-up rules, regex patterns, and parsers to extract final mathematical answers from raw LLM generations. It handles LaTeX formatting anomalies, multiple-choice options, boxed equations, and dataset-specific ground truth formats.

---

## Input Format

- **Raw Model Response**: Arbitrary text strings from text generation outputs.
- **Example Dictionaries**: Sample dictionaries containing raw answer keys.

---

## Output Format

- **Sanitized Outputs**: Clean mathematical expression strings (free of units, formatting clutter, spaces, and currency symbols).

---

## What It Does

1. **LaTeX Normalization (`strip_string`)**: Sanitizes mathematical notation:
   - Removes formatting tags like `\left`, `\right`, percentage signs, dollar signs, and units.
   - Cleans fractions, square roots, and decimals.
   - Standardizes matrix notation (`bmatrix` to `pmatrix`).
2. **Targeted Answer Extraction**:
   - **`find_box()`**: Extracts text enclosed within LaTeX `\boxed{...}` tags by tracking brace depth.
   - **`choice_answer_clean()`**: Extracts single character options (A-E) for multiple-choice questions.
   - **`extract_theoremqa_answer()`**: Parses boolean values (True/False) or LaTeX representations of equations.
   - **`extract_answer()`**: Extracts mathematical expressions or falls back to retrieving the last number in the text.
3. **Ground Truth Standardization (`parse_ground_truth`)**: Parses and standardizes ground-truth labels based on the target dataset.

---

## How It Does It (In Detail)

### 1. LaTeX Normalization Rules (`strip_string`)

- Replaces linebreaks, trailing dots, and spacing tokens (`\!`, `\ `).
- Replaces division expressions like `\tfrac` or `\dfrac` with `\frac`.
- Normalizes relational signs (e.g. `\neq` to `\ne`, `\leq` to `\le`, `\geq` to `\ge`).
- Removes non-mathematical unit strings (e.g. `meter`, `kg`, `hour`, `degrees`, etc.) using a predefined word list.
- Strips parentheses, brackets, or braces if they enclose alphanumeric strings.
- Converts written number words (e.g., "three") to digits using the `word2number` library.

### 2. Answer Extraction Logic

- **Brace Tracking (`find_box`)**: Tracks brace nesting depth: incrementing at `{` and decrementing at `}`. When depth returns to 0, it extracts the enclosed content.
- **Choice Extraction (`choice_answer_clean`)**: Splits text at triggers like "answer is" or "choice is", extracts letters (A-E), and removes trailing punctuation.
- **Last Number Fallback**: If no standard patterns match, it searches for decimal formats (`-?\d*\.?\d+`) and returns the last match.

### 3. Dataset-Specific Ground Truth Parsing (`parse_ground_truth`)

Applies dataset-specific rules to parse ground-truth labels:
- **MATH**: Extracts answers from CoT solutions.
- **GSM8K**: Splits answers at the `####` delimiter.
- **BoolQ**: Converts values to boolean strings (`yes`/`no` or `True`/`False`).
- **TabMWP**: Converts fractions, percentages, and comma-separated numbers to floats.
