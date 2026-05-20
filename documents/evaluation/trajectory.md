# `evaluation/trajectory.py` — Interaction Trajectory & Program Extraction

## What Is This File?

This file provides utilities to parse and represent interleaved model trajectories containing natural language rationales, python code segments, and environment execution outputs.

---

## Input Format

- **Trajectory String**: A raw string representing a full agent session conversation.
- **Trajectory List**: A structured representation of the conversation as a list of dicts.

---

## Output Format

- **`text_to_trajectory()`**: Returns a `list[dict[str, str]]` where each dict contains keys `role` (`"rationale"`, `"program"`, or `"output"`) and `content`.
- **`trajectory_to_text()`**: Returns a formatted string (with ```python and ```output blocks).
- **`extract_program()`**: Returns a single concatenated `str` of code blocks.

---

## What It Does

1. **State Partitioning**: Parses raw completion strings into structured dialogue trees based on code fencing selectors (```` ```python ```` and ```` ```output ````).
2. **Program Filtering**: Evaluates code execution logs to strip out broken imports, print commands, or segments that resulted in errors.
3. **Trajectory Serialization**: Serializes structured lists of turns back to prompt string formats.

---

## How It Does It (In Detail)

### 1. Parsing Logics (`text_to_trajectory`)

- Iterates through the generation text line by line.
- Identifies block starts/ends:
  - ```` ```python ```` transitions the state to `"program"`.
  - The closing ```` ``` ```` transitions the state to `"output"`.
  - ```` ```output ```` captures execution stdout strings.
- Aggregates content line-by-line and appends dialogue dicts to the trajectory list when transitions occur.

### 2. Extracting Code Blocks (`extract_program`)

- Traverses the structured trajectory.
- If a program block executed successfully (verified using `is_execution_success` on the corresponding output block), it adds it to the output script.
- If a program block failed, the function ignores the logic but retains its import lines (e.g. `import sympy` or `from math import *`) to avoid breaking subsequent blocks.
- Strips debugging print statements from early steps to prevent duplicated stdout noise, keeping only the final output.
