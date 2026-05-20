# `sal/utils/parser.py` — YAML + CLI Argument Parser

## What Is This File?

This file provides `H4ArgumentParser`, an extended argument parser that can load configuration from a **YAML file**, and then **override individual fields via CLI flags**. This is the standard entry-point mechanism for all scripts in the pipeline.

---

## Input Format

The parser is typically used in scripts as follows:
```python
parser = H4ArgumentParser(Config)
config = parser.parse()
```

It supports three invocation modes (detected automatically from `sys.argv`):

| Mode | `sys.argv` Pattern | Behavior |
|---|---|---|
| YAML only | `script.py config.yaml` | Loads all config from YAML |
| YAML + overrides | `script.py config.yaml --n=32 --temperature=0.9` | Loads YAML, then applies CLI overrides |
| CLI only | `script.py --approach=beam_search --n=16 ...` | Pure argparse mode |

**YAML file format** (example):
```yaml
approach: best_of_n
model_path: meta-llama/Llama-3.2-1B-Instruct
n: 16
temperature: 0.8
dataset_name: HuggingFaceH4/MATH-500
output_dir: ./output
```

**CLI override format:**
```bash
python run_inference.py config.yaml --n=32 --seed=123
```

---

## Output Format

`parser.parse()` returns a **`Config` dataclass instance** (or a tuple of dataclass instances if multiple dataclass types were registered).

---

## What It Does

`H4ArgumentParser` extends HuggingFace's `HfArgumentParser` with two capabilities:

1. **`parse_yaml_and_args(yaml_arg, other_args)`** — Loads a YAML file into the registered dataclass fields, then applies additional CLI key=value overrides on top.
2. **`parse()`** — A smart dispatcher that auto-detects the invocation mode and calls the right parsing method.

---

## How It Does It (In Detail)

### `parse_yaml_and_args(yaml_arg, other_args)`

1. Calls `self.parse_yaml_file(yaml_arg)` to deserialize the YAML into a dataclass-like list.
2. Converts `other_args` from `["--key=val", ...]` format into a dict `{"key": "val"}`.
3. Iterates over all declared fields of the dataclass. For each CLI override key that matches a field name:
   - Casts the string value to the field's declared type (`int`, `float`, `bool`, `List[str]`).
   - Bool handling is explicit: `"true"/"True"` → `True`, `"None"/"none"` → `None`, anything else → `False`. (This avoids Python's default behavior where `bool("False")` returns `True`.)
4. Detects **duplicate arguments** (same key provided twice) and raises `ValueError`.
5. Detects **unrecognized arguments** (keys that don't match any dataclass field) and raises `ValueError`.

### `parse()`

```python
if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
    # YAML-only mode
elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
    # YAML + CLI overrides mode
else:
    # Pure CLI mode (HfArgumentParser default)
```

If only one dataclass was registered and `parse()` returns a single-element list, it automatically unwraps it and returns the dataclass directly (not a tuple).

---

## Key Design Notes

- **`bool` parsing is explicit and important:** Standard `argparse` and `bool()` conversions would incorrectly treat `"False"` as `True`. The custom string comparison ensures correct boolean CLI overrides.
- **Unrecognized argument detection** prevents silent misconfiguration — if a CLI key doesn't exist in the dataclass, the script fails loudly instead of silently ignoring the override.
- This pattern (YAML base config + CLI overrides) is a common research workflow: the YAML captures the experiment's default configuration, while CLI overrides enable quick ablations without editing files.
- The `H4ArgumentParser` name and design follow the HuggingFace Alignment Handbook convention, from which this project is derived.
