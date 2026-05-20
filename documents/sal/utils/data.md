# `sal/utils/data.py` — Dataset Loading and Output Saving

## What Is This File?

This file handles the two I/O boundaries of the pipeline:
1. **`get_dataset()`** — loads and normalizes the benchmark dataset from HuggingFace Hub or local CSV/JSONL files.
2. **`save_dataset()`** — persists the output dataset (with completions and predictions) either to the HuggingFace Hub or to a local `.jsonl` file.

---

## Input Format

### `get_dataset(config)`
| Argument | Type | Description |
|---|---|---|
| `config` | `Config` | Pipeline config providing `dataset_name`, `dataset_split`, `dataset_start`, `dataset_end`, `num_samples` |

Supported dataset names and their sources:

| `config.dataset_name` | Source | Special Handling |
|---|---|---|
| `"sampled_math500"` | Local CSV file | Loaded via `pd.read_csv()` |
| `"prm_math500"` | Local JSONL file (`train.jsonl`) | Loaded via `pd.read_json()` |
| `"gsm8k"` / `"openai/gsm8k"` | HuggingFace Hub | Column `"question"` renamed to `"problem"` |
| `"math500"` / `"HuggingFaceH4/MATH-500"` | HuggingFace Hub | Used as-is |
| `"boolq"` | HuggingFace Hub | `"validation"` split used for `"test"`; problem reformatted from passage+question |
| Any other string | HuggingFace Hub | `load_dataset()` with fallback to `trust_remote_code=True` |

### `save_dataset(dataset, config)`
| Argument | Type | Description |
|---|---|---|
| `dataset` | `datasets.Dataset` | The dataset after search and scoring |
| `config` | `Config` | Used to determine output path, approach name, and hub settings |

---

## Output Format

### `get_dataset()`
Returns a HuggingFace `Dataset` object. All datasets are normalized so that:
- The math problem text is in the `"problem"` column.
- The correct answer is in the `"answer"` column.
- Optional slicing via `dataset_start`/`dataset_end` or `num_samples` is applied.

### `save_dataset()`
Saves to one of two destinations:
- **HuggingFace Hub:** A dataset branch named by the revision string (encodes all hyperparameters).
- **Local disk:** A `.jsonl` file at:
  ```
  {output_dir}/{folder_name}/{approach_fn}_completions_T-{temperature}--top_p-{top_p}--n-{n}--m-{beam_width}--iters-{num_iterations}--look-{lookahead}--seed-{seed}--agg_strategy--{agg_strategy}_threshold-{threshold}_{num_samples}.jsonl
  ```

---

## What It Does

### `get_dataset()`

Provides a **unified interface** for all benchmark datasets. Each dataset has its own quirks:
- **BoolQ** requires reformatting: the `"passage"` and `"question"` fields are merged into a single `"problem"` string, and the boolean `"answer"` is converted to `"yes"`/`"no"`.
- **GSM8K** uses `"question"` instead of `"problem"`, so a column rename is applied.
- Slicing logic allows running on a subset (e.g., chunks for distributed evaluation).

### `save_dataset()`

The **output path** is constructed deterministically from hyperparameters to make results easily reproducible and identifiable:
- `folder_name` distinguishes between `smart_prm`, `smart_conf`, `base_prm`, `base_conf` depending on `config.draft_model_path` and `config.score_method`.
- `approach_fn` maps `beam_width == 1` to `"best_of_n"` and otherwise uses `config.approach` (e.g., `"beam_search"`).

---

## How It Does It (In Detail)

### Hub Push (with Retry)
If `config.push_to_hub=True`, `save_dataset()` tries up to **20 times** to push the dataset to the Hub, with a 5-second sleep between attempts. This is needed because concurrent pushes from parallel jobs can be rejected by the Hub API.

It also creates a branch for the dataset revision from the **initial commit** (not `main`):
```python
initial_commit = list_repo_commits(config.hub_dataset_id)[-1]
create_branch(repo_id=..., branch=config.revision, revision=initial_commit.commit_id)
```
This ensures each experiment gets a clean, isolated branch that doesn't inherit data from previous runs.

### Local Save (with Auto-mkdir)
`Path(config.output_dir).mkdir(parents=True, exist_ok=True)` ensures the full output directory tree is created before saving.

---

## Key Design Notes

- The **filename encodes all hyperparameters** — this makes it possible to identify any output file and know exactly what configuration produced it, without reading the file contents.
- The BoolQ reformatting includes a **prompt template** designed to elicit `\boxed{yes}` / `\boxed{no}` responses compatible with the math parser.
- The fallback `trust_remote_code=True` for unknown datasets is a safety net for community Hub datasets that require custom loading scripts.
