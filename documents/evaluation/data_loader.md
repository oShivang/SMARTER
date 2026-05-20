# `evaluation/data_loader.py` — Dataset Loading and Preprocessing

## What Is This File?

This file manages dataset ingestion. It loads math and reasoning benchmarks from local files or downloads them from the Hugging Face Hub, performs preprocessing, and caches the processed samples to local JSONL files for offline evaluations.

---

## Input Format

### `load_data(data_name, split, data_dir)`
| Argument | Type | Description |
|---|---|---|
| `data_name` | `str` | Name of the dataset (e.g. `gsm8k`, `math`, `svamp`, `asdiv`, `mawps`, `mmlu_stem`, `carp_en`) |
| `split` | `str` | Name of the dataset split (e.g. `test`, `train`, `validation`) |
| `data_dir` | `str` | Local directory path to load from or cache datasets to (default: `./data`) |

---

## Output Format

- **List of Dictionaries**: Returns a list of processed dictionaries representing dataset examples (e.g. `list[dict[str, Any]]`).

---

## What It Does

1. **Caching Layer**: Checks if the target dataset has already been processed and cached under `data_dir/data_name/split.jsonl`. If it exists, it loads it immediately.
2. **Unified Fetching**: If cache misses, it fetches data from Hugging Face:
   - `math` dataset from `competition_math`
   - `gsm8k` from `gsm8k`
   - `svamp` from `ChilleD/SVAMP`
   - `asdiv` from `EleutherAI/asdiv`
   - `mmlu_stem` from `hails/mmlu_no_train` (filters only STEM topics)
   - `mawps` by concatenating sub-task files (`singleeq`, `singleop`, `addsub`, `multiarith`)
3. **Common Formatting**: Lowercases dictionary keys, generates unique ID indices (`idx`), and sorts the examples by index.

---

## How It Does It (In Detail)

### 1. Cache-First Logic

- Computes a dataset file path: `{data_dir}/{data_name}/{split}.jsonl`.
- If the path exists, it loads the file line-by-line using `load_jsonl()` and returns it.

### 2. Dataset Normalization & STEM Filtering

If the local cache does not exist, the module downloads the dataset and normalizes it:
- **ASDIV filtering**: Filters out entries with multi-answer coordinates (containing `;`).
- **MMLU STEM Extraction & Column Renaming**: 
  - **What is MMLU?**: Massive Multitask Language Understanding. It is a large test suite covering many subjects (humanities, social sciences, history, etc.).
  - **What is STEM?**: **S**cience, **T**echnology, **E**ngineering, and **M**athematics. The loader filters the MMLU dataset to keep only the 19 STEM subjects (e.g. astronomy, biology, computer security, machine learning, algebra).
  - **Why and How Columns are Renamed**:
    - In the original MMLU dataset, the subject classification is stored in a column named `"subject"`.
    - The SMARTER pipeline expects categories to be stored in a column named `"type"` for consistency across all datasets.
    - Thus, `dataset.rename_column("subject", "type")` is called to standardize the schema. This allows easy filtering and consistent processing during accuracy reporting.
- **Key Standardization**: Processes all dictionary keys to lowercase (`lower_keys()`) for consistent attribute access (e.g. mapping `Question` to `question`, `Answer` to `answer`).
- **Index Generation**: If `idx` is missing from the schemas, it enumerates the list and maps index integers to each example.
- **JSONL Caching**: Creates parent directories and writes the formatted dataset to disk as a JSONL file to speed up subsequent runs.
