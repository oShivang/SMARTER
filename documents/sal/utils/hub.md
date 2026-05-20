# `sal/utils/hub.py` — HuggingFace Hub Dataset Revision Utilities

## What Is This File?

This file contains a single utility function for interacting with the HuggingFace Hub. It is used by `Config.__post_init__()` to check whether an experiment's output has already been pushed to the Hub before starting a potentially long inference run.

---

## Input Format

### `get_dataset_revisions(dataset_id)`
| Argument | Type | Description |
|---|---|---|
| `dataset_id` | `str` | A HuggingFace dataset repository ID, e.g. `"username/my-dataset"` |

---

## Output Format

Returns a `list[str]` — the names of all non-`main` branches (revisions) currently existing in the dataset repository. Returns an empty list `[]` if the repository does not exist.

---

## What It Does

`get_dataset_revisions()` checks if a given HuggingFace dataset repository has a specific revision (branch) already. This enables **early exit** in `Config.__post_init__()`:

```python
# In config.py
if not self.overwrite_hub_revision and self.revision in revisions:
    exit()  # Skip re-running an experiment that's already done
```

---

## How It Does It

1. Calls `repo_exists(dataset_id, repo_type="dataset")` to check if the repository exists on the Hub. Returns `[]` immediately if not.
2. Calls `list_repo_refs(dataset_id, repo_type="dataset")` to fetch all branches (Git refs) from the dataset repository.
3. Filters out the `"main"` branch (which is the default and not an experiment revision).
4. Returns a list of branch names.

```python
refs = list_repo_refs(dataset_id, repo_type="dataset")
return [ref.name for ref in refs.branches if ref.name != "main"]
```

---

## Key Design Notes

- This function is a **guard** against redundant computation. When running many experiments (different seeds, temperature settings, dataset splits), it's easy to accidentally re-run something already done. This function enables the pipeline to detect and skip completed runs automatically.
- The `huggingface_hub` library handles authentication automatically via the `HF_TOKEN` environment variable or cached login credentials.
- The function is kept intentionally minimal — a pure utility with no side effects.
