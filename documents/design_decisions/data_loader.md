# Design Decision: Handling BoolQ Separately from Core Math Loaders

## Overview

In the SMARTER architecture, the dataset ingestion layer is divided between:
1. **Core Math Evaluation Loader** (`src/evaluation/data_loader.py`)
2. **Search Pipeline Loader** (`src/sal/utils/data.py`)

While standard mathematical datasets (GSM8K, MATH, SVAMP, etc.) are processed within the core evaluation loaders, **BoolQ** is handled specifically and separately. This document details the engineering reasons behind this design.

---

## Rationale for Separate Handling

### 1. Different Task Modality (Reading Comprehension vs. Math QA)
Unlike math datasets that present self-contained questions, BoolQ is a reading comprehension dataset. It requires pairing a long **passage** with a **question** to form the prompt. 
* To make it compatible with the pipeline, the prompt must be formatted to guide the model's reasoning and ensure it outputs a parseable prediction format.
* The search loader maps the passage and question into a customized prompt that instructs the model to reason and end its text with standard LaTeX boxes (e.g., `\boxed{yes}` or `\boxed{no}`).

### 2. Missing Test Split Labels on Hugging Face
The official test split for `google/boolq` on Hugging Face does not provide target labels (to prevent test-set leakage). 
* To conduct local benchmarking and compute validation accuracy, the search loader intercepts `"test"` dataset requests and redirects them to the `"validation"` split (which contains the required ground-truth labels).
* The core evaluation loader is designed for standard split configurations and does not contain redirection rules.

### 3. Separation of Search-Time Preprocessing
During speculative decoding/search execution, the pipeline runs iterative rollouts. 
* Having BoolQ handled in `src/sal/utils/data.py` separates prompt formatting, label mapping, and validation routing from the standard offline evaluation loading, keeping the evaluation script cleaner and focused solely on processing pre-cached mathematical files.
