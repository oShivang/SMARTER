# SMARTER Documentation

This folder mirrors the structure of `src/` and contains detailed Markdown documentation for every source file in the SMARTER pipeline.

---

## Directory Structure

```
documents/
├── README.md                          ← This file (index)
├── workflow.md                        ← Overall project workflow & architecture diagram
├── calibration.md                     ← Threshold calibration process & elbow methods
│
├── sal/
    ├── config.md                      ← Global config dataclass (all hyperparameters)
│   │
│   ├── models/
│   │   └── reward_models.md           ← PRM wrappers (MathShepherd, RLHFFlow)
│   │
│   ├── search/
│   │   ├── utils.md                   ← Core data structures & generation helpers
│   │   ├── best_of_n.md               ← Best-of-N with PRM scoring (baseline)
│   │   ├── best_of_n_conf.md          ← Best-of-N with confidence scoring
│   │   ├── best_of_n_smart.md         ← SMART Best-of-N (PRM-based LLM intervention)
│   │   ├── best_of_n_smart_conf.md    ← SMART Best-of-N (confidence-based, post-hoc loop)
│   │   ├── beam_search.md             ← Iterative beam search with PRM pruning
│   │   ├── beam_search_conf.md        ← Beam search with confidence-based pruning
│   │   ├── beam_search_smart.md       ← SMART beam search (PRM-triggered LLM)
│   │   └── beam_search_smart_conf.md  ← SMART beam search (confidence-triggered LLM)
│   │
│   └── utils/
│       ├── data.md                    ← Dataset loading and output saving
│       ├── hub.md                     ← HuggingFace Hub revision utilities
│       ├── math.md                    ← Answer canonicalization and voting
│       ├── parser.md                  ← YAML + CLI argument parser
│       ├── qwen_math_parser.md        ← Math answer extraction from model text
│       └── score.md                   ← Confidence scoring, PRM aggregation, pass@k
│
└── evaluation/
    ├── evaluate.md                    ← Offline benchmark evaluation script
    ├── model_utils.md                 ← Model loading & tokenization helpers
    ├── python_executor.md             ← Sandboxed python runtime & pool executor
    ├── math_utils.md                  ← SymPy LaTeX comparison & cleaner
    ├── grader.md                      ← Core grading engine & symbolic checker
    ├── data_loader.md                 ← Local/HF dataset parser and cacher
    ├── math_eval.md                   ← Main offline evaluation runner script
    ├── trajectory.md                  ← Conversation & trace sequence extractor
    ├── rm_maj_eval.md                 ← Reward model & majority@k evaluator
    ├── parser.md                      ← Sanitizer & answer text extractor
    ├── utils.md                       ← Prompt templating & seeding helpers
    └── examples.md                    ← Few-shot examples database
│
└── design_decisions/
    ├── data_loader.md                 ← Rationale for handling BoolQ outside core loaders
    ├── grader.md                      ← Explanation of using the math grader for BoolQ
    └── math_utils.md                  ← Rationale and status of the standalone math_utils.py
```

---

## How to Navigate This Documentation

### High-Level Architecture
- Read **`workflow.md`** first to understand the end-to-end pipeline architecture, speculative decoding flow, and model integration patterns.

### Starting a Run
1. Understand the **`sal/config.md`** — all hyperparameters are defined here.
2. Use **`sal/utils/parser.md`** to see how YAML configs and CLI flags are parsed.
3. Read **`sal/utils/data.md`** to understand which datasets are supported and how they are loaded.
4. Read **`calibration.md`** to see how threshold grid searches are executed and resolved.

### Understanding the Search Algorithms
The search algorithms come in an 8-way matrix:

| | **PRM Scoring** | **Confidence Scoring** |
|---|---|---|
| **Best-of-N (no iterative pruning)** | `best_of_n.md` | `best_of_n_conf.md` |
| **Best-of-N + SMART (LLM correction)** | `best_of_n_smart.md` | `best_of_n_smart_conf.md` |
| **Beam Search (iterative pruning)** | `beam_search.md` | `beam_search_conf.md` |
| **Beam Search + SMART (LLM correction)** | `beam_search_smart.md` | `beam_search_smart_conf.md` |

All algorithms share the `Beam` data structure and generation utilities defined in **`sal/search/utils.md`**.

### Understanding Scoring
- **`sal/models/reward_models.md`** — how the PRM assigns a quality score to each reasoning step.
- **`sal/utils/score.md`** — how confidence scores are computed from token log-probs, and how step-scores are aggregated into a final selection score.

### Understanding Evaluation
The codebase has two distinct evaluation contexts: search pipeline utilities (`src/sal/`) and general offline benchmark evaluation tools (`src/evaluation/`).

#### 1. Search Pipeline Evaluators
- **`sal/utils/qwen_math_parser.md`** — parses Qwen-specific model responses to extract raw math answers.
- **`sal/utils/math.md`** — performs mathematical answer canonicalization and voting.

#### 2. Offline Benchmark Runner & Suite (`src/evaluation/`)
- **`evaluation/math_eval.md`** — the main CLI runner for executing math evaluations (supporting zero-shot CoT, PAL/PoT, and multi-turn tool interaction).
- **`evaluation/data_loader.md`** — loads benchmark datasets (GSM8K, MATH, SVAMP, ASDIV, MAWPS, MMLU STEM) and manages local JSONL caching.
- **`evaluation/parser.md`** — parses LaTeX math formats, multiple-choice options, and ground truths.
- **`evaluation/trajectory.md`** — extracts structured trace sequences (rationale vs. code vs. output) from multi-turn responses.
- **`evaluation/python_executor.md`** — executes model-generated code in a sandboxed, parallel process pool with timeouts.
- **`evaluation/math_utils.md`** — cleans expressions and compares equations using SymPy.
- **`evaluation/grader.md`** — the core grading logic checking float proximity or symbolic equality in isolated background processes.
- **`evaluation/model_utils.md`** — manages model loading, quantization, and sequence stopping criteria.
- **`evaluation/rm_maj_eval.md`** — evaluates offline RM metrics (`rm@k` and `maj@k`).
- **`evaluation/utils.md`** — defines prompt assembly templates and setup functions.
- **`evaluation/examples.md`** — the database of few-shot prompt demonstrations.
- **`evaluation/evaluate.md`** — grades parsed JSONL predictions against reference answers to compute accuracy.

### Design Decisions
- **`design_decisions/data_loader.md`** — documents the architecture rationale for processing BoolQ outside the main evaluation loading module.
- **`design_decisions/grader.md`** — explains why a separate grading module is omitted for BoolQ and how exact string matching is used.
- **`design_decisions/math_utils.md`** — documents the standalone status of `math_utils.py` and its role as reference utility code.

---

## Key Concepts Glossary

| Term | Meaning |
|---|---|
| **SLM** | Small Language Model — the fast draft generator (e.g., Llama-3.2-1B) |
| **LLM** | Large Language Model — the expensive correction oracle (e.g., Llama-3.1-70B) |
| **PRM** | Process Reward Model — scores each reasoning *step*, not just the final answer |
| **SMART** | Speculative Mediation At Reasoning Time — the LLM-correction framework |
| **Beam** | A single hypothesis (partial solution) in beam search |
| **Confidence Score** | Token log-probability-based self-assessment of model uncertainty |
| **Canonical Form** | Sympy-simplified LaTeX form for math-equivalent answer comparison |
| **agg_strategy** | How per-step PRM scores are reduced to one number (`last`, `min`, `prod`) |
| **conf_strategy** | Which confidence metric drives SMART decisions (`probs_mean`, `entropy`, etc.) |
