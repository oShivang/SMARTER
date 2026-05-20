# `sal/search/beam_search.py` — Iterative Beam Search with PRM Scoring

## What Is This File?

This file implements **standard iterative Beam Search** for math reasoning. Unlike Best-of-N which generates complete solutions and then ranks them, Beam Search grows solutions one step at a time and **prunes low-scoring beams at each step** using the PRM. Only the top-scoring partial solutions survive into the next iteration.

---

## Input Format

### `beam_search(examples, config, llm, prm)`
| Argument | Type | Description |
|---|---|---|
| `examples` | `dict` | Dataset batch with `"problem"` key → `list[str]` |
| `config` | `Config` | Pipeline configuration (`n`, `beam_width`, `num_iterations`, `threshold`, etc.) |
| `llm` | `vllm.LLM` | The policy/generation model |
| `prm` | `PRM` | Process Reward Model for per-step scoring |

---

## Output Format

```python
{
  "completions": [["sol1", "sol2", ...], ...],  # N completions per problem
  "pred": ["best_solution_per_problem", ...],
}
```

The best prediction is the completion with the highest PRM aggregate score among all completed beams.

---

## What It Does

Beam search maintains `N` active **beams** (partial solutions) at each step. At each iteration:
1. Each active beam generates one more reasoning step (stopping at `\n\n`).
2. The PRM scores each beam's full accumulated solution.
3. Beams are pruned: only the top `N / beam_width` beams survive.
4. This repeats until beams reach EOS or `num_iterations` is exhausted.

---

## How It Does It (In Detail)

### Initialization
`N` Beam objects are created per problem, all starting with `current_text = ""`. The `Beam` dataclass tracks accumulated text, step history, PRM scores, and completion status.

### Main Iteration Loop

For each of `config.num_iterations` iterations:

#### 1. Active Beam Management
- First iteration: all non-pruned beams are active.
- Subsequent iterations: only non-pruned beams from the previous round.
- If active count < `config.n`, beams are duplicated to always maintain `N` candidates (ensuring diversity).

#### 2. Step Generation
- Conversations are built: `build_conv(prompt, current_text, system_prompt)`.
- The tokenizer formats them with `apply_chat_template()`, using `continue_final_message=True` after the first iteration (so the assistant continues its partial response).
- `generate_k_steps()` calls vLLM with `stop=["\n\n"]` to generate exactly one reasoning step per beam.
- On the final iteration, the stop string is removed so the model can finish the answer completely.

#### 3. Beam Update
For each beam:
- `current_text += new_step` (step is appended)
- `history.append(new_step)` (for token length tracking)
- If token history exceeds 2048, the beam is marked `completed` (length limit hit)
- If stop reason is `"EOS"` or `"length"`, beam is marked `completed`
- Completed beams move to `completed_beams` list

#### 4. PRM Scoring
`prm.score(prompts, completions)` scores each beam's full current text.
`aggregate_scores()` reduces step-level scores to a scalar per beam using `config.agg_strategy`.

#### 5. Pruning
Active (non-completed) beams are sorted by their aggregate scores. Only the top `N // beam_width` beams survive; the rest are marked `pruned = True`.

```python
top_indices = np.argsort(agg_scores.flatten())[-(config.n // config.beam_width):]
for idx, beam in enumerate(active_beams):
    if idx not in top_indices:
        beam.pruned = True
```

#### 6. Early Stopping
The loop exits early if:
- All active beams are completed, OR
- At least `N` beams have been completed (when `sort_completed=False`)

### Post-Processing
- If `sort_completed=True`, completed beams are re-sorted by PRM score and top-N are kept.
- If fewer than N beams completed, the list is padded by repeating beams.
- Results are grouped by problem prompt using `defaultdict`.
- The final `pred` is the completion with the highest aggregate PRM score.

---

## Key Design Notes

- `config.beam_width` (called `m` in papers) determines the branching factor. With `n=16` and `beam_width=4`, 4 beams survive each pruning round.
- `config.filter_duplicates=True` deduplicate beams with identical `current_text` before pruning, encouraging diversity.
- **Beam search is forced to `search_batch_size=1`** (one problem at a time) because the iterative nature makes true batching complex. This is acknowledged as a TODO in the code.
- Unlike Best-of-N, beam search's compute budget grows with `num_iterations × N`, not just `N`.
