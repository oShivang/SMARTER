# Threshold Calibration (`scripts/calibrate.py`)

This document describes the **threshold calibration pipeline** of the SMARTER repository. Calibration is the phase where the optimal confidence strategy and threshold are selected for each dataset (e.g., BoolQ, GSM8K, MATH-500) before initiating full-scale evaluation.

---

## The Core Concept

When using confidence-based speculative decoding (SMART), we must determine when a step's logprob/entropy score is "poor" enough to trigger an LLM correction. 
- A threshold that is **too high** triggers the LLM too frequently, increasing inference cost.
- A threshold that is **too low** allows incorrect draft steps to pass, reducing accuracy.

The calibration script evaluates a subset of the dataset (typically 100 samples) across a grid of thresholds and methods, plotting the resulting cost-vs-accuracy curves. It then programmatically selects the "best" point on the curve.

---

## Selection Methods

The pipeline supports **four distinct mathematical methods** to select the optimal threshold and confidence strategy from the cost-vs-accuracy curves.

### 1. Greedy Selection (`greedy`)
Selects the configuration that yields the highest absolute accuracy. If multiple configurations tie for the highest accuracy, it breaks the tie by choosing the one with the lowest cost.
* **Pros**: Guarantees the maximum possible accuracy on the calibration set.
* **Cons**: Can drift to very high costs (right side of the curve) for marginal accuracy gains.

### 2. Kneedle Algorithm (`kneedle`)
An implementation of the *Satopaa et al. (2011)* Knee-Detection algorithm.
* **Logic**: It draws a straight line from the start of the curve $(Cost_{min}, Acc_{min})$ to the end of the curve $(Cost_{max}, Acc_{max})$. It then calculates the perpendicular distance from each point on the curve to this line and selects the point with the **maximum distance**.
* **Pros**: Mathematically captures the visual "elbow" of the curve where the rate of return begins to drop off.
* **Cons**: Requires a well-behaved curved trajectory.

### 3. Slope Threshold (`slope`)
Checks the rate of change between consecutive points on the sorted cost-accuracy curve ($\Delta\text{Accuracy} / \Delta\text{Cost}$).
* **Logic**: Iterates from left to right and selects the last point where the slope remains greater than or equal to a user-defined threshold $\theta$ (configured via `--elbow_slope_theta`).
* **Pros**: Gives strict control over the "exchange rate" (e.g., "I am willing to pay at least $0.5\%$ cost for $1\%$ accuracy").

### 4. Utility Maximization (`utility`) [Default]
Evaluates a simple decision-theoretic utility function for each point:
$$\text{Utility} = \text{Accuracy} - \lambda \times \text{Cost}$$
Where $\lambda$ (configured via `--elbow_utility_lambda`) penalizes the cost of LLM queries.
* **Pros**: Simple, highly customizable, and models resource constraints directly.
* **Cons**: Requires tuning the $\lambda$ parameter depending on how expensive LLM calls are.

---

## Configuration & Usage

The selection method and parameters can be passed directly as command-line arguments to the calibration script or the main shell runner.

### Command Line Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--elbow_method` | `str` | `"utility"` | Method to select the optimal threshold: `["greedy", "kneedle", "slope", "utility"]` |
| `--elbow_slope_theta` | `float` | `0.5` | $\theta$ parameter for the `slope` method |
| `--elbow_utility_lambda` | `float` | `0.5` | $\lambda$ parameter (cost penalty) for the `utility` method |

### Integration in Benchmark Suite (`scripts/run_final_benchmark.sh`)

The master benchmark script triggers the calibration pipeline first and saves the winners to a JSON summary file:

```bash
# 1. Run Calibration using the utility maximization method
python scripts/calibrate.py \
    --config recipes/qwen_calibrate.yaml \
    --num_calibration_samples 100 \
    --num_full_samples 0 \
    --datasets "boolq,gsm8k,math500" \
    --elbow_method "utility"
```

The script then automatically reads the winning method and threshold from `outputs/calibration/calibration_summary.json` and launches the full dataset runs under those optimal settings.
