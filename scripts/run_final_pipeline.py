import sys
import os
import gc
import json
import argparse
import copy
import math
import subprocess
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import Dataset

os.environ["PYTHONWARNINGS"] = "ignore"
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("src/evaluation"))

from sal.config import Config
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score, STRATEGY_MAP, needs_correction, calculate_confidence_score, aggregate_scores
from sal.search import (
    best_of_n,
    best_of_n_conf,
    smart_best_of_n,
    smart_best_of_n_conf,
    beam_search,
    beam_search_conf,
    smart_beam_search,
    smart_beam_search_conf
)
from sal.search.utils import build_conv
from evaluation.evaluate import evaluate

APPROACHES = {
    "beam_search": beam_search,
    "beam_search_smart": smart_beam_search,
    "beam_search_conf": beam_search_conf,
    "beam_search_smart_conf": smart_beam_search_conf,
    "best_of_n": best_of_n,
    "best_of_n_smart": smart_best_of_n,
    "best_of_n_conf": best_of_n_conf,
    "best_of_n_smart_conf": smart_best_of_n_conf,
}

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def unload_model(model):
    if model is None:
        return
    if hasattr(model, "llm_engine") and hasattr(model.llm_engine, "engine_core"):
        try:
            model.llm_engine.engine_core.shutdown()
        except Exception:
            pass
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    del model
    clear_gpu_memory()

def get_gpu_info():
    try:
        import psutil
        total_sys_ram = psutil.virtual_memory().total / (1024**3)
    except Exception:
        total_sys_ram = 32.0
    if not torch.cuda.is_available():
        return 0, (0, 0), 0, total_sys_ram
    num_gpus = torch.cuda.device_count()
    device_capability = torch.cuda.get_device_capability()
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) # GB
    return num_gpus, device_capability, total_memory, total_sys_ram

def extract_bool(text):
    if not text: return ""
    text = str(text).lower().strip()
    import re
    
    # 1. Check for boxed answers first
    boxed = re.findall(r'\\boxed\{(.*?)\}', text)
    if boxed:
        ans = boxed[-1].lower().strip()
        if "yes" in ans or "true" in ans: return "yes"
        if "no" in ans or "false" in ans: return "no"

    # 2. Check for "The answer is X"
    match = re.search(r'the answer is[:\s]+(yes|no|true|false)', text)
    if match:
        ans = match.group(1)
        return "yes" if ans in ["yes", "true"] else "no"

    # 3. Fallback to word boundaries at the very end
    words = re.findall(r'\b(yes|no|true|false)\b', text)
    if words:
        ans = words[-1]
        return "yes" if ans in ["yes", "true"] else "no"
        
    return ""

def load_slm(config, num_gpus, device_capability, dtype, gpu_util, enforce_eager=False):
    from vllm import LLM
    print(f"Loading SLM (vLLM) from: {config.draft_model_path if config.draft_model_path else config.model_path}", flush=True)
    slm = LLM(
        model=config.draft_model_path if config.draft_model_path else config.model_path,
        gpu_memory_utilization=gpu_util,
        enforce_eager=enforce_eager,
        enable_prefix_caching=True,
        tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
        max_model_len=8192,
        dtype=dtype
    )
    print("SLM Loaded successfully.", flush=True)
    return slm

def load_hf_model(model_path, num_gpus, torch_dtype, should_load_4bit):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model (Transformers) from: {model_path}", flush=True)
    device_map = "cuda:0" if num_gpus == 1 else "auto"
    if should_load_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model Loaded successfully.", flush=True)
    return model, tokenizer

def generate_hf_batch(problems, config, model, tokenizer, batch_size=32):
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    do_sample = config.temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": config.max_tokens,
    }
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = config.temperature
        gen_kwargs["top_p"] = config.top_p
        
    outputs = []
    for i in range(0, len(problems), batch_size):
        batch_probs = problems[i:i+batch_size]
        batch_prompts = []
        for problem in batch_probs:
            conv = build_conv(problem, "", config.system_prompt)
            templated = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(templated)
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)
            
        input_len = inputs.input_ids.shape[1]
        decoded = tokenizer.batch_decode(out_ids[:, input_len:], skip_special_tokens=True)
        outputs.extend(decoded)
        
    tokenizer.padding_side = orig_padding_side
    return outputs

def run_baseline_hf(dataset, config, model, tokenizer, batch_size=32):
    problems = dataset["problem"]
    print(f"Running baseline search (HF) with batch size {batch_size}...", flush=True)
    decoded_outputs = generate_hf_batch(problems, config, model, tokenizer, batch_size)
    
    completions_list = [[text] for text in decoded_outputs]
    preds_list = decoded_outputs
        
    def add_cols(example, idx):
        example["completions"] = completions_list[idx]
        example["pred"] = preds_list[idx]
        return example
        
    return dataset.map(add_cols, with_indices=True, load_from_cache_file=False)

def evaluate_and_save_dataset(dataset, config, is_baseline=True):
    if is_baseline and (config.approach == "best_of_n" or config.approach == "beam_search"):
        subsets = [2**i for i in range(int(math.log2(config.n)) + 1)]
        if config.n not in subsets:
            subsets.append(config.n)
        subsets = sorted(list(set(subsets)))
        keys = []
        for n in subsets:
            keys.extend([f"pred_weighted@{n}", f"pred_maj@{n}", f"pred_naive@{n}"])
    else:
        keys = ["pred"]

    if "boolq" in config.dataset_name.lower():
        correct_counts = {k: 0 for k in keys}
        samples_list = list(dataset)
        for sample in samples_list:
            gt_val = sample["answer"]
            gt = gt_val if isinstance(gt_val, str) else ("yes" if gt_val else "no")
            gt = gt.lower().strip()
            for k in keys:
                pred = extract_bool(sample.get(k, ""))
                if pred == gt:
                    correct_counts[k] += 1
        
        acc = {k: round((v / len(samples_list)) * 100, 1) for k, v in correct_counts.items()}
        result = {"acc": acc, "num_samples": len(samples_list)}
        
        # Add 'correct' column so that fallback evaluation counts are accurate
        correct_flags = []
        for sample in samples_list:
            gt_val = sample["answer"]
            gt = gt_val if isinstance(gt_val, str) else ("yes" if gt_val else "no")
            gt = gt.lower().strip()
            pred = extract_bool(sample.get("pred", ""))
            correct_flags.append(pred == gt)
        dataset = dataset.add_column("correct", correct_flags)
    else:
        if "math" in config.dataset_name.lower():
            data_type = "math"
        elif "gsm8k" in config.dataset_name.lower():
            data_type = "gsm8k"
        else:
            data_type = config.dataset_name
        dataset, result = evaluate(data_name=data_type, prompt_type=None, samples=dataset, pred_keys=keys)
        
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
            
        # If dataset evaluation doesn't add 'correct' column, make sure it is added
        if "correct" not in dataset.column_names:
            if "correct_completions" in dataset.column_names:
                correct_flags = [s["correct_completions"][0] for s in dataset]
                dataset = dataset.add_column("correct", correct_flags)
            else:
                dataset = dataset.add_column("correct", [True] * len(dataset))

    # Save final dataset (exclude pred_completions)
    save_ds = Dataset.from_list([{k: v for k, v in dict(sample).items() if k != 'pred_completions'} for sample in dataset])
    save_dataset(save_ds, config)
    
    # Write companion _metrics.json
    output_base = "output" if os.path.exists("output") else "outputs"
    if config.output_dir is None:
        config.output_dir = f"{output_base}/{config.model_path}"
    
    if config.draft_model_path is not None:
        folder_name = "smart_prm" if config.score_method == 'prm' else "smart_conf"
    else:
        folder_name = "base_prm" if config.score_method == 'prm' else "base_conf"
            
    approach_fn_name = "best_of_n" if config.beam_width == 1 else config.approach
        
    if config.dataset_start is not None and config.dataset_end is not None:
        filename = f"{config.output_dir}/{folder_name}/{approach_fn_name}_completions_T-{config.temperature}--top_p-{config.top_p}--n-{config.n}--m-{config.beam_width}--iters-{config.num_iterations}--look-{config.lookahead}--seed-{config.seed}--agg_strategy--{config.agg_strategy}_{config.num_samples}_datasplit_{config.dataset_start}-{config.dataset_end}.jsonl"
    else:
        filename = f"{config.output_dir}/{folder_name}/{approach_fn_name}_completions_T-{config.temperature}--top_p-{config.top_p}--n-{config.n}--m-{config.beam_width}--iters-{config.num_iterations}--look-{config.lookahead}--seed-{config.seed}--agg_strategy--{config.agg_strategy}_threshold-{config.threshold}_{config.num_samples}.jsonl"
        
    metrics_filename = filename.replace(".jsonl", "_metrics.json")
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)
    with open(metrics_filename, "w") as f:
        json.dump(result, f, indent=4)
        
    print(f"Saved metrics to {metrics_filename}", flush=True)
    return result

def run_calibration_elbow(problems, problem_results, slm_correct_list, llm_fixable, ds_name, elbow_method, elbow_slope_theta, elbow_utility_lambda):
    print(f"Running Threshold Sweep (Elbow) for {ds_name}...", flush=True)
    method_stats = {}
    best_ds_config = {"method": None, "threshold": 0.0, "acc": 0.0, "cost": 0.0}

    plt.figure(figsize=(10, 6))

    for method in STRATEGY_MAP.keys():
        all_bottleneck_scores = []
        for res in problem_results:
            step_scores = [step["scores"][method] for step in res["steps"]]
            if not step_scores:
                all_bottleneck_scores.append(None)
                continue
            
            # Find the first bad step
            found = False
            for step in res["steps"]:
                val = step["scores"][method]
                # Default high-threshold check to find first bad step simulation
                if needs_correction(val, 0.9 if method != "entropy" else 0.1, method):
                    all_bottleneck_scores.append(val)
                    found = True
                    break
            if not found:
                all_bottleneck_scores.append(step_scores[0])

        valid_scores = [s for s in all_bottleneck_scores if s is not None]
        if not valid_scores:
            continue

        thresholds = np.linspace(min(valid_scores), max(valid_scores), 20)
        accs, costs = [], []
        for tau in thresholds:
            correct = 0
            problem_token_ratios = []
            for i in range(len(problems)):
                if all_bottleneck_scores[i] is None:
                    problem_token_ratios.append(0.0)
                    if slm_correct_list[i]: correct += 1
                    continue

                triggered = needs_correction(all_bottleneck_scores[i], tau, method)

                if triggered:
                    trigger_step_idx = -1
                    for s_idx, step in enumerate(problem_results[i]["steps"]):
                        if needs_correction(step["scores"][method], tau, method):
                            trigger_step_idx = s_idx
                            break
                    
                    step_tokens = problem_results[i]["steps"][trigger_step_idx]["token_count"]
                    total_tokens = sum(s["token_count"] for s in problem_results[i]["steps"])
                    usage_ratio = (step_tokens / total_tokens) if total_tokens > 0 else 1.0
                    problem_token_ratios.append(usage_ratio)

                    if llm_fixable[i]: correct += 1
                else:
                    problem_token_ratios.append(0.0)
                    if slm_correct_list[i]: correct += 1

            accuracy = (correct / len(problems)) * 100
            cost = np.mean(problem_token_ratios) * 100
            accs.append(accuracy)
            costs.append(cost)
        
        sort_idx = np.argsort(costs)
        sorted_costs = np.array(costs)[sort_idx]
        sorted_accs = np.array(accs)[sort_idx]
        
        best_idx = 0
        if elbow_method == "kneedle":
            p1 = np.array([sorted_costs[0], sorted_accs[0]])
            p2 = np.array([sorted_costs[-1], sorted_accs[-1]])
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            if line_len > 0:
                line_unit_vec = line_vec / line_len
                distances = []
                for i in range(len(sorted_costs)):
                    p3 = np.array([sorted_costs[i], sorted_accs[i]])
                    p3_p1 = p3 - p1
                    dist = np.linalg.norm(p3_p1 - np.dot(p3_p1, line_unit_vec) * line_unit_vec)
                    distances.append(dist)
                best_idx_sorted = np.argmax(distances)
                best_idx = sort_idx[best_idx_sorted]
        elif elbow_method == "slope":
            best_idx_sorted = 0
            for i in range(1, len(sorted_costs)):
                d_cost = sorted_costs[i] - sorted_costs[i-1]
                d_acc = sorted_accs[i] - sorted_accs[i-1]
                if d_cost > 0:
                    slope = d_acc / d_cost
                    if slope >= elbow_slope_theta:
                        best_idx_sorted = i
                    else:
                        break
            best_idx = sort_idx[best_idx_sorted]
        elif elbow_method == "utility":
            utilities = np.array(accs) - elbow_utility_lambda * np.array(costs)
            best_idx = np.argmax(utilities)
        else: # greedy
            for idx in range(len(accs)):
                if accs[idx] > accs[best_idx]:
                    best_idx = idx
                elif accs[idx] == accs[best_idx] and costs[idx] < costs[best_idx]:
                    best_idx = idx
        
        if accs[best_idx] > best_ds_config["acc"] or (accs[best_idx] == best_ds_config["acc"] and costs[best_idx] < best_ds_config["cost"]):
            best_ds_config = {"method": method, "threshold": float(thresholds[best_idx]), "acc": float(accs[best_idx]), "cost": float(costs[best_idx])}
        method_stats[method] = {"thresholds": thresholds.tolist(), "accuracies": accs, "costs": costs}
        plt.plot(costs, accs, marker='o', label=method)
        print(f"  [{method:12}] best: acc={accs[best_idx]:.1f}% at cost={costs[best_idx]:.2f}% | threshold={thresholds[best_idx]:.6f}", flush=True)

    plt.xlabel("Cost (% LLM Calls)"); plt.ylabel("Accuracy (%)"); plt.title(f"Elbow Graph - {ds_name}"); plt.legend(); plt.grid(True)
    os.makedirs("outputs/calibration", exist_ok=True)
    plt.savefig(f"outputs/calibration/elbow_{ds_name}.png")
    plt.close()
    
    if best_ds_config["method"] is None:
        best_ds_config = {"method": "probs_mean", "threshold": 0.0, "acc": 0.0, "cost": 0.0}
        
    print(f"\nWINNER FOR DATASET: {ds_name} | method={best_ds_config['method']} | threshold={best_ds_config['threshold']:.6f} | acc={best_ds_config['acc']:.2f}% | cost={best_ds_config['cost']:.2f}%", flush=True)
    return {"best_config": best_ds_config, "all_methods": method_stats}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_calibrate", type=str, default="recipes/qwen_calibrate.yaml")
    parser.add_argument("--config_test", type=str, default="recipes/qwen_test.yaml")
    parser.add_argument("--num_calibration_samples", type=int, default=300)
    parser.add_argument("--num_full_samples", type=int, default=-1)
    parser.add_argument("--datasets", type=str, default="gsm8k,math500,boolq")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--elbow_method", type=str, default="utility", choices=["greedy", "kneedle", "slope", "utility"])
    parser.add_argument("--elbow_slope_theta", type=float, default=0.5)
    parser.add_argument("--elbow_utility_lambda", type=float, default=0.5)
    
    args, unknown = parser.parse_known_args()
    
    h4_parser_calib = H4ArgumentParser(Config)
    config_calib = h4_parser_calib.parse_yaml_and_args(args.config_calibrate, unknown)[0]
    
    h4_parser_test = H4ArgumentParser(Config)
    config_test = h4_parser_test.parse_yaml_and_args(args.config_test, unknown)[0]
    
    # Configure Calibration samples
    config_calib.num_samples = args.num_calibration_samples
    
    # Configure Full/Evaluation samples
    if args.num_full_samples == -1:
        config_test.num_samples = None # None means all samples of datasets
    else:
        config_test.num_samples = args.num_full_samples
        
    dataset_list = args.datasets.split(",")
    print(f"Target Datasets: {dataset_list}", flush=True)
    
    # Hardware detection
    num_gpus, device_capability, total_vram, total_sys_ram = get_gpu_info()
    dtype = "bfloat16" if device_capability[0] >= 8 else "half"
    torch_dtype = torch.bfloat16 if device_capability[0] >= 8 else torch.float16
    
    should_load_4bit = args.load_in_4bit or config_test.load_in_4bit
    if not should_load_4bit:
        if total_vram > 0 and (total_vram <= 24.0 or total_sys_ram <= 16.0):
            should_load_4bit = True
            
    use_parallel_loading = (total_vram > 24)
    gpu_util = args.gpu_memory_utilization if args.gpu_memory_utilization is not None else (0.4 if use_parallel_loading else 0.5)
    hf_batch_size = 64 if use_parallel_loading else 16
    
    print("="*60, flush=True)
    print(f"SYSTEM PROFILE (FULL PIPELINE):", flush=True)
    print(f"  GPUs Available : {num_gpus} | Capability: {device_capability}", flush=True)
    print(f"  Total VRAM     : {total_vram:.2f} GB | System RAM: {total_sys_ram:.2f} GB", flush=True)
    print(f"  4-Bit Precision: {'ENABLED' if should_load_4bit else 'DISABLED'}", flush=True)
    print(f"  HF Batch Size  : {hf_batch_size}", flush=True)
    print("="*60, flush=True)
    
    # ----------------------------------------------------
    # PHASE 1 & 2: calibration generation on 300 samples
    # ----------------------------------------------------
    print("\n==================================================")
    print("   PHASE 1: RUNNING CALIBRATION SWEEP (300 SAMPLES)")
    print("==================================================")
    
    # 1. Load SLM in vLLM for step generation
    slm_vllm = load_slm(config_calib, num_gpus, device_capability, dtype, gpu_util, enforce_eager=False)
    clear_gpu_memory()
    
    slm_calibration_results = {}
    for ds_name in dataset_list:
        print(f"\n--- Generating SLM Calibration Steps for {ds_name} ---", flush=True)
        calib_config_ds = copy.deepcopy(config_calib)
        calib_config_ds.dataset_name = ds_name
        calib_dataset = get_dataset(calib_config_ds)
        problems = calib_dataset["problem"]
        
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=calib_config_ds.temperature,
            max_tokens=calib_config_ds.max_tokens,
            top_p=calib_config_ds.top_p,
            stop=["\n\n"],
            n=1,
            logprobs=10
        )
        
        problem_results = []
        for prob_idx, problem in enumerate(problems):
            if (prob_idx + 1) % 20 == 0 or prob_idx == 0 or (prob_idx + 1) == len(problems):
                print(f"  [SLM Calibration] Generating sample {prob_idx + 1}/{len(problems)}...", flush=True)
            current_text = ""
            steps_info = []
            for i in range(calib_config_ds.num_iterations):
                conv = build_conv(problem, current_text, calib_config_ds.system_prompt)
                templated = slm_vllm.get_tokenizer().apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=(i==0), continue_final_message=(i>0)
                )
                outputs = slm_vllm.generate([templated], sampling_params, use_tqdm=False)
                output = outputs[0].outputs[0]
                conf_metrics = calculate_confidence_score(output.logprobs)
                scores_dict = {name: conf_metrics[idx] for name, idx in STRATEGY_MAP.items()}
                steps_info.append({"scores": scores_dict, "text": output.text, "token_count": len(output.token_ids)})
                current_text += output.text
                if output.stop_reason == "EOS" or output.text == "": 
                    break
            problem_results.append({"slm_final_text": current_text, "steps": steps_info})
            
        slm_calibration_results[ds_name] = {
            "problems": problems,
            "problem_results": problem_results,
            "answers": calib_dataset["answer"]
        }
        
    print("Unloading SLM vLLM to make room for LLM...", flush=True)
    unload_model(slm_vllm)
    clear_gpu_memory()
    
    # 2. Load LLM in HF for fixing bottleneck steps in calibration
    llm_hf, llm_tokenizer = load_hf_model(config_calib.model_path, num_gpus, torch_dtype, should_load_4bit)
    
    calibration_summary = {}
    for ds_name in dataset_list:
        print(f"\n--- Generating LLM Calibration fixes for {ds_name} ---", flush=True)
        res_data = slm_calibration_results[ds_name]
        problems = res_data["problems"]
        problem_results = res_data["problem_results"]
        answers = res_data["answers"]
        
        # Evaluate SLM correctness on calibration
        slm_correct_list = []
        if ds_name == "boolq":
            for idx, res in enumerate(problem_results):
                gt_val = answers[idx]
                gt = gt_val if isinstance(gt_val, str) else ("yes" if gt_val else "no")
                slm_correct_list.append(gt.lower().strip() == extract_bool(res["slm_final_text"]))
        else:
            eval_name = "math" if "math" in ds_name.lower() else ("gsm8k" if "gsm8k" in ds_name.lower() else ds_name)
            temp_samples = [
                {
                    "problem": p,
                    "pred": r["slm_final_text"],
                    "solution": answers[idx],
                    "answer": answers[idx],
                    "completions": [r["slm_final_text"]]
                }
                for idx, (p, r) in enumerate(zip(problems, problem_results))
            ]
            slm_eval_samples, _ = evaluate(data_name=eval_name, prompt_type="cot", samples=temp_samples, pred_keys=["pred"])
            slm_correct_list = [s["correct_completions"][0] for s in slm_eval_samples]
            
        # Parallel prompt formulation for LLM fixes
        batch_prompts = []
        for i, problem in enumerate(problems):
            # Check correctness of final output directly
            conv = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": problem}
            ]
            templated = llm_tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(templated)
            
        print("Running LLM batched generation for calibration fixes...", flush=True)
        llm_tokenizer.padding_side = "left"
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
            
        llm_outputs = []
        gen_kwargs = {"max_new_tokens": config_calib.max_tokens}
        for idx_b in range(0, len(batch_prompts), hf_batch_size):
            sub_batch = batch_prompts[idx_b:idx_b+hf_batch_size]
            inputs = llm_tokenizer(sub_batch, return_tensors="pt", padding=True).to(llm_hf.device)
            with torch.no_grad():
                out_ids = llm_hf.generate(**inputs, **gen_kwargs)
            input_len = inputs.input_ids.shape[1]
            decoded = llm_tokenizer.batch_decode(out_ids[:, input_len:], skip_special_tokens=True)
            llm_outputs.extend(decoded)
            
        if ds_name == "boolq":
            llm_fixable = []
            for prob_idx, problem in enumerate(problems):
                llm_text = llm_outputs[prob_idx]
                gt = answers[prob_idx]
                gt_str = "yes" if gt else "no"
                llm_fixable.append(gt_str == extract_bool(llm_text))
        else:
            eval_name = "math" if "math" in ds_name.lower() else ("gsm8k" if "gsm8k" in ds_name.lower() else ds_name)
            temp_samples = [
                {
                    "problem": p,
                    "pred": out,
                    "solution": ans,
                    "answer": ans,
                    "completions": [out]
                }
                for p, out, ans in zip(problems, llm_outputs, answers)
            ]
            llm_eval_samples, _ = evaluate(data_name=eval_name, prompt_type="cot", samples=temp_samples, pred_keys=["pred"])
            llm_fixable = [s["correct_completions"][0] for s in llm_eval_samples]
                
        # Perform threshold elbow sweep
        ds_summary = run_calibration_elbow(
            problems, problem_results, slm_correct_list, llm_fixable,
            ds_name, args.elbow_method, args.elbow_slope_theta, args.elbow_utility_lambda
        )
        calibration_summary[ds_name] = ds_summary
        
    # Save calibration summary
    calib_json = "outputs/calibration/calibration_summary.json"
    os.makedirs(os.path.dirname(calib_json), exist_ok=True)
    with open(calib_json, "w") as f:
        json.dump(calibration_summary, f, indent=4)
    print(f"Calibration finished. Summary saved to {calib_json}.", flush=True)
    
    # ----------------------------------------------------
    # PHASE 3: SLM Baseline (Full Dataset, HF Batched)
    # ----------------------------------------------------
    print("\n==================================================")
    print("   PHASE 2: RUNNING SLM BASELINE ON FULL DATASETS")
    print("==================================================")
    
    # Since we loaded LLM earlier, we unload it to load SLM in HF
    print("Unloading LLM to prepare for SLM baseline loading...", flush=True)
    del llm_hf, llm_tokenizer
    clear_gpu_memory()
    
    slm_model_path = config_test.draft_model_path if config_test.draft_model_path else config_test.model_path
    slm_hf, slm_tokenizer = load_hf_model(slm_model_path, num_gpus, torch_dtype, should_load_4bit)
    
    for ds_name in dataset_list:
        print(f"\n--- Running SLM Baseline (Full) on {ds_name} ---", flush=True)
        baseline_config_ds = copy.deepcopy(config_test)
        baseline_config_ds.dataset_name = ds_name
        baseline_config_ds.smart_search = False
        baseline_config_ds.score_method = "conf"
        baseline_config_ds.n = 1
        baseline_config_ds.num_iterations = 1
        
        # Override num_samples for boolq standard evaluation length if specified
        if args.num_full_samples == -1 and ds_name == "boolq":
            baseline_config_ds.num_samples = 1635
            
        full_dataset = get_dataset(baseline_config_ds)
        
        # Map baseline using HF
        baseline_ds = run_baseline_hf(full_dataset, baseline_config_ds, slm_hf, slm_tokenizer, hf_batch_size)
        
        n_completions = baseline_config_ds.n
        baseline_ds = baseline_ds.map(lambda x: {"scores": [[1.0] for _ in range(n_completions)]}, load_from_cache_file=False)
        baseline_ds = score(baseline_ds, baseline_config_ds)
        
        baseline_config_ds.output_dir = f"outputs/{ds_name}/slm_baseline"
        baseline_config_ds.draft_model_path = baseline_config_ds.model_path
        evaluate_and_save_dataset(baseline_ds, baseline_config_ds, is_baseline=True)
        
    print("Unloading SLM HF model...", flush=True)
    del slm_hf, slm_tokenizer
    clear_gpu_memory()
    
    # ----------------------------------------------------
    # PHASE 4: LLM Baseline (Full Dataset, HF Batched)
    # ----------------------------------------------------
    print("\n==================================================")
    print("   PHASE 3: RUNNING LLM BASELINE ON FULL DATASETS")
    print("==================================================")
    
    llm_hf, llm_tokenizer = load_hf_model(config_test.model_path, num_gpus, torch_dtype, should_load_4bit)
    
    for ds_name in dataset_list:
        print(f"\n--- Running LLM Baseline (Full) on {ds_name} ---", flush=True)
        baseline_config_ds = copy.deepcopy(config_test)
        baseline_config_ds.dataset_name = ds_name
        baseline_config_ds.smart_search = False
        baseline_config_ds.score_method = "conf"
        baseline_config_ds.n = 1
        baseline_config_ds.num_iterations = 1
        
        # Override num_samples for boolq standard evaluation length if specified
        if args.num_full_samples == -1 and ds_name == "boolq":
            baseline_config_ds.num_samples = 1635
            
        full_dataset = get_dataset(baseline_config_ds)
        
        # Map baseline using HF
        baseline_ds = run_baseline_hf(full_dataset, baseline_config_ds, llm_hf, llm_tokenizer, hf_batch_size)
        
        n_completions = baseline_config_ds.n
        baseline_ds = baseline_ds.map(lambda x: {"scores": [[1.0] for _ in range(n_completions)]}, load_from_cache_file=False)
        baseline_ds = score(baseline_ds, baseline_config_ds)
        
        baseline_config_ds.output_dir = f"outputs/{ds_name}/llm_baseline"
        baseline_config_ds.draft_model_path = baseline_config_ds.model_path
        evaluate_and_save_dataset(baseline_ds, baseline_config_ds, is_baseline=True)
        
    # ----------------------------------------------------
    # PHASE 5: SMART speculative run (Full Dataset, vLLM for SLM)
    # ----------------------------------------------------
    print("\n==================================================")
    print("   PHASE 4: RUNNING SMART ON FULL DATASETS")
    print("==================================================")
    
    # Load SLM in vLLM alongside LLM HF Causal model
    clear_gpu_memory()
    slm_vllm = load_slm(config_test, num_gpus, device_capability, dtype, gpu_util, enforce_eager=False)
    
    for ds_name in dataset_list:
        print(f"\n--- Running SMART Speculative Decoding on {ds_name} ---", flush=True)
        
        if ds_name in calibration_summary:
            bc = calibration_summary[ds_name].get("best_config", {})
            best_method = bc.get("method", "probs_mean")
            best_threshold = bc.get("threshold", 0.8)
        else:
            print(f"No calibration winner for {ds_name} found in calibration_summary.json. Using defaults.", flush=True)
            best_method = "probs_mean"
            best_threshold = 0.8
            
        print(f"Selected Threshold settings for SMART: Strategy={best_method}, Threshold={best_threshold}", flush=True)
        
        smart_config_ds = copy.deepcopy(config_test)
        smart_config_ds.dataset_name = ds_name
        smart_config_ds.smart_search = True
        smart_config_ds.score_method = "conf"
        smart_config_ds.conf_strategy = best_method
        smart_config_ds.threshold = best_threshold
        
        if args.num_full_samples == -1 and ds_name == "boolq":
            smart_config_ds.num_samples = 1635
            
        full_dataset = get_dataset(smart_config_ds)
        
        approach_suffix = "_smart" if smart_config_ds.smart_search else ""
        approach_suffix += "_conf" if smart_config_ds.score_method == 'conf' else ""
        approach_name = smart_config_ds.approach + approach_suffix
        approach_fn = APPROACHES[approach_name]
        
        # Map SMART search using SLM in vLLM and LLM in HF Causal model
        smart_ds = full_dataset.map(
            approach_fn,
            batched=True,
            batch_size=smart_config_ds.search_batch_size,
            fn_kwargs={"config": smart_config_ds, "slm": slm_vllm, "prm": None, "llm": llm_hf},
            desc=f"Running SMART search ({ds_name})",
            load_from_cache_file=False
        )
        
        smart_config_ds.output_dir = f"outputs/{ds_name}/smart_results"
        evaluate_and_save_dataset(smart_ds, smart_config_ds, is_baseline=False)
        
    print("\nTearing down models and freeing up GPU memory...", flush=True)
    unload_model(slm_vllm)
    del llm_hf
    clear_gpu_memory()
    
    # ----------------------------------------------------
    # PHASE 6: Compile Final Report Card
    # ----------------------------------------------------
    print("\n==================================================")
    print("   PHASE 5: COMPILING FINAL REPORT CARD")
    print("==================================================")
    
    cmd = [
        sys.executable,
        "scripts/generate_report.py",
        "--calibration_json", calib_json,
        "--output_dir", "outputs",
        "--smart_tag", "smart_results",
        "--slm_tag", "slm_baseline",
        "--llm_tag", "llm_baseline"
    ]
    subprocess.run(cmd)
    
    print("\n" + "="*50)
    echo_msg = "✅ BENCHMARK COMPLETE"
    print(f"| {echo_msg:^46} |")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
