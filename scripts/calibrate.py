import sys
import os
os.environ["PYTHONWARNINGS"] = "ignore"
sys.path.append(os.path.abspath("src"))

import warnings
warnings.filterwarnings("ignore")

import logging
import argparse
import numpy as np
import torch
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import gc

from sal.config import Config
from sal.utils.data import get_dataset
from sal.utils.score import calculate_confidence_score, STRATEGY_MAP, needs_correction
from sal.search.utils import build_conv
from sal.models.reward_models import load_prm

sys.path.append(os.path.abspath("src/evaluation"))
from evaluation.evaluate import evaluate
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
import datasets
datasets.utils.logging.set_verbosity_warning()
import transformers
transformers.logging.set_verbosity_warning()
logging.getLogger("datasets.fingerprint").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

def clear_gpu_memory():
    gc.collect()
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

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
    
    # 1. Check for boxed answers first
    import re
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

def calibrate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_calibration_samples", type=int, default=50)
    parser.add_argument("--num_full_samples", type=int, default=None)
    parser.add_argument("--datasets", type=str, default="math500,gsm8k,boolq")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    args, unknown = parser.parse_known_args()

    # Load config
    from sal.utils.parser import H4ArgumentParser
    h4_parser = H4ArgumentParser(Config)
    config = h4_parser.parse_yaml_and_args(args.config, unknown)[0]
    
    dataset_list = args.datasets.split(",")
    os.makedirs("outputs/calibration", exist_ok=True)
    all_dataset_results = {}

    model_path = str(config.model_path)
    draft_model_path = str(config.draft_model_path) if config.draft_model_path else model_path
    
    # Adaptive Memory Strategy
    num_gpus, device_capability, total_vram, total_sys_ram = get_gpu_info()
    dtype = "bfloat16" if device_capability[0] >= 8 else "half"
    torch_dtype = torch.bfloat16 if device_capability[0] >= 8 else torch.float16
    
    # Auto-detect if we should load in 4-bit dynamically based on hardware limits
    should_load_4bit = args.load_in_4bit
    if not should_load_4bit:
        if total_vram > 0 and (total_vram <= 24.0 or total_sys_ram <= 16.0):
            should_load_4bit = True
            
    # Parallel mode needs more head-room; Sequential can be aggressive
    use_parallel_loading = total_vram > 24
    gpu_util = args.gpu_memory_utilization if args.gpu_memory_utilization is not None else (0.4 if use_parallel_loading else 0.5)
    
    logger.info(f"GPU Detected: {total_vram:.1f}GB VRAM | System RAM: {total_sys_ram:.1f}GB. Strategy: {'Parallel' if use_parallel_loading else 'Sequential'} (Util: {gpu_util})")
    logger.info(f"Dynamic 4-bit precision loading: {'ENABLED' if should_load_4bit else 'DISABLED'}")

    for ds_name in dataset_list:
        logger.info(f"\n{'='*20}\nProcessing Dataset: {ds_name}\n{'='*20}")
        config.dataset_name = ds_name
        config.num_samples = args.num_calibration_samples
        dataset = get_dataset(config)
        problems = dataset["problem"]

        problem_results = []
        llm_fixable = []
        llm_token_counts = []  # Token count of LLM output per problem

        if use_parallel_loading:
            # --- PARALLEL MODE (Fast for A100/L4) ---
            from vllm import LLM, SamplingParams
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print("Loading SLM and LLM in Parallel...", flush=True)
            slm = LLM(
                model=draft_model_path, 
                gpu_memory_utilization=gpu_util, 
                enforce_eager=False,
                enable_prefix_caching=True, 
                tensor_parallel_size=num_gpus if num_gpus > 0 else 1, 
                max_model_len=8192, 
                dtype=dtype
            )
            
            # Clear cache to make room for LLM
            torch.cuda.empty_cache()
            
            print("Loading LLM into GPU...", flush=True)
            llm_device_map = "cuda:0" if num_gpus == 1 else "auto"
            if should_load_4bit:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                llm = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=llm_device_map,
                    quantization_config=bnb_config,
                    low_cpu_mem_usage=True
                ).eval()
            else:
                llm = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    device_map=llm_device_map, 
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True
                ).eval()
            llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            sampling_params = SamplingParams(temperature=config.temperature, max_tokens=config.max_tokens, top_p=config.top_p, stop=["\n\n"], n=1, logprobs=10)

            for prob_idx, problem in enumerate(problems):
                # Print progress to prevent the appearance of being "stuck"
                if prob_idx % 5 == 0:
                    print(f"Processing Calibration Sample {prob_idx + 1}/{len(problems)}...", flush=True)
                
                # SLM part
                current_text = ""
                steps_info = []
                for i in range(config.num_iterations):
                    conv = build_conv(problem, current_text, config.system_prompt)
                    templated = slm.get_tokenizer().apply_chat_template(conv, tokenize=False, add_generation_prompt=(i==0), continue_final_message=(i>0))
                    outputs = slm.generate([templated], sampling_params, use_tqdm=False)
                    output = outputs[0].outputs[0]
                    conf_metrics = calculate_confidence_score(output.logprobs)
                    scores_dict = {name: conf_metrics[idx] for name, idx in STRATEGY_MAP.items()}
                    steps_info.append({"scores": scores_dict, "text": output.text, "token_count": len(output.token_ids)})
                    current_text += output.text
                    if output.stop_reason == "EOS" or output.text == "": break
                problem_results.append({"slm_final_text": current_text, "steps": steps_info})

                # LLM part
                conv = build_conv(problem, "", config.system_prompt)
                templated = llm_tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                input_ids = llm_tokenizer(templated, return_tensors="pt").to(llm.device)
                with torch.no_grad():
                    out_ids = llm.generate(**input_ids, max_new_tokens=config.max_tokens)
                llm_out_tokens = out_ids.shape[1] - input_ids.input_ids.shape[1]
                llm_token_counts.append(llm_out_tokens)
                llm_text = llm_tokenizer.decode(out_ids[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True)
                
                gt = dataset[prob_idx].get("answer", dataset[prob_idx].get("solution", ""))
                if "boolq" in ds_name.lower():
                    gt_ans_val = dataset[prob_idx]["answer"]
                    gt_str = gt_ans_val if isinstance(gt_ans_val, str) else ("yes" if gt_ans_val else "no")
                    llm_fixable.append(gt_str == extract_bool(llm_text))
                else:
                    temp_samples = [{"problem": problem, "pred": llm_text, "solution": gt, "answer": gt, "completions": [llm_text]}]
                    _, llm_eval = evaluate(data_name="math", prompt_type="cot", samples=temp_samples, pred_keys=["pred"])
                    llm_fixable.append(llm_eval["acc"]["pred"] > 0)
            
            if hasattr(slm, "llm_engine") and hasattr(slm.llm_engine, "engine_core"):
                try:
                    slm.llm_engine.engine_core.shutdown()
                except Exception:
                    pass
            del slm, llm
            clear_gpu_memory()

        else:
            # --- SEQUENTIAL MODE (Safe for T4) ---
            # (Same as the previous sequential implementation)
            from vllm import LLM, SamplingParams
            print("PHASE 1: Loading SLM (vLLM)...", flush=True)
            slm = LLM(
                model=draft_model_path, 
                gpu_memory_utilization=gpu_util, 
                enforce_eager=True,
                enable_prefix_caching=True, 
                tensor_parallel_size=num_gpus if num_gpus > 0 else 1, 
                max_model_len=8192, 
                dtype=dtype
            )
            print("SLM Loaded successfully.", flush=True)
            sampling_params = SamplingParams(temperature=config.temperature, max_tokens=config.max_tokens, top_p=config.top_p, stop=["\n\n"], n=1, logprobs=10)

            print(f"Starting SLM generation for {len(problems)} problems...", flush=True)
            for prob_idx, problem in enumerate(problems):
                if prob_idx % 5 == 0: print(f"SLM Problem {prob_idx+1}/{len(problems)}...", flush=True)
                current_text = ""
                steps_info = []
                for i in range(config.num_iterations):
                    conv = build_conv(problem, current_text, config.system_prompt)
                    templated = slm.get_tokenizer().apply_chat_template(conv, tokenize=False, add_generation_prompt=(i==0), continue_final_message=(i>0))
                    outputs = slm.generate([templated], sampling_params, use_tqdm=False)
                    output = outputs[0].outputs[0]
                    conf_metrics = calculate_confidence_score(output.logprobs)
                    scores_dict = {name: conf_metrics[idx] for name, idx in STRATEGY_MAP.items()}
                    steps_info.append({"scores": scores_dict, "text": output.text, "token_count": len(output.token_ids)})
                    current_text += output.text
                    if output.stop_reason == "EOS" or output.text == "": break
                problem_results.append({"slm_final_text": current_text, "steps": steps_info})
            if hasattr(slm, "llm_engine") and hasattr(slm.llm_engine, "engine_core"):
                try:
                    slm.llm_engine.engine_core.shutdown()
                except Exception:
                    pass
            del slm; clear_gpu_memory()

            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("PHASE 2: Loading LLM (Transformers)...", flush=True)
            llm_device_map = "cuda:0" if num_gpus == 1 else "auto"
            if should_load_4bit:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                llm = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=llm_device_map,
                    quantization_config=bnb_config,
                    low_cpu_mem_usage=True
                ).eval()
            else:
                llm = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=llm_device_map,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True
                ).eval()
            llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            print(f"Starting LLM evaluation for {len(problems)} problems...", flush=True)
            for prob_idx, problem in enumerate(problems):
                if prob_idx % 5 == 0: print(f"LLM Problem {prob_idx+1}/{len(problems)}...", flush=True)
                conv = build_conv(problem, "", config.system_prompt)
                templated = llm_tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                input_ids = llm_tokenizer(templated, return_tensors="pt").to(llm.device)
                with torch.no_grad():
                    out_ids = llm.generate(**input_ids, max_new_tokens=config.max_tokens)
                llm_out_tokens = out_ids.shape[1] - input_ids.input_ids.shape[1]
                llm_token_counts.append(llm_out_tokens)
                llm_text = llm_tokenizer.decode(out_ids[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True)
                gt = dataset[prob_idx].get("answer", dataset[prob_idx].get("solution", ""))
                if "boolq" in ds_name.lower():
                    gt_ans_val = dataset[prob_idx]["answer"]
                    gt_str = gt_ans_val if isinstance(gt_ans_val, str) else ("yes" if gt_ans_val else "no")
                    llm_fixable.append(gt_str == extract_bool(llm_text))
                else:
                    temp_samples = [{"problem": problem, "pred": llm_text, "solution": gt, "answer": gt, "completions": [llm_text]}]
                    _, llm_eval = evaluate(data_name="math", prompt_type="cot", samples=temp_samples, pred_keys=["pred"])
                    llm_fixable.append(llm_eval["acc"]["pred"] > 0)
            del llm; clear_gpu_memory()

        # --- Accuracy Evaluation & Sweeping (Common) ---
        if "boolq" in ds_name.lower():
            slm_correct_list = []
            for idx, res in enumerate(problem_results):
                gt_val = dataset[idx]["answer"]
                gt_str = gt_val if isinstance(gt_val, str) else ("yes" if gt_val else "no")
                slm_correct_list.append(gt_str == extract_bool(res["slm_final_text"]))
        else:
            if "math" in ds_name.lower():
                eval_name = "math"
            elif "gsm8k" in ds_name.lower():
                eval_name = "gsm8k"
            elif "boolq" in ds_name.lower():
                eval_name = "boolq"
            else:
                eval_name = ds_name
            temp_samples = [{"problem": p, "pred": r["slm_final_text"], "solution": dataset[idx]["answer"], "answer": dataset[idx]["answer"], "completions": [r["slm_final_text"]]} for idx, (p, r) in enumerate(zip(problems, problem_results))]
            slm_eval_samples, _ = evaluate(data_name=eval_name, prompt_type="cot", samples=temp_samples, pred_keys=["pred"])
            slm_correct_list = [s["correct_completions"][0] for s in slm_eval_samples]

        best_ds_config = {"method": None, "threshold": 0, "acc": 0, "cost": 0}
        plt.figure(figsize=(10, 6))
        method_stats = {}

        # ── FIX 1: Compute bottleneck scores ONLY for fixable problems ──
        # A "fixable" problem is one where SLM was WRONG but LLM was RIGHT.
        # These are the only problems where an intervention would actually help,
        # so only their worst-step scores should drive the threshold calibration.
        fixable_mask = [
            (not slm_correct_list[i]) and llm_fixable[i]
            for i in range(len(problems))
        ]
        print(f"Fixable problems (SLM wrong, LLM right): {sum(fixable_mask)}/{len(problems)}", flush=True)

        for method in ["probs_mean", "probs_min", "entropy", "top_2_diff", "mean_least_3"]:
            # ── FIX 2: Collect bottleneck (worst-step) score per problem ──
            # For ALL problems (needed for the full threshold sweep below)
            all_bottleneck_scores = []
            for res in problem_results:
                scores = [step["scores"][method] for step in res["steps"]]
                if not scores:
                    all_bottleneck_scores.append(None)
                    continue
                # Worst score: max entropy (high = uncertain), min for probs (low = uncertain)
                if method in ("entropy", "mean_least_3"):
                    all_bottleneck_scores.append(max(scores))
                else:
                    all_bottleneck_scores.append(min(scores))

            # Derive threshold range from ALL bottleneck scores so the sweep covers 0% to 100% cost
            valid_scores = [s for s in all_bottleneck_scores if s is not None]
            if not valid_scores:
                print(f"  [{method}] No valid scores found – skipping.", flush=True)
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
                        # ── FIX 3: Cost = llm_tokens / (slm_tokens + llm_tokens) ──
                        slm_tokens = sum(s["token_count"] for s in problem_results[i]["steps"])
                        llm_tokens = llm_token_counts[i]
                        total_tokens = slm_tokens + llm_tokens
                        usage_ratio = llm_tokens / total_tokens if total_tokens > 0 else 1.0
                        problem_token_ratios.append(usage_ratio)

                        if llm_fixable[i]: correct += 1
                    else:
                        problem_token_ratios.append(0.0)
                        if slm_correct_list[i]: correct += 1

                accuracy = (correct / len(problems)) * 100
                cost = np.mean(problem_token_ratios) * 100
                accs.append(accuracy); costs.append(cost)
            
            # Find the best threshold for this method
            # We want highest accuracy. If tied, we want lowest cost.
            best_idx = 0
            for idx in range(len(accs)):
                if accs[idx] > accs[best_idx]:
                    best_idx = idx
                elif accs[idx] == accs[best_idx] and costs[idx] < costs[best_idx]:
                    best_idx = idx
            
            if accs[best_idx] > best_ds_config["acc"] or (accs[best_idx] == best_ds_config["acc"] and costs[best_idx] < best_ds_config["cost"]):
                best_ds_config = {"method": method, "threshold": thresholds[best_idx], "acc": accs[best_idx], "cost": costs[best_idx]}
            method_stats[method] = {"thresholds": thresholds.tolist(), "accuracies": accs, "costs": costs}
            plt.plot(costs, accs, marker='o', label=method)

        plt.xlabel("Cost (% LLM Calls)"); plt.ylabel("Accuracy (%)"); plt.title(f"Elbow Graph - {ds_name}"); plt.legend(); plt.grid(True)
        plt.savefig(f"outputs/calibration/elbow_{ds_name}.png"); plt.close()
        
        # Fallback to probs_mean with threshold 0.0 if no method outperformed SLM (e.g. 0% calibration accuracy)
        if best_ds_config["method"] is None:
            best_ds_config = {"method": "probs_mean", "threshold": 0.0, "acc": 0.0, "cost": 0.0}
            
        all_dataset_results[ds_name] = {"best_config": best_ds_config, "all_methods": method_stats}
        
        print("\n" + "-"*40, flush=True)
        print(f"WINNER FOR DATASET: {ds_name}", flush=True)
        print(f"method={best_ds_config['method']}", flush=True)
        print(f"threshold={best_ds_config['threshold']:.6f}", flush=True)
        print(f"accuracy={best_ds_config['acc']:.2f}%", flush=True)
        print(f"cost={best_ds_config['cost']:.2f}% LLM usage", flush=True)
        print("-"*40 + "\n", flush=True)

        # --- FULL RUN (Phase 4) ---
        if args.num_full_samples is not None and args.num_full_samples > 0:
            print(f"Launching Full Run for {ds_name} using optimal threshold...", flush=True)
            import subprocess
            cmd = [
                "python", "scripts/test_time_compute.py",
                "recipes/qwen_test.yaml",
                f"--dataset_name={ds_name}",
                "--smart_search=True",
                "--score_method=conf",
                f"--conf_strategy={best_ds_config['method']}",
                f"--threshold={best_ds_config['threshold']}",
                f"--num_samples={args.num_full_samples}"
            ]
            if should_load_4bit:
                cmd.append("--load_in_4bit=True")
            subprocess.run(cmd)
        elif args.num_full_samples == -1:
            print(f"Launching Full Run for {ds_name} using optimal threshold...", flush=True)
            import subprocess
            cmd = [
                "python", "scripts/test_time_compute.py",
                "recipes/qwen_test.yaml",
                f"--dataset_name={ds_name}",
                "--smart_search=True",
                "--score_method=conf",
                f"--conf_strategy={best_ds_config['method']}",
                f"--threshold={best_ds_config['threshold']}",
                "--num_samples=-1"
            ]
            if should_load_4bit:
                cmd.append("--load_in_4bit=True")
            subprocess.run(cmd)
        else:
            print(f"Skipping Full Run for {ds_name} as num_full_samples is {args.num_full_samples}", flush=True)

    print("\n" + "="*50, flush=True)
    print("ALL CALIBRATIONS COMPLETE", flush=True)
    for ds, res in all_dataset_results.items():
        bc = res["best_config"]
        method_str = str(bc['method']) if bc['method'] is not None else "None"
        threshold_val = bc['threshold'] if bc['threshold'] is not None else 0.0
        print(f"Dataset: {ds:10} | Best Method: {method_str:12} | Threshold: {threshold_val:.6f}", flush=True)
    print("="*50 + "\n", flush=True)

    with open("outputs/calibration/calibration_summary.json", "w") as f:
        json.dump(all_dataset_results, f, indent=4)

if __name__ == "__main__":
    calibrate()
