import sys
import os
sys.path.append(os.path.abspath("src"))

import logging
import argparse
import numpy as np
import torch
import json
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
logger = logging.getLogger(__name__)

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def get_gpu_info():
    if not torch.cuda.is_available():
        return 0, (0, 0), 0
    num_gpus = torch.cuda.device_count()
    device_capability = torch.cuda.get_device_capability()
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) # GB
    return num_gpus, device_capability, total_memory

def calibrate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_calibration_samples", type=int, default=50)
    parser.add_argument("--num_full_samples", type=int, default=None)
    parser.add_argument("--datasets", type=str, default="math500,gsm8k,boolq")
    args, unknown = parser.parse_known_args()

    # Load config
    from sal.utils.parser import H4ArgumentParser
    h4_parser = H4ArgumentParser(Config)
    config = h4_parser.parse_yaml_file(os.path.abspath(args.config))[0]
    
    dataset_list = args.datasets.split(",")
    os.makedirs("outputs/calibration", exist_ok=True)
    all_dataset_results = {}

    model_path = str(config.model_path)
    draft_model_path = str(config.draft_model_path) if config.draft_model_path else model_path
    
    # Adaptive Memory Strategy
    num_gpus, device_capability, total_vram = get_gpu_info()
    dtype = "bfloat16" if device_capability[0] >= 8 else "half"
    torch_dtype = torch.bfloat16 if device_capability[0] >= 8 else torch.float16
    
    # Parallel mode needs more head-room; Sequential can be aggressive
    use_parallel_loading = total_vram > 24
    gpu_util = 0.4 if use_parallel_loading else 0.5
    
    logger.info(f"GPU Detected: {total_vram:.1f}GB VRAM. Strategy: {'Parallel' if use_parallel_loading else 'Sequential'} (Util: {gpu_util})")

    for ds_name in dataset_list:
        logger.info(f"\n{'='*20}\nProcessing Dataset: {ds_name}\n{'='*20}")
        config.dataset_name = ds_name
        config.num_samples = args.num_calibration_samples
        dataset = get_dataset(config)
        problems = dataset["problem"]

        problem_results = []
        llm_fixable = []

        if use_parallel_loading:
            # --- PARALLEL MODE (Fast for A100/L4) ---
            from vllm import LLM, SamplingParams
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info("Loading SLM and LLM in Parallel...")
            slm = LLM(model=draft_model_path, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, tensor_parallel_size=num_gpus if num_gpus > 0 else 1, max_model_len=2048, dtype=dtype)
            llm_device_map = "cuda:0" if num_gpus == 1 else "auto"
            llm = AutoModelForCausalLM.from_pretrained(model_path, device_map=llm_device_map, torch_dtype=torch_dtype).eval()
            llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            sampling_params = SamplingParams(temperature=config.temperature, max_tokens=config.max_tokens, top_p=config.top_p, stop=["\n\n"], n=1, logprobs=10)

            for prob_idx, problem in enumerate(problems):
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
                    steps_info.append({"scores": scores_dict, "text": output.text})
                    current_text += output.text
                    if output.stop_reason == "EOS" or output.text == "": break
                problem_results.append({"slm_final_text": current_text, "steps": steps_info})

                # LLM part
                conv = build_conv(problem, "", config.system_prompt)
                templated = llm_tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                input_ids = llm_tokenizer(templated, return_tensors="pt").to(llm.device)
                with torch.no_grad():
                    out_ids = llm.generate(**input_ids, max_new_tokens=config.max_tokens)
                llm_text = llm_tokenizer.decode(out_ids[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True)
                
                gt = dataset[prob_idx].get("answer", dataset[prob_idx].get("solution", ""))
                if ds_name == "boolq":
                    llm_fixable.append(("yes" if dataset[prob_idx]["answer"] else "no") in llm_text.lower())
                else:
                    temp_samples = [{"problem": problem, "pred": llm_text, "solution": gt, "answer": gt, "completions": [llm_text]}]
                    _, llm_eval = evaluate(data_name="math", prompt_type="cot", samples=temp_samples, pred_keys=["pred"])
                    llm_fixable.append(llm_eval["acc"]["pred"] > 0)
            
            del slm, llm
            clear_gpu_memory()

        else:
            # --- SEQUENTIAL MODE (Safe for T4) ---
            # (Same as the previous sequential implementation)
            from vllm import LLM, SamplingParams
            logger.info("PHASE 1: Loading SLM (vLLM)...")
            slm = LLM(model=draft_model_path, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, tensor_parallel_size=num_gpus if num_gpus > 0 else 1, max_model_len=2048, dtype=dtype)
            sampling_params = SamplingParams(temperature=config.temperature, max_tokens=config.max_tokens, top_p=config.top_p, stop=["\n\n"], n=1, logprobs=10)

            for prob_idx, problem in enumerate(problems):
                if prob_idx % 2 == 0: logger.info(f"SLM Problem {prob_idx+1}/{len(problems)}...")
                current_text = ""
                steps_info = []
                for i in range(config.num_iterations):
                    conv = build_conv(problem, current_text, config.system_prompt)
                    templated = slm.get_tokenizer().apply_chat_template(conv, tokenize=False, add_generation_prompt=(i==0), continue_final_message=(i>0))
                    outputs = slm.generate([templated], sampling_params, use_tqdm=False)
                    output = outputs[0].outputs[0]
                    conf_metrics = calculate_confidence_score(output.logprobs)
                    scores_dict = {name: conf_metrics[idx] for name, idx in STRATEGY_MAP.items()}
                    steps_info.append({"scores": scores_dict, "text": output.text})
                    current_text += output.text
                    if output.stop_reason == "EOS" or output.text == "": break
                problem_results.append({"slm_final_text": current_text, "steps": steps_info})
            del slm; clear_gpu_memory()

            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info("PHASE 2: Loading LLM (Transformers)...")
            llm_device_map = "cuda:0" if num_gpus == 1 else "auto"
            llm = AutoModelForCausalLM.from_pretrained(model_path, device_map=llm_device_map, torch_dtype=torch_dtype).eval()
            llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
            for prob_idx, problem in enumerate(problems):
                if prob_idx % 2 == 0: logger.info(f"LLM Problem {prob_idx+1}/{len(problems)}...")
                conv = build_conv(problem, "", config.system_prompt)
                templated = llm_tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                input_ids = llm_tokenizer(templated, return_tensors="pt").to(llm.device)
                with torch.no_grad():
                    out_ids = llm.generate(**input_ids, max_new_tokens=config.max_tokens)
                llm_text = llm_tokenizer.decode(out_ids[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True)
                gt = dataset[prob_idx].get("answer", dataset[prob_idx].get("solution", ""))
                if ds_name == "boolq":
                    llm_fixable.append(("yes" if dataset[prob_idx]["answer"] else "no") in llm_text.lower())
                else:
                    temp_samples = [{"problem": problem, "pred": llm_text, "solution": gt, "answer": gt, "completions": [llm_text]}]
                    _, llm_eval = evaluate(data_name="math", prompt_type="cot", samples=temp_samples, pred_keys=["pred"])
                    llm_fixable.append(llm_eval["acc"]["pred"] > 0)
            del llm; clear_gpu_memory()

        # --- Accuracy Evaluation & Sweeping (Common) ---
        if ds_name == "boolq":
            slm_correct_list = [("yes" if dataset[idx]["answer"] else "no") in res["slm_final_text"].lower() for idx, res in enumerate(problem_results)]
        else:
            eval_name = "math" if "math" in ds_name or ds_name == "gsm8k" else ds_name
            temp_samples = [{"problem": p, "pred": r["slm_final_text"], "solution": dataset[idx]["answer"], "answer": dataset[idx]["answer"], "completions": [r["slm_final_text"]]} for idx, (p, r) in enumerate(zip(problems, problem_results))]
            slm_eval_samples, _ = evaluate(data_name=eval_name, prompt_type="cot", samples=temp_samples, pred_keys=["pred"])
            slm_correct_list = [s["correct_completions"][0] for s in slm_eval_samples]

        best_ds_config = {"method": None, "threshold": 0, "acc": 0, "cost": 0}
        plt.figure(figsize=(10, 6))
        method_stats = {}
        for method in ["probs_mean", "entropy", "top_2_diff", "mean_least_3"]:
            all_scores = [step["scores"][method] for res in problem_results for step in res["steps"]]
            if not all_scores: continue
            thresholds = np.linspace(min(all_scores), max(all_scores), 20)
            accs, costs = [], []
            for tau in thresholds:
                correct = 0; triggered_count = 0
                for i in range(len(problems)):
                    triggered = any(needs_correction(step["scores"][method], tau, method) for step in problem_results[i]["steps"])
                    if triggered:
                        triggered_count += 1
                        if llm_fixable[i]: correct += 1
                    else:
                        if slm_correct_list[i]: correct += 1
                accs.append((correct / len(problems)) * 100); costs.append((triggered_count / len(problems)) * 100)
            best_idx = np.argmax(accs)
            if accs[best_idx] > best_ds_config["acc"]:
                best_ds_config = {"method": method, "threshold": thresholds[best_idx], "acc": accs[best_idx], "cost": costs[best_idx]}
            method_stats[method] = {"thresholds": thresholds.tolist(), "accuracies": accs, "costs": costs}
            plt.plot(costs, accs, marker='o', label=method)

        plt.xlabel("Cost (% LLM Calls)"); plt.ylabel("Accuracy (%)"); plt.title(f"Elbow Graph - {ds_name}"); plt.legend(); plt.grid(True)
        plt.savefig(f"outputs/calibration/elbow_{ds_name}.png"); plt.close()
        all_dataset_results[ds_name] = {"best_config": best_ds_config, "all_methods": method_stats}
        logger.info(f"Best for {ds_name}: {best_ds_config['method']} @ {best_ds_config['threshold']:.4f}")

        # --- FULL RUN (Phase 4) ---
        import subprocess
        cmd = ["python", "scripts/test_time_compute.py", args.config, f"--dataset_name={ds_name}", "--smart_search=True", "--score_method=conf", f"--conf_strategy={best_ds_config['method']}", f"--threshold={best_ds_config['threshold']:.4f}"]
        if args.num_full_samples: cmd.append(f"--num_samples={args.num_full_samples}")
        subprocess.run(cmd)

    with open("outputs/calibration/calibration_summary.json", "w") as f:
        json.dump(all_dataset_results, f, indent=4)
    logger.info("Calibration complete. Summary saved.")

if __name__ == "__main__":
    calibrate()
