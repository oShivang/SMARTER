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
    
    # Detect GPU capability
    num_gpus = torch.cuda.device_count()
    device_capability = torch.cuda.get_device_capability() if num_gpus > 0 else (0, 0)
    dtype = "bfloat16" if device_capability[0] >= 8 else "half"
    torch_dtype = torch.bfloat16 if device_capability[0] >= 8 else torch.float16

    for ds_name in dataset_list:
        logger.info(f"\n{'='*20}\nProcessing Dataset: {ds_name}\n{'='*20}")
        config.dataset_name = ds_name
        config.num_samples = args.num_calibration_samples
        dataset = get_dataset(config)
        problems = dataset["problem"]

        # --- PHASE 1: SLM Trajectories ---
        logger.info("Loading SLM (vLLM)...")
        from vllm import LLM, SamplingParams
        slm = LLM(
            model=draft_model_path,
            gpu_memory_utilization=0.8,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
            max_model_len=2048,
            dtype=dtype,
        )
        
        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=["\n\n"],
            n=1,
            logprobs=10,
        )

        problem_results = []
        logger.info(f"Generating trajectories for {len(problems)} problems...")
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

        # UNLOAD SLM
        logger.info("Unloading SLM...")
        del slm
        clear_gpu_memory()

        # --- PHASE 2: LLM Potential ---
        logger.info("Loading LLM (Transformers)...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        llm = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch_dtype).eval()
        llm_tokenizer = AutoTokenizer.from_pretrained(model_path)

        llm_fixable = []
        logger.info(f"Checking LLM potential for {len(problems)} problems...")
        for prob_idx, problem in enumerate(problems):
            if prob_idx % 2 == 0: logger.info(f"LLM Problem {prob_idx+1}/{len(problems)}...")
            conv = build_conv(problem, "", config.system_prompt)
            templated = llm_tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            input_ids = llm_tokenizer(templated, return_tensors="pt").to(llm.device)
            with torch.no_grad():
                out_ids = llm.generate(**input_ids, max_new_tokens=config.max_tokens)
            llm_text = llm_tokenizer.decode(out_ids[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True)
            
            gt = dataset[prob_idx].get("answer", "")
            if ds_name == "boolq":
                gt_str = "yes" if dataset[prob_idx]["answer"] else "no"
                llm_fixable.append(gt_str in llm_text.lower())
            else:
                # evaluate expects 'solution' for math, 'answer' for others
                temp_samples = [{"problem": problem, "pred": llm_text, "solution": gt, "answer": gt, "completions": []}]
                _, llm_eval = evaluate(data_name="math", prompt_type="cot", samples=temp_samples, pred_keys=["pred"])
                llm_fixable.append(llm_eval["acc"]["pred"] > 0)

        # UNLOAD LLM
        logger.info("Unloading LLM...")
        del llm
        clear_gpu_memory()

        # --- PHASE 3: Accuracy Evaluation & Sweeping ---
        if ds_name == "boolq":
            slm_correct_list = [("yes" if dataset[idx]["answer"] else "no") in res["slm_final_text"].lower() for idx, res in enumerate(problem_results)]
        else:
            eval_name = "math" if "math" in ds_name or ds_name == "gsm8k" else ds_name
            temp_samples = [{"problem": p, "pred": r["slm_final_text"], "solution": dataset[idx]["answer"], "answer": dataset[idx]["answer"], "completions": []} for idx, (p, r) in enumerate(zip(problems, problem_results))]
            slm_eval_samples, _ = evaluate(data_name=eval_name, prompt_type="cot", samples=temp_samples, pred_keys=["pred"])
            slm_correct_list = [s["correct"][0] for s in slm_eval_samples]

        best_ds_config = {"method": None, "threshold": 0, "acc": 0, "cost": 0}
        plt.figure(figsize=(10, 6))
        method_stats = {}

        for method in ["probs_mean", "entropy", "top_2_diff", "mean_least_3"]:
            all_scores = [step["scores"][method] for res in problem_results for step in res["steps"]]
            if not all_scores: continue
            thresholds = np.linspace(min(all_scores), max(all_scores), 20)
            accs, costs = [], []
            for tau in thresholds:
                correct = 0
                triggered_count = 0
                for i in range(len(problems)):
                    triggered = any(needs_correction(step["scores"][method], tau, method) for step in problem_results[i]["steps"])
                    if triggered:
                        triggered_count += 1
                        if llm_fixable[i]: correct += 1
                    else:
                        if slm_correct_list[i]: correct += 1
                accs.append((correct / len(problems)) * 100)
                costs.append((triggered_count / len(problems)) * 100)
            
            # Elbow logic: find max acc with min cost (simplified: best acc in test run)
            best_idx = np.argmax(accs)
            if accs[best_idx] > best_ds_config["acc"]:
                best_ds_config = {"method": method, "threshold": thresholds[best_idx], "acc": accs[best_idx], "cost": costs[best_idx]}

            method_stats[method] = {"thresholds": thresholds.tolist(), "accuracies": accs, "costs": costs}
            plt.plot(costs, accs, marker='o', label=method)

        plt.xlabel("Cost (% LLM Calls)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Elbow Graph - {ds_name}")
        plt.legend(); plt.grid(True)
        plt.savefig(f"outputs/calibration/elbow_{ds_name}.png"); plt.close()

        all_dataset_results[ds_name] = {"best_config": best_ds_config, "all_methods": method_stats}
        logger.info(f"Best for {ds_name}: {best_ds_config['method']} @ {best_ds_config['threshold']:.4f}")

        # --- PHASE 4: FULL RUN ---
        import subprocess
        cmd = ["python", "scripts/test_time_compute.py", args.config, f"--dataset_name={ds_name}", "--smart_search=True", "--score_method=conf", f"--conf_strategy={best_ds_config['method']}", f"--threshold={best_ds_config['threshold']:.4f}"]
        if args.num_full_samples: cmd.append(f"--num_samples={args.num_full_samples}")
        subprocess.run(cmd)

    with open("outputs/calibration/calibration_summary.json", "w") as f:
        json.dump(all_dataset_results, f, indent=4)
    logger.info("Calibration complete. Summary saved.")

if __name__ == "__main__":
    calibrate()
