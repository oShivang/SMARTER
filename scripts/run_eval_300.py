import sys
import os
import gc
import json
import argparse
import copy
import math
import subprocess
import torch
from datasets import Dataset

os.environ["PYTHONWARNINGS"] = "ignore"
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("src/evaluation"))

from sal.config import Config
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score
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
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
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

def load_slm_vllm(config, num_gpus, device_capability, dtype, gpu_util):
    from vllm import LLM
    print(f"Loading SLM (vLLM) from: {config.draft_model_path}", flush=True)
    slm = LLM(
        model=config.draft_model_path,
        gpu_memory_utilization=gpu_util,
        enforce_eager=False,
        enable_prefix_caching=True,
        tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
        max_model_len=8192,
        dtype=dtype
    )
    print("SLM Loaded successfully in vLLM.", flush=True)
    return slm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_test", type=str, default="recipes/qwen_test.yaml")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--datasets", type=str, default="gsm8k,math500,boolq")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    
    args, unknown = parser.parse_known_args()
    
    # Load Configuration structure
    h4_parser_test = H4ArgumentParser(Config)
    config_test = h4_parser_test.parse_yaml_and_args(args.config_test, unknown)[0]
    
    # Force 300 samples
    config_test.num_samples = args.num_samples
    dataset_list = args.datasets.split(",")
    print(f"Target Datasets: {dataset_list} | Samples per dataset: {args.num_samples}", flush=True)
    
    # Find calibration_summary.json
    calib_json = None
    for opt in [
        "outputs/calibration/calibration_summary.json",
        "outputs/calibration_summary.json",
        "output/calibration/calibration_summary.json",
        "output/calibration_summary.json"
    ]:
        if os.path.exists(opt):
            calib_json = opt
            break
            
    if not calib_json:
        output_base = "output" if os.path.exists("output") else "outputs"
        calib_json = f"{output_base}/calibration/calibration_summary.json"
        if not os.path.exists(calib_json):
            print(f"WARNING: Calibration summary not found at {calib_json}. Using defaults.", flush=True)
            calibration_summary = {}
        else:
            with open(calib_json) as f:
                calibration_summary = json.load(f)
    else:
        with open(calib_json) as f:
            calibration_summary = json.load(f)
            
    # Auto-detect GPU and memory settings
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
    print(f"SYSTEM PROFILE (300-SAMPLE PIPELINE):", flush=True)
    print(f"  GPUs Available : {num_gpus} | Capability: {device_capability}", flush=True)
    print(f"  Total VRAM     : {total_vram:.2f} GB | System RAM: {total_sys_ram:.2f} GB", flush=True)
    print(f"  4-Bit Precision: {'ENABLED' if should_load_4bit else 'DISABLED'}", flush=True)
    print(f"  HF Batch Size  : {hf_batch_size}", flush=True)
    print("="*60, flush=True)
    
    # ----------------------------------------------------
    # STEP 1: SLM Baseline Run (No vLLM, HF batching)
    # ----------------------------------------------------
    print("\n==================================================")
    print("   PHASE 1: RUNNING SLM BASELINE (NO VLLM BATCHED)")
    print("==================================================")
    
    slm_model_path = config_test.draft_model_path if config_test.draft_model_path else config_test.model_path
    slm_hf, slm_tokenizer = load_hf_model(slm_model_path, num_gpus, torch_dtype, should_load_4bit)
    
    for ds_name in dataset_list:
        print(f"\n--- Running SLM Baseline on {ds_name} ---", flush=True)
        baseline_config_ds = copy.deepcopy(config_test)
        baseline_config_ds.dataset_name = ds_name
        baseline_config_ds.smart_search = False
        baseline_config_ds.score_method = "conf"
        baseline_config_ds.n = 1
        baseline_config_ds.num_iterations = 1
        
        full_dataset = get_dataset(baseline_config_ds)
        
        # Map baseline using HF
        baseline_ds = run_baseline_hf(full_dataset, baseline_config_ds, slm_hf, slm_tokenizer, hf_batch_size)
        
        n_completions = baseline_config_ds.n
        baseline_ds = baseline_ds.map(lambda x: {"scores": [[1.0] for _ in range(n_completions)]}, load_from_cache_file=False)
        baseline_ds = score(baseline_ds, baseline_config_ds)
        
        output_base = "output" if os.path.exists("output") else "outputs"
        baseline_config_ds.output_dir = f"{output_base}/{ds_name}/slm_baseline"
        baseline_config_ds.draft_model_path = baseline_config_ds.model_path
        evaluate_and_save_dataset(baseline_ds, baseline_config_ds, is_baseline=True)
        
    print("Unloading SLM HF model...", flush=True)
    del slm_hf, slm_tokenizer
    clear_gpu_memory()
    
    # ----------------------------------------------------
    # STEP 2: LLM Baseline Run (No vLLM, HF batching)
    # ----------------------------------------------------
    print("\n==================================================")
    print("   PHASE 2: RUNNING LLM BASELINE (NO VLLM BATCHED)")
    print("==================================================")
    
    llm_hf, llm_tokenizer = load_hf_model(config_test.model_path, num_gpus, torch_dtype, should_load_4bit)
    
    for ds_name in dataset_list:
        print(f"\n--- Running LLM Baseline on {ds_name} ---", flush=True)
        baseline_config_ds = copy.deepcopy(config_test)
        baseline_config_ds.dataset_name = ds_name
        baseline_config_ds.smart_search = False
        baseline_config_ds.score_method = "conf"
        baseline_config_ds.n = 1
        baseline_config_ds.num_iterations = 1
        
        full_dataset = get_dataset(baseline_config_ds)
        
        # Map baseline using HF
        baseline_ds = run_baseline_hf(full_dataset, baseline_config_ds, llm_hf, llm_tokenizer, hf_batch_size)
        
        n_completions = baseline_config_ds.n
        baseline_ds = baseline_ds.map(lambda x: {"scores": [[1.0] for _ in range(n_completions)]}, load_from_cache_file=False)
        baseline_ds = score(baseline_ds, baseline_config_ds)
        
        output_base = "output" if os.path.exists("output") else "outputs"
        baseline_config_ds.output_dir = f"{output_base}/{ds_name}/llm_baseline"
        baseline_config_ds.draft_model_path = baseline_config_ds.model_path
        evaluate_and_save_dataset(baseline_ds, baseline_config_ds, is_baseline=True)
        
    # ----------------------------------------------------
    # STEP 3: SMART Evaluation (vLLM for SLM, HF for LLM)
    # ----------------------------------------------------
    print("\n==================================================")
    print("   PHASE 3: RUNNING SMART EVALUATION (300 SAMPLES)")
    print("==================================================")
    
    # Load SLM in vLLM (with conservative memory config to run alongside LLM Transformers model)
    slm_vllm = load_slm_vllm(config_test, num_gpus, device_capability, dtype, gpu_util)
    clear_gpu_memory()
    
    for ds_name in dataset_list:
        print(f"\n--- Running SMART Search on {ds_name} ---", flush=True)
        
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
        
        full_dataset = get_dataset(smart_config_ds)
        
        approach_suffix = "_smart" if smart_config_ds.smart_search else ""
        approach_suffix += "_conf" if smart_config_ds.score_method == 'conf' else ""
        approach_name = smart_config_ds.approach + approach_suffix
        approach_fn = APPROACHES[approach_name]
        
        # Map SMART search using slm (vLLM) and llm (HF Causal model already loaded in step 2)
        smart_ds = full_dataset.map(
            approach_fn,
            batched=True,
            batch_size=smart_config_ds.search_batch_size,
            fn_kwargs={"config": smart_config_ds, "slm": slm_vllm, "prm": None, "llm": llm_hf},
            desc=f"Running SMART search ({ds_name})",
            load_from_cache_file=False
        )
        
        output_base = "output" if os.path.exists("output") else "outputs"
        smart_config_ds.output_dir = f"{output_base}/{ds_name}/smart_results"
        evaluate_and_save_dataset(smart_ds, smart_config_ds, is_baseline=False)
        
    print("\nTearing down models and freeing up GPU memory...", flush=True)
    unload_model(slm_vllm)
    del llm_hf
    clear_gpu_memory()
    
    # ----------------------------------------------------
    # STEP 4: Generate Report Card
    # ----------------------------------------------------
    print("\n==================================================")
    print("   PHASE 4: COMPILING FINAL REPORT CARD")
    print("==================================================")
    
    output_base = "output" if os.path.exists("output") else "outputs"
    if calib_json and os.path.exists(calib_json):
        calib_path = calib_json
    else:
        calib_path = f"{output_base}/calibration/calibration_summary.json"
        
    cmd = [
        sys.executable,
        "scripts/generate_report.py",
        "--calibration_json", calib_path,
        "--output_dir", output_base,
        "--smart_tag", "smart_results",
        "--slm_tag", "slm_baseline",
        "--llm_tag", "llm_baseline"
    ]
    subprocess.run(cmd)
    
    print("\n" + "="*50)
    print("✅ 300-SAMPLE EVALUATION COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
