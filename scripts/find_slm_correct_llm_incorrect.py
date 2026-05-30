#!/usr/bin/env python
import sys
import os
import gc
import json
import argparse
import copy
import torch
from datasets import Dataset

os.environ["PYTHONWARNINGS"] = "ignore"
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("src/evaluation"))

from sal.config import Config
from sal.utils.data import get_dataset
from sal.utils.parser import H4ArgumentParser
from sal.search.utils import build_conv
from evaluation.evaluate import evaluate

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
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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

def load_slm(config, num_gpus, dtype, gpu_util):
    from vllm import LLM
    print(f"Loading SLM (vLLM) from: {config.draft_model_path if config.draft_model_path else config.model_path}", flush=True)
    slm = LLM(
        model=config.draft_model_path if config.draft_model_path else config.model_path,
        gpu_memory_utilization=gpu_util,
        enforce_eager=False,
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="recipes/qwen_calibrate.yaml")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math500", "boolq"])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--load_in_4bit", action="store_true")
    
    args, unknown = parser.parse_known_args()
    
    h4_parser = H4ArgumentParser(Config)
    config = h4_parser.parse_yaml_and_args(args.config, unknown)[0]
    config.dataset_name = args.dataset
    config.num_samples = args.num_samples
    
    print(f"Configuring Run:")
    print(f"  Dataset    : {config.dataset_name}")
    print(f"  SLM Path   : {config.draft_model_path}")
    print(f"  LLM Path   : {config.model_path}")
    print(f"  Num Samples: {config.num_samples}")
    
    num_gpus, device_capability, total_memory, total_sys_ram = get_gpu_info()
    print(f"Hardware Info: GPUs: {num_gpus}, Mem: {total_memory:.1f}GB, RAM: {total_sys_ram:.1f}GB")
    
    # Configure precision/memory
    if device_capability[0] >= 8 and total_memory >= 70:
        torch_dtype = torch.bfloat16
        dtype = "bfloat16"
        gpu_util = 0.85
        use_parallel_loading = True
    elif total_memory >= 20:
        torch_dtype = torch.float16
        dtype = "float16"
        gpu_util = 0.85
        use_parallel_loading = True
    else:
        torch_dtype = torch.float16
        dtype = "float16"
        gpu_util = 0.70
        use_parallel_loading = False

    hf_batch_size = 64 if use_parallel_loading else 16
    
    dataset = get_dataset(config)
    problems = dataset["problem"]
    answers = dataset["answer"]
    
    # -----------------
    # 1. RUN SLM (vLLM)
    # -----------------
    slm_vllm = load_slm(config, num_gpus, dtype, gpu_util)
    clear_gpu_memory()
    
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        n=1,
        logprobs=10
    )
    
    print("\n--- Generating SLM responses step-by-step ---", flush=True)
    problem_results = []
    for prob_idx, problem in enumerate(problems):
        if (prob_idx + 1) % 20 == 0 or prob_idx == 0 or (prob_idx + 1) == len(problems):
            print(f"  [SLM] Generating sample {prob_idx + 1}/{len(problems)}...", flush=True)
        current_text = ""
        for i in range(config.num_iterations):
            conv = build_conv(problem, current_text, config.system_prompt)
            templated = slm_vllm.get_tokenizer().apply_chat_template(
                conv, tokenize=False, add_generation_prompt=(i==0), continue_final_message=(i>0)
            )
            outputs = slm_vllm.generate([templated], sampling_params, use_tqdm=False)
            output = outputs[0].outputs[0]
            current_text += output.text
            if output.stop_reason == "EOS" or output.text == "": 
                break
        problem_results.append({"slm_final_text": current_text})
        
    print("Unloading SLM vLLM...", flush=True)
    unload_model(slm_vllm)
    clear_gpu_memory()
    
    # Evaluate SLM
    print("Evaluating SLM predictions...", flush=True)
    slm_correct_list = []
    if config.dataset_name == "boolq":
        for idx, res in enumerate(problem_results):
            gt_val = answers[idx]
            gt = gt_val if isinstance(gt_val, str) else ("yes" if gt_val else "no")
            slm_correct_list.append(gt.lower().strip() == extract_bool(res["slm_final_text"]))
    else:
        eval_name = "math" if "math" in config.dataset_name.lower() else ("gsm8k" if "gsm8k" in config.dataset_name.lower() else config.dataset_name)
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
        
    # -----------------
    # 2. RUN LLM (HF)
    # -----------------
    llm_hf, llm_tokenizer = load_hf_model(config.model_path, num_gpus, torch_dtype, args.load_in_4bit)
    
    batch_prompts = []
    for i, problem in enumerate(problems):
        conv = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": problem}
        ]
        templated = llm_tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        batch_prompts.append(templated)
        
    print("\n--- Generating LLM responses batched ---", flush=True)
    llm_tokenizer.padding_side = "left"
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        
    llm_outputs = []
    gen_kwargs = {"max_new_tokens": config.max_tokens}
    for idx_b in range(0, len(batch_prompts), hf_batch_size):
        sub_batch = batch_prompts[idx_b:idx_b+hf_batch_size]
        inputs = llm_tokenizer(sub_batch, return_tensors="pt", padding=True).to(llm_hf.device)
        with torch.no_grad():
            out_ids = llm_hf.generate(**inputs, **gen_kwargs)
        input_len = inputs.input_ids.shape[1]
        decoded = llm_tokenizer.batch_decode(out_ids[:, input_len:], skip_special_tokens=True)
        llm_outputs.extend(decoded)
        
    # Evaluate LLM
    print("Evaluating LLM predictions...", flush=True)
    llm_correct_list = []
    if config.dataset_name == "boolq":
        for prob_idx, problem in enumerate(problems):
            llm_text = llm_outputs[prob_idx]
            gt = answers[prob_idx]
            gt_str = "yes" if gt else "no"
            llm_correct_list.append(gt_str == extract_bool(llm_text))
    else:
        eval_name = "math" if "math" in config.dataset_name.lower() else ("gsm8k" if "gsm8k" in config.dataset_name.lower() else config.dataset_name)
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
        llm_correct_list = [s["correct_completions"][0] for s in llm_eval_samples]
        
    # Find Mismatches: SLM Correct, LLM Incorrect
    mismatches = []
    for i in range(len(problems)):
        if slm_correct_list[i] and not llm_correct_list[i]:
            mismatches.append({
                "index": i,
                "problem": problems[i],
                "answer": answers[i],
                "slm_output": problem_results[i]["slm_final_text"],
                "llm_output": llm_outputs[i]
            })
            
    print(f"\nFound {len(mismatches)} examples where SLM was correct but LLM was incorrect.")
    
    # Save all mismatches
    out_dir = "outputs/analysis"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mismatches_{config.dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(mismatches, f, indent=4)
    print(f"Saved all mismatches to {out_path}")
    
    # Print the first 5 mismatches nicely
    print("\n" + "="*80)
    print(f"   5 EXAMPLES WHERE SLM WAS CORRECT & LLM WAS INCORRECT ({config.dataset_name})")
    print("="*80)
    
    show_count = min(5, len(mismatches))
    for idx in range(show_count):
        m = mismatches[idx]
        print(f"\n### EXAMPLE {idx + 1} (Dataset Index: {m['index']})")
        print(f"**Question:**\n{m['problem']}")
        print(f"\n**Ground Truth Answer:** {m['answer']}")
        print(f"\n**SLM Final Output (Correct):**\n{m['slm_output']}")
        print(f"\n**LLM Final Output (Incorrect):**\n{m['llm_output']}")
        print("-" * 80)
        
    if len(mismatches) == 0:
        print("\nNo mismatches found in this sample batch where SLM was correct and LLM was incorrect.")

if __name__ == "__main__":
    main()
