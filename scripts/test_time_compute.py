import sys
import os
sys.path.append(os.path.abspath("src"))

import logging
import random
import numpy as np
import math

import torch
from vllm import LLM
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

from sal.config import Config
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score
from sal.search import \
    best_of_n, \
    best_of_n_conf, \
    smart_best_of_n, \
    smart_best_of_n_conf, \
    beam_search, \
    beam_search_conf, \
    smart_beam_search, \
    smart_beam_search_conf
from datasets import Dataset
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

APPROACHES = {
    "beam_search":beam_search,
    "beam_search_smart": smart_beam_search,
    "beam_search_conf": beam_search_conf,
    "beam_search_smart_conf": smart_beam_search_conf,
    "best_of_n": best_of_n,
    "best_of_n_smart": smart_best_of_n,
    "best_of_n_conf": best_of_n_conf,
    "best_of_n_smart_conf": smart_best_of_n_conf,
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable optimizations for reproducibility

set_seed(42) 

def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    num_gpus = torch.cuda.device_count()
    print(f"The number of available GPUs: {num_gpus}")
    print("="*20)

    if config.num_samples == -1:
        config.num_samples = None
    
    # configure approach name
    approach_suffix = "_smart" if config.smart_search else ""
    approach_suffix += "_conf" if config.score_method == 'conf' else ""
    approach_name = config.approach + approach_suffix
    
    if approach_name not in APPROACHES:
        raise ValueError(f"Invalid score method: {config.score_method}")
    approach_fn = APPROACHES[approach_name]
    
    # Log the search method and score method with detailed parameters
    print("\n" + "="*50)
    print("      🚀 SMART INFERENCE PIPELINE INITIALIZED")
    print("="*50)
    print(f"Search Approach:      {approach_name.upper()}")
    print(f"Scoring Method:       {config.score_method.upper()}")
    
    if config.score_method == 'conf':
        print(f"Confidence Strategy:  {config.conf_strategy}")
        print(f"Aggregation Strategy: {config.agg_strategy}")
    
    print("-" * 50)
    print(f"Dataset:              {config.dataset_name}")
    print(f"Split:                {config.dataset_split}")

    if config.num_samples == -1:
        config.num_samples = None

    if config.smart_search:
        print(f"Intervention Threshold: {config.threshold}")
        print(f"Max Iterations:       {config.num_iterations}")
    
    print(f"N (Completions):      {config.n}")
    if "beam" in approach_name:
        print(f"Beam Width:           {config.beam_width}")
    
    print(f"SLM (Draft):          {config.draft_model_path if config.draft_model_path else config.model_path}")
    print(f"LLM (Intervention):   {config.model_path}")
    print("="*50 + "\n")
    
    if config.smart_search:                
        mp.set_start_method("spawn", force=True)
        # Detect if GPU supports bf16, else use fp16 (T4 fallback)
        num_gpus = torch.cuda.device_count()
        device_capability = torch.cuda.get_device_capability() if num_gpus > 0 else (0, 0)
        dtype = "bfloat16" if device_capability[0] >= 8 else "half"
        torch_dtype = torch.bfloat16 if device_capability[0] >= 8 else torch.float16

        # Conservative VRAM allocation when loading two models
        gpu_util = 0.35 if config.smart_search else config.gpu_memory_utilization

        draft_path = config.draft_model_path if config.draft_model_path else config.model_path
        slm = LLM(
            model=draft_path,
            gpu_memory_utilization=gpu_util,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            max_model_len=2048,
            dtype=dtype,
        )
        
        torch.cuda.empty_cache()

        # Auto-detect if we should load in 4-bit dynamically based on hardware limits
        should_load_4bit = config.load_in_4bit
        if not should_load_4bit:
            try:
                import psutil
                total_sys_ram = psutil.virtual_memory().total / (1024**3)
            except Exception:
                total_sys_ram = 32.0
            
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                total_vram = 0
            
            if total_vram > 0 and (total_vram <= 24.0 or total_sys_ram <= 16.0):
                should_load_4bit = True
                logger.info(f"Auto-detect: VRAM ({total_vram:.1f}GB) or System RAM ({total_sys_ram:.1f}GB) is limited. Dynamically enabling 4-bit precision loading.")

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
                config.model_path,
                device_map=llm_device_map,
                quantization_config=bnb_config,
            ).eval()
        else:
            llm = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                device_map=llm_device_map,
                torch_dtype=torch_dtype,
            ).eval()
        
        if config.score_method == 'prm':
            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": None, "llm": llm},
                desc="Running search",
                load_from_cache_file=False,
            )    
        elif config.score_method == 'conf':
            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "slm": slm, "prm": None, "llm": llm},
                desc="Running search",
                load_from_cache_file=False,
            )
        else:
            raise ValueError(f"Invalid score method: {config.score_method}")
    else:
        llm = LLM(
            model=config.model_path,
            revision="main",
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            max_model_len=2048,
        )
        
        if config.score_method == 'prm':
            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "llm": llm, "prm": None, "slm": None},
                desc="Running search",
                load_from_cache_file=False,
            )
        
        elif config.score_method == 'conf':
            dataset = get_dataset(config)
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "llm": llm, "prm": None, "slm": None},
                desc="Running search",
                load_from_cache_file=False,
            )    
        else: 
            raise ValueError(f"Invalid score method: {config.score_method}")

    if not config.smart_search:
        dataset = score(dataset, config)
    
    save_dataset(dataset, config)
    
    import sys
    sys.path.append("src/evaluation")
    from evaluation.evaluate import evaluate
    
    if not config.smart_search and (config.approach == "best_of_n" or config.approach == "beam_search"):
        # Calculate powers of 2 up to config.n, and explicitly include config.n
        subsets = [2**i for i in range(int(math.log2(config.n)) + 1)]
        if config.n not in subsets:
            subsets.append(config.n)
        subsets = sorted(list(set(subsets)))
        
        keys = []
        for n in subsets:
            keys.extend([f"pred_weighted@{n}", f"pred_maj@{n}", f"pred_naive@{n}"])
    else:
        keys = ["pred"]

    # Evaluation logic based on dataset
    if "boolq" in config.dataset_name.lower():
        logger.info("Using simple yes/no matching for BoolQ evaluation")
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
    else:
        if "math" in config.dataset_name.lower():
            data_type = "math"
        elif "gsm8k" in config.dataset_name.lower():
            data_type = "gsm8k"
        elif "boolq" in config.dataset_name.lower():
            data_type = "boolq"
        else:
            data_type = config.dataset_name
        dataset, result = evaluate(data_name=data_type, prompt_type=None, samples=dataset, pred_keys=keys)
    
    # Save the final dataset
    dataset = Dataset.from_list([{k: v for k, v in dict(sample).items() if k != 'pred_completions'} for sample in dataset])
    save_dataset(dataset, config)
    
    logger.info(result)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
