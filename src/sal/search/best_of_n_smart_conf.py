#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import logging
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, GenerationConfig
from transformers.generation.stopping_criteria import StopStringCriteria
import torch

from sal.config import Config
from sal.utils.score import aggregate_scores, calculate_confidence_score, STRATEGY_MAP, needs_correction
from sal.search.utils import build_conv

_TOKENIZER_CACHE = {}

def get_cached_tokenizer(model_path):
    if model_path not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model_path] = AutoTokenizer.from_pretrained(model_path)
    return _TOKENIZER_CACHE[model_path]

def smart_best_of_n_conf(x, config: Config, slm: LLM, llm: None, prm=None, **kwargs):
    logger = logging.getLogger(__name__)
    
    # Setup tokenizers
    slm_tokenizer = slm.get_tokenizer()
    if config.custom_chat_template is not None:
        slm_tokenizer.chat_template = config.custom_chat_template
        
    llm_tokenizer = get_cached_tokenizer(config.model_path)
    if config.custom_chat_template is not None:
        llm_tokenizer.chat_template = config.custom_chat_template
        
    stopping_criteria = StopStringCriteria(stop_strings="\n\n", tokenizer=llm_tokenizer)
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_tokens,
    )

    # Initial full generation params (no \n\n stop string)
    sampling_params_full = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,
        logprobs=10,
    )

    # Initialize trajectories
    trajectories = []
    for problem in x["problem"]:
        for _ in range(config.n):
            trajectories.append({
                "prompt": problem,
                "current_text": "",
                "history": [],
                "completed": False,
                "all_scores": [],
                "smart_step": [],
                "gen_update": []
            })
            
    strategy_idx = STRATEGY_MAP.get(config.conf_strategy, 2)
    
    # Post-Hoc Loop
    for iterate_idx in range(config.num_iterations):
        active_trajs = [t for t in trajectories if not t["completed"]]
        if len(active_trajs) == 0:
            break
            
        logger.info(f"Iteration {iterate_idx}: Starting SLM generation for {len(active_trajs)} active trajectories...")
            
        convs = [build_conv(t["prompt"], t["current_text"], config.system_prompt) for t in active_trajs]
        add_gen_prompt = (iterate_idx == 0)
        cont_final_msg = (iterate_idx > 0)
        
        templated_convs = slm_tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=add_gen_prompt, continue_final_message=cont_final_msg
        )
        
        # 1. SLM Full Generation of the remainder
        responses = slm.generate(templated_convs, sampling_params_full, use_tqdm=False)
        logger.info("SLM generation complete. Chunking and evaluating steps...")
        
        fixes_needed = []
        for i, (traj, response) in enumerate(zip(active_trajs, responses)):
            output = response.outputs[0]
            token_ids = output.token_ids
            logprobs = output.logprobs
            
            # Chunking logic by \n\n
            step_token_ids = []
            step_logprobs = []
            steps = []
            
            for token_id, logprob_dict in zip(token_ids, logprobs):
                step_token_ids.append(token_id)
                step_logprobs.append(logprob_dict)
                text = slm_tokenizer.decode(step_token_ids)
                if text.endswith("\n\n"):
                    steps.append((text, step_logprobs))
                    step_token_ids = []
                    step_logprobs = []
            
            if step_token_ids:
                steps.append((slm_tokenizer.decode(step_token_ids), step_logprobs))
                
            # 2. Find Bottleneck (First bad step)
            first_bad_step_idx = -1
            bad_step_score = None
            
            for step_idx, (text, logprob_list) in enumerate(steps):
                conf_metrics = calculate_confidence_score(logprob_list)
                score = conf_metrics[strategy_idx]
                if needs_correction(score, config.threshold, config.conf_strategy):
                    first_bad_step_idx = step_idx
                    bad_step_score = score
                    break
                    
            if first_bad_step_idx != -1:
                # Add all good steps before the bottleneck to history
                for step_idx in range(first_bad_step_idx):
                    text, logprob_list = steps[step_idx]
                    traj["history"].append(text)
                    traj["current_text"] += text
                    conf_metrics = calculate_confidence_score(logprob_list)
                    traj["all_scores"].append(conf_metrics[strategy_idx])
                    
                bad_text, _ = steps[first_bad_step_idx]
                
                current_conv = build_conv(traj["prompt"], traj["current_text"], config.system_prompt)
                fixes_needed.append((traj, bad_text, bad_step_score, current_conv))
                
            else:
                # No bad steps found. The remainder is fully accepted.
                for step_idx in range(len(steps)):
                    text, logprob_list = steps[step_idx]
                    traj["history"].append(text)
                    traj["current_text"] += text
                    conf_metrics = calculate_confidence_score(logprob_list)
                    traj["all_scores"].append(conf_metrics[strategy_idx])
                    
                traj["completed"] = True
                    
        # 3. Surgical Intervention (LLM fixes the bad step)
        if fixes_needed:
            logger.info(f"Fixing {len(fixes_needed)} draft completions with LLM...")
            batch_prompts = []
            for traj, slm_text, slm_score, conv in fixes_needed:
                templated = llm_tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=add_gen_prompt, continue_final_message=cont_final_msg
                )
                batch_prompts.append(templated)
            
            llm_tokenizer.padding_side = "left"
            if llm_tokenizer.pad_token is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
            
            inputs = llm_tokenizer(batch_prompts, return_tensors="pt", padding=True).to(llm.device)
            
            # Chunk LLM generation to prevent OOM when many trajectories are triggered.
            llm_chunk_size = 8
            all_new_ids = []
            for chunk_start in range(0, inputs["input_ids"].shape[0], llm_chunk_size):
                chunk_input = {k: v[chunk_start:chunk_start+llm_chunk_size] for k, v in inputs.items()}
                with torch.no_grad():
                    chunk_ids = llm.generate(
                        **chunk_input,
                        stopping_criteria=[stopping_criteria],
                        generation_config=generation_config
                    )[:, chunk_input["input_ids"].shape[1]:]
                all_new_ids.append(chunk_ids)
                del chunk_input, chunk_ids
                torch.cuda.empty_cache()
            
            # Each chunk may produce sequences of different lengths → pad to max before cat.
            pad_id = llm_tokenizer.pad_token_id if llm_tokenizer.pad_token_id is not None else llm_tokenizer.eos_token_id
            max_len = max(c.shape[1] for c in all_new_ids)
            padded_chunks = []
            for c in all_new_ids:
                if c.shape[1] < max_len:
                    pad = torch.full((c.shape[0], max_len - c.shape[1]), pad_id, dtype=c.dtype, device=c.device)
                    c = torch.cat([c, pad], dim=1)
                padded_chunks.append(c)
            new_ids_all = torch.cat(padded_chunks, dim=0)
            del all_new_ids, padded_chunks
            
            batch_steps = llm_tokenizer.batch_decode(new_ids_all)
            
            for j, (traj, slm_text, slm_score, conv) in enumerate(fixes_needed):
                llm_step = batch_steps[j]
                
                if not llm_step or llm_step.strip() == "":
                    llm_step = "\n\n"
                
                # Append LLM step to history and return control to SLM in the next iteration
                traj["current_text"] += llm_step
                traj["history"].append(llm_step)
                traj["all_scores"].append(slm_score) 
                
                traj["smart_step"].append(iterate_idx)
                traj["gen_update"].append((slm_text, llm_step))
                
                if not llm_step.endswith("\n\n") and len(new_ids_all[j]) >= config.max_tokens:
                    traj["completed"] = True
            logger.info("LLM batch generation complete. Resuming SLM generation for remaining steps...")
                    
    logger.info("Batch completed. Selecting final predictions based on confidence scores...")
    
    # Format output
    completions = [[] for _ in range(len(x["problem"]))]
    agg_scores = [[] for _ in range(len(x["problem"]))]
    for i, problem in enumerate(x["problem"]):
        problem_trajs = [t for t in trajectories if t["prompt"] == problem]
        completions[i] = [t["current_text"] for t in problem_trajs]
        agg_scores[i] = [aggregate_scores(t["all_scores"], config.agg_strategy) if len(t["all_scores"]) > 0 else 0 for t in problem_trajs]
        
    pred = [comp[np.argmax(s)] for comp, s in zip(completions, agg_scores)]

    x["completions"] = completions
    x["pred"] = pred

    return x
