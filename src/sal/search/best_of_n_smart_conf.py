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
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, GenerationConfig
from transformers.generation.stopping_criteria import StopStringCriteria

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores, calculate_confidence_score, STRATEGY_MAP, needs_correction
from sal.search.utils import build_conv

_TOKENIZER_CACHE = {}

def get_cached_tokenizer(model_path):
    if model_path not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model_path] = AutoTokenizer.from_pretrained(model_path)
    return _TOKENIZER_CACHE[model_path]

def smart_best_of_n_conf(x, config: Config, slm: LLM, prm: PRM, llm: None):
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

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,
        logprobs=10,
        stop=["\n\n"],
        include_stop_str_in_output=True,
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
    
    # Step-by-step generation loop
    # Removed tqdm from inner loop to prevent Colab output freezing
    for iterate_idx in range(config.num_iterations):
        active_trajs = [t for t in trajectories if not t["completed"]]
        if len(active_trajs) == 0:
            break
            
        if iterate_idx == config.num_iterations - 1:
            # Generate to EOS on the final step
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
                logprobs=10,
            )
            
        convs = [build_conv(t["prompt"], t["current_text"], config.system_prompt) for t in active_trajs]
        add_gen_prompt = (iterate_idx == 0)
        cont_final_msg = (iterate_idx > 0)
        
        templated_convs = slm_tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=add_gen_prompt, continue_final_message=cont_final_msg
        )
        
        # 1. SLM Step Generation
        responses = slm.generate(templated_convs, sampling_params, use_tqdm=False)
        
        fixes_needed = []
        for i, (traj, response) in enumerate(zip(active_trajs, responses)):
            output = response.outputs[0]
            next_text = output.text
            stop_reason = output.stop_reason
            
            # 2. Immediate Confidence Check
            conf_metrics = calculate_confidence_score(output.logprobs)
            score = conf_metrics[strategy_idx]
            
            if needs_correction(score, config.threshold, config.conf_strategy):
                fixes_needed.append((i, traj, next_text, score, convs[i]))
            else:
                # Accept SLM step
                traj["current_text"] += next_text
                traj["history"].append(next_text)
                traj["all_scores"].append(score)
                
                # Check completion
                if stop_reason == "EOS" or stop_reason == "length" or next_text == "":
                    traj["completed"] = True
                elif len(slm_tokenizer.encode(" ".join(traj["history"]))) > 2048:
                    traj["completed"] = True
                    
        # 3. Surgical Intervention (LLM generates replacement steps in BATCH)
        if fixes_needed:
            # Prepare all prompts for batching
            batch_prompts = []
            for i, traj, slm_text, slm_score, conv in fixes_needed:
                templated = llm_tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=add_gen_prompt, continue_final_message=cont_final_msg
                )
                batch_prompts.append(templated)
            
            # Batch Tokenize
            llm_tokenizer.padding_side = "left"
            inputs = llm_tokenizer(batch_prompts, return_tensors="pt", padding=True).to(llm.device)
            
            # Batch Generate
            with torch.no_grad():
                new_ids_all = llm.generate(
                    **inputs,
                    stopping_criteria=[stopping_criteria],
                    generation_config=generation_config
                )[:, inputs["input_ids"].shape[1]:]
            
            batch_steps = llm_tokenizer.batch_decode(new_ids_all)
            
            # Distribute results
            for j, (i, traj, slm_text, slm_score, conv) in enumerate(fixes_needed):
                llm_step = batch_steps[j]
                
                # Ensure the LLM didn't just return an empty string
                if not llm_step or llm_step.strip() == "":
                    llm_step = "\n\n"
                
                # 4. Append LLM step to history and return control to SLM
                traj["current_text"] += llm_step
                traj["history"].append(llm_step)
                traj["all_scores"].append(slm_score) # Keep the original low score for analysis
                
                traj["smart_step"].append(iterate_idx)
                traj["gen_update"].append((slm_text, llm_step))
                
                if llm_step.endswith("\n\n"):
                    pass # continues loop seamlessly
                elif len(new_ids_all[j]) >= config.max_tokens:
                    traj["completed"] = True
                else:
                    traj["completed"] = True
                    
    # Format output
    completions = [[] for _ in range(len(x["problem"]))]
    for i, problem in enumerate(x["problem"]):
        # Find all trajectories for this problem
        problem_trajs = [t for t in trajectories if t["prompt"] == problem]
        completions[i] = [t["current_text"] for t in problem_trajs]
        
    # 5. Final PRM Evaluation
    prm_scores = prm.score(x["problem"], completions, config.prm_batch_size)
    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score] for score in prm_scores
    ]

    pred = [comp[np.argmax(s)] for comp, s in zip(completions, agg_scores)]

    x["completions"] = completions
    x["pred"] = pred

    return x
