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

def find_box(pred_str: str):
    if "boxed" not in pred_str:
        return None
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a

def build_meta_conv(prompt: str, response: str | None) -> list[dict[str, str]]:
    meta_system_prompt = (
        "You are an advanced AI helping a smaller language model (SLM) solve a math problem. "
        "The SLM got stuck and we rolled back its incorrect reasoning. Below is the problem and the correct steps so far. "
        "Your task is to generate the SINGLE next step (the most critical mathematical leap) that will guide the SLM to finish the problem correctly. "
        "You MUST put this next step inside a LaTeX box like: \\boxed{your_step}."
    )
    user_content = f"Problem:\n{prompt}\n\nCorrect steps so far:\n{response if response else ''}\n\nPlease generate the next step and put it inside \\boxed{{}}."
    return [
        {"role": "system", "content": meta_system_prompt},
        {"role": "user", "content": user_content}
    ]

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
                
                current_conv = build_meta_conv(traj["prompt"], traj["current_text"])
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
                    conv, tokenize=False, add_generation_prompt=True, continue_final_message=False
                )
                batch_prompts.append(templated)
            
            llm_tokenizer.padding_side = "left"
            if llm_tokenizer.pad_token is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
            
            inputs = llm_tokenizer(batch_prompts, return_tensors="pt", padding=True).to(llm.device)
            
            # Single full-batch generation — requires H100 (80GB).
            # All triggered sequences generated in one parallel GPU call for maximum speed.
            with torch.no_grad():
                new_ids_all = llm.generate(
                    **inputs,
                    stopping_criteria=[stopping_criteria],
                    generation_config=generation_config
                )[:, inputs["input_ids"].shape[1]:]
            
            batch_steps = llm_tokenizer.batch_decode(new_ids_all, skip_special_tokens=True)
            
            for j, (traj, slm_text, slm_score, conv) in enumerate(fixes_needed):
                raw_llm_step = batch_steps[j]
                boxed = find_box(raw_llm_step)
                llm_step = boxed if boxed is not None else raw_llm_step
                
                if not llm_step.endswith("\n\n"):
                    llm_step += "\n\n"
                
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
                    
    logger.info("Batch completed. Selecting final predictions using weighted voting...")
    
    # Format output
    completions = [[] for _ in range(len(x["problem"]))]
    agg_scores = [[] for _ in range(len(x["problem"]))]
    pred = []
    
    for i, problem in enumerate(x["problem"]):
        problem_trajs = [t for t in trajectories if t["prompt"] == problem]
        completions[i] = [t["current_text"] for t in problem_trajs]
        agg_scores[i] = [
            aggregate_scores(t["all_scores"], config.agg_strategy) if len(t["all_scores"]) > 0 else 0.0
            for t in problem_trajs
        ]
        
        # Weighted voting: extract final answer from each completion,
        # group by answer, sum confidence scores per group, pick best.
        from sal.utils.qwen_math_parser import extract_answer
        
        answer_scores: dict[str, float] = {}
        answer_to_completion: dict[str, str] = {}
        
        for comp, score_val in zip(completions[i], agg_scores[i]):
            ans = extract_answer(comp, config.dataset_name).strip()
            if not ans:
                continue
            answer_scores[ans] = answer_scores.get(ans, 0.0) + score_val
            # Keep the completion with the highest individual score for this answer
            if ans not in answer_to_completion or score_val > answer_scores.get(ans + "__best_score__", -1e9):
                answer_to_completion[ans] = comp
                answer_scores[ans + "__best_score__"] = score_val

        if answer_scores:
            # Filter out the helper __best_score__ keys before finding argmax
            real_answers = {k: v for k, v in answer_scores.items() if not k.endswith("__best_score__")}
            best_answer = max(real_answers, key=real_answers.__getitem__)
            pred.append(answer_to_completion[best_answer])
        else:
            # Fallback: naive argmax on aggregate scores
            pred.append(completions[i][int(np.argmax(agg_scores[i]))])

    x["completions"] = completions
    x["pred"] = pred

    return x
