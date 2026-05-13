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
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores, calculate_confidence_score, STRATEGY_MAP, needs_correction
from transformers.generation.stopping_criteria import StopStringCriteria
from transformers import AutoTokenizer

STRATEGY_MAP = {
    "likelihood": 0,
    "likelihood_mean": 1,
    "probs_mean": 2,
    "entropy": 3,
    "top_2_diff": 4,
    "mean_least_3": 5,
}

def needs_correction(score, threshold, strategy):
    if strategy == "entropy":
        return score > threshold
    elif strategy == "mean_least_3":
        return score > threshold
    elif strategy == "top_2_diff":
        return score < threshold
    else: # probs_mean, likelihood, etc.
        return score < threshold

def convert_to_chat_template(problem_str, partial_completion: None, config: Config, tokenizer: AutoTokenizer):
    if partial_completion is None:
        conv = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": problem_str}
        ]
        return tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    else:
        conv = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": problem_str},
            {"role": "assistant", "content": partial_completion}
        ]
        return tokenizer.apply_chat_template(conv, tokenize=False, continue_final_message=True)
        

def smart_best_of_n_conf(x, config: Config, slm: LLM, prm: PRM, llm: None):
    tokenizer = slm.get_tokenizer()
    tokenizer.padding_side = "left"
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
        
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,
        logprobs=10,
    )
    
    templated_convs = [convert_to_chat_template(problem, None, config, tokenizer) for problem in x["problem"]]
    templated_convs = [c for conv in templated_convs for c in [conv] * config.n]

    completions = [[] for _ in range(len(x["problem"]))]
    completion_tokens = [[] for _ in range(len(x["problem"]))]
    
    responses = slm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    
    strategy_idx = STRATEGY_MAP.get(config.conf_strategy, 2)
    
    for i in range(len(completions)):
        batch_responses = responses[i * config.n : (i + 1) * config.n]
        completions[i] = [r.outputs[0].text for r in batch_responses]
        completion_tokens[i] = [len(r.outputs[0].token_ids) for r in batch_responses]

    stopping_criteria = StopStringCriteria(stop_strings="\n\n", tokenizer=tokenizer)
    
    fixes_needed = []
    for problem_idx, problem_completions in enumerate(completions):
        for candidate_idx, text in enumerate(problem_completions):
            # For best_of_n_smart_conf, we calculate the confidence score of the entire generation
            # or we could split it into steps. Given the context, let's treat the whole generation.
            logprobs = responses[problem_idx * config.n + candidate_idx].outputs[0].logprobs
            score = calculate_confidence_score(logprobs)[strategy_idx]
            if needs_correction(score, config.threshold, config.conf_strategy):
                fixes_needed.append((problem_idx, candidate_idx))
    
    if fixes_needed:
        print(f"Fixing {len(fixes_needed)} draft completions using {config.conf_strategy}")
        llm_inputs = []
        for problem_idx, candidate_idx in fixes_needed:
            # For simplicity, we fix from the beginning if confidence is low
            templated_conv = convert_to_chat_template(x["problem"][problem_idx], None, config, tokenizer)
            llm_inputs.append(templated_conv)
            
        input_ids = tokenizer(llm_inputs, return_tensors="pt", padding=True).to(llm.device)
        new_ids = llm.generate(
            input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_tokens,
        )[:, input_ids["input_ids"].shape[1]:]
        new_steps = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        
        for i, (problem_idx, candidate_idx) in enumerate(fixes_needed):
            completions[problem_idx][candidate_idx] = new_steps[i]

    # Finally, score everything with PRM to select the best one
    prm_scores = prm.score(x["problem"], completions, config.prm_batch_size)
    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score] for score in prm_scores
    ]

    pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]

    x["completions"] = completions
    x["pred"] = pred

    return x
