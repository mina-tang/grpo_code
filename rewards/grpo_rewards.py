import re

def mcq_reward(prompts, completions, answer):
    rewards = []
    for prompt, response, correct_answers in zip(prompts, completions, answer):
        # Normalize correct answers list
        correct_norm = [a.replace(" ", "").lower() for a in correct_answers]

        # Extract answer inside <answer> tags from response
        match = re.search(r"<answer>\s*(.*?)\s*<answer>", response, re.IGNORECASE)
        if not match:
            rewards.append(0.0)
            continue
        
        resp_ans = match.group(1).replace(" ", "").lower()
        
        # Check if extracted answer matches any of the correct answers
        reward = 1.0 if resp_ans in correct_norm else 0.0
        rewards.append(reward)

    return rewards


import re
from typing import List, Dict, Any

def compute_set_f1(predicted: List[str], ground_truth: List[str], alpha: float = 0.5) -> float:
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    pred_set = set(p.replace(" ", "").lower() for p in predicted)
    gt_set = set(t.replace(" ", "").lower() for t in ground_truth)
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    return alpha * recall + (1 - alpha) * precision

def compute_episode_reward(predicted_causes: List[str],
                           true_causes: List[str],
                           path_branch_matches: List[bool],
                           num_steps: int,
                           num_safety_skips: int,
                           abnormality_flags: List[bool] = None,
                           alpha: float = 0.5,
                           beta: float = 0.3,
                           step_cost: float = 0.05,
                           safety_penalty: float = 1.0) -> float:
    # 1. Diagnosis reward
    diag_reward = compute_set_f1(predicted_causes, true_causes, alpha=alpha)
    
    # 2. Path-shaping bonus
    path_bonus = beta * sum(path_branch_matches)
    
    # 3. Step cost
    cost = step_cost * num_steps
    
    # 4. Safety penalty
    safe_pen = safety_penalty * num_safety_skips
    
    # 5. Abnormality bonus (optional)
    abn_bonus = 0.0
    if abnormality_flags is not None:
        abn_bonus = 0.05 * sum(abnormality_flags)
    
    return diag_reward + path_bonus + abn_bonus - cost - safe_pen

def epoct_reward(prompts: List[str],
                 completions: List[str],
                 answers: List[Dict[str, Any]]) -> List[float]:
    """
    Compute episode rewards for e-POCT diagnostic completions.
    
    Args:
        prompts: List of prompt strings (signature compatibility).
        completions: List of completion strings. Each should embed a JSON-like dict:
                     {
                         "predicted_causes": List[str],
                         "path_branch_matches": List[bool],
                         "num_steps": int,
                         "num_safety_skips": int,
                         "abnormality_flags": List[bool] (optional)
                     }
        answers: List of dicts, each with key 'true_causes': List[str].
    
    Returns:
        List of float rewards.
    """
    rewards = []
    for completion, answer in zip(completions, answers):
        try:
            # Extract the JSON-like dict from the completion
            data_match = re.search(r"\{.*\}", completion, re.DOTALL)
            episode = eval(data_match.group(0))
            
            reward = compute_episode_reward(
                predicted_causes=episode.get("predicted_causes", []),
                true_causes=answer.get("true_causes", []),
                path_branch_matches=episode.get("path_branch_matches", []),
                num_steps=episode.get("num_steps", 0),
                num_safety_skips=episode.get("num_safety_skips", 0),
                abnormality_flags=episode.get("abnormality_flags", None)
            )
        except Exception:
            reward = 0.0
        rewards.append(reward)
    return rewards
