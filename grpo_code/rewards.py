import math
import os
import re
from pathlib import Path

import grpo_code
from grpo_code.executor import execute_tasks

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
MAX_PROCESSES = max(1, int(os.environ.get("MAX_PROCESSES", 1)) // WORLD_SIZE)
TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", 1))
WASM_PATH = os.environ.get(
    "WASM_PATH", Path(grpo_code.__file__).parent.parent / "wasm" / "python-3.12.0.wasm"
)
FUEL = int(os.environ.get("FUEL", 1_000_000_000))

if not os.path.exists(WASM_PATH):
    raise FileNotFoundError(f"WASM file not found at {WASM_PATH}")


def extract_xml_answer(text: str) -> str:
    """
    Extract text between <answer> and </answer> tags.

    Args:
        text (str): The text to extract the answer from.
    Returns:
        str: The answer extracted from the text. "" if no answer is found.

    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.S)
    return match.group(1).strip() if match else ""


def code_execution_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Reward function for code execution.

    Args:
        completions (list[list[dict]]): The predicted code completions to execute. This takes the format
            [
                [
                    {"role": "user", "content": "<reasoning>...</reasoning><answer>...</answer>"}
                ]
            ]
    Returns:
        list[float]: The rewards for the completions. Each completion is rewarded 0.5 if the code executes, -0.25 otherwise.
    """
    model_answers = [
        extract_xml_answer(completion[0]["content"]) for completion in completions
    ]
    task_results = execute_tasks(
        model_answers, MAX_PROCESSES, WASM_PATH, FUEL, TASK_TIMEOUT
    )
    return [0.5 if result == 1.0 else -0.25 for result in task_results]


def answer_execution_reward_func(
    completions: list[list[dict]], answers: list[list[str]], **kwargs
) -> list[float]:
    """
    Reward function for answer execution.

    Args:
        completions (list[list[dict]]): The predicted code completions to execute. This takes the format
            [
                [
                    {"role": "user", "content": "<reasoning>...</reasoning><answer>...</answer>"}
                ]
            ]
        answers (list[list[str]]): The expected answers to the code completions. These take the form of executable
            assert statements, e.g.
            [
                [
                    "assert foo(1) == 2",
                    "assert foo(2) == 3",
                ]
            ]
    Returns:
        list[float]: The accuracy rewards for the completions. Each completion is rewarded
            (accuracy)^3 * 2, where accuracy is the proportion of test cases that pass.
    """
    model_answers = [
        extract_xml_answer(completion[0]["content"]) for completion in completions
    ]
    tasks = []
    test_indices = []
    for i, (code, tests) in enumerate(zip(model_answers, answers)):
        for test in tests:
            tasks.append(code + "\n" + test)
            test_indices.append(i)

    task_results = execute_tasks(tasks, MAX_PROCESSES, WASM_PATH, FUEL, TASK_TIMEOUT)

    completion_results = {}
    for idx, result in zip(test_indices, task_results):
        if idx not in completion_results:
            completion_results[idx] = []
        completion_results[idx].append(result)

    rewards = []
    for i in range(len(completions)):
        if i in completion_results:
            test_results = completion_results[i]
            accuracy = sum(test_results) / len(test_results)
            reward = math.pow(accuracy, 3) * 2
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function for soft format checking.

    Args:
        completions (list[list[dict]]): The predicted code completions to execute. This takes the format
            [
                [
                    {"role": "user", "content": content}
                ]
            ]
    Returns:
        list[float]: The rewards for the completions. Each completion is rewarded 0.25 if the format is correct, 0.0 otherwise.
    """

    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        if re.match(
            r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>.*", response, re.S
        ):
            rewards.append(0.25)
        else:
            rewards.append(0.0)
    return rewards
