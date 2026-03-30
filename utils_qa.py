"""
Part 1: Basic Setup and Imports
"""

# Import necessary libraries
import os
import random
import re
import string
import numpy as np

# PyTorch and related libraries for deep learning
import torch

# Hugging Face libraries for transformer models
from datasets import load_dataset


def set_random_seed(seed: int = 42):
    """Set the random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def get_next_trial_number(base_dir, gating_name):
    """Find the next available trial number for this gating type."""
    gating_dir = os.path.join(base_dir, gating_name)
    if not os.path.exists(gating_dir):
        return 1
    existing_trials = [
        int(d.replace("trial_", ""))
        for d in os.listdir(gating_dir)
        if d.startswith("trial_") and d.replace("trial_", "").isdigit()
    ]
    return max(existing_trials, default=0) + 1


"""
Part 2: Data Formatting and Answer Extraction
"""
SYSTEM_PROMPT = """
Answer the following question. Think through your reasoning step by step, then give your final answer.

Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def normalize_answer(text):
    """
    Normalize an answer string for comparison.

    Lowercases, removes articles (a/an/the), strips punctuation,
    and collapses whitespace. Follows standard TriviaQA evaluation.
    """
    if text is None:
        return ""
    text = text.lower()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    text = ' '.join(text.split())
    return text


def extract_answer_from_model_output(text):
    """
    Extracts the value from the last <answer> tag in the text.

    Returns str or None.
    """
    parts = text.split("<answer>")
    if len(parts) < 2:
        return None
    last_part = parts[-1]
    if "</answer>" not in last_part:
        return None
    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer


"""
Part 3: Dataset Preparation
"""
def prepare_dataset(split="train", max_examples=10000):
    """
    Load and prepare the TriviaQA dataset (rc.nocontext config).

    Args:
        split (str): "train" or "validation".
        max_examples (int): Maximum number of examples to load.

    Returns:
        list: Each element is {"prompt": str, "answers": list[str]}
              where answers is a list of normalized acceptable strings.
    """
    hf_split = "validation" if split == "test" else split
    data = load_dataset("trivia_qa", "rc.nocontext", split=hf_split)
    if max_examples is not None and len(data) > max_examples:
        data = data.select(range(max_examples))
    formatted_data = []
    for example in data:
        prompt_str = build_prompt([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]}
        ])
        # Collect all normalized acceptable answers from aliases
        raw_aliases = example["answer"].get("normalized_aliases") or example["answer"].get("aliases") or []
        # Also include the canonical value
        canonical = example["answer"].get("normalized_value") or example["answer"].get("value") or ""
        all_answers = list({normalize_answer(a) for a in raw_aliases + [canonical] if a})
        formatted_data.append({
            "prompt": prompt_str,
            "answers": all_answers,
        })
    return formatted_data


def build_prompt(messages):
    """Build a single prompt string from a list of messages."""
    return "\n".join([msg["content"].strip() for msg in messages])


"""
Part 4: Evaluation Functions
"""
def evaluate_model(model, tokenizer, eval_examples, device):
    """
    Evaluates the model on a set of examples and prints detailed results.

    Each example must have "prompt" and "answers" (list of normalized strings).

    Returns:
        float: Accuracy percentage.
    """
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "=" * 50)
    print("EVALUATION ON", total, "EXAMPLES")
    print("=" * 50)

    for example in eval_examples:
        full_prompt = example["prompt"]
        expected_answers = example["answers"]  # list of normalized strings

        inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id,
                early_stopping=False,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            predicted = extract_answer_from_model_output(response)
            is_correct = normalize_answer(predicted) in set(expected_answers)
            if is_correct:
                correct += 1
            print("\nCorrect:", "✓" if is_correct else "✗")
        except Exception as e:
            print("\nFailed to parse model output for prompt:")
            print(full_prompt)
            print("Error:", e)
            print("-" * 50)

    accuracy = (correct / total) * 100
    model.train()
    return accuracy


"""
Part 5: Reward Functions
"""
def correctness_reward(prompts, completions, answer, **kwargs):
    """
    Assigns a reward based on the correctness of the model's answer.

    Args:
        prompts (list): List of input prompts.
        completions (list): List of model completions.
        answer (list): List of expected answer sets (each element is a list of
                       normalized acceptable answer strings).

    Returns:
        list: 2.0 for a correct answer, 0.0 otherwise.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer_from_model_output(r) for r in responses]
    rewards = []
    for r, ans_set in zip(extracted, answer):
        norm_pred = normalize_answer(r)
        if norm_pred and norm_pred in set(ans_set):
            rewards.append(2.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions, **kwargs):
    """
    Assigns a reward for adhering to the desired XML format.

    0.2 points each for <reasoning>, </reasoning>, <answer>, </answer> tags.
    Maximum score: 0.8.
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.2
        if "</reasoning>" in response: score += 0.2
        if "<answer>" in response: score += 0.2
        if "</answer>" in response: score += 0.2
        rewards.append(score)
    return rewards


def eos_reward(completions, **kwargs):
    """
    Assigns a reward for properly terminating generation after </answer> tag.

    0.5 for clean stop, 0.4 for <20 chars after, 0.2 for <50 chars, 0.0 otherwise.
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        if "</answer>" not in response:
            rewards.append(0.0)
            continue
        last_answer_pos = response.rfind("</answer>")
        text_after_answer = response[last_answer_pos + len("</answer>"):].strip()
        if len(text_after_answer) == 0:
            rewards.append(0.5)
        elif len(text_after_answer) < 20:
            rewards.append(0.4)
        elif len(text_after_answer) < 50:
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


def combined_reward(prompts, completions, answer):
    """
    Combines correctness, format, and EOS rewards.

    Args:
        prompts (list[str]): List of prompt texts.
        completions (list[list[dict]]): List of completion dictionaries.
        answer (list[list[str]]): List of expected answer sets (each a list of
                                  normalized acceptable strings).

    Returns:
        list[float]: Combined rewards. Range: 0.0 to 3.3.
    """
    correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)
    eos_scores = eos_reward(completions=completions)
    return [c + f + e for c, f, e in zip(correctness_scores, format_scores, eos_scores)]
