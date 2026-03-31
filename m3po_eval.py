###########################
# Step 4. LOAD AND TEST MODEL  #
###########################
import sys
import os
sys.path.insert(0, os.path.abspath("transformers/src"))
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import SYSTEM_PROMPT, build_prompt, extract_answer_from_model_output, extract_single_number, extract_last_number, prepare_dataset, set_random_seed
from collections import Counter
import torch

BASE_SEED = 42
set_random_seed(BASE_SEED)

def check_correct(predicted, expected):
    """Check if predicted answer matches expected using multiple methods."""
    if predicted == expected:
        return True
    # Try single number matching
    pred_num = extract_single_number(str(predicted))
    exp_num = extract_single_number(str(expected))
    if pred_num is not None and exp_num is not None and pred_num == exp_num:
        return True
    # Try last number matching
    pred_num = extract_last_number(str(predicted))
    exp_num = extract_last_number(str(expected))
    return pred_num is not None and exp_num is not None and pred_num == exp_num


def evaluate_model(model, tokenizer, eval_examples, device, batch_size=16):
    """
    Evaluates the model on a set of examples using batched generation.

    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer for encoding inputs and decoding outputs.
        eval_examples (list): List of evaluation examples, each containing "prompt" and "answer".
        device: The device (CPU or GPU) to run evaluation on.
        batch_size (int): Number of examples to process in each batch.

    Returns:
        float: The accuracy percentage (correct predictions / total examples * 100).
    """
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "="*50)
    print("EVALUATION ON", total, "EXAMPLES")
    print("="*50)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = eval_examples[batch_start:batch_end]
        prompts = [ex["prompt"] for ex in batch]
        expected_answers = [ex["answer"] for ex in batch]

        print(f"Evaluating batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} "
              f"(examples {batch_start+1}-{batch_end}/{total})...", end=" ", flush=True)

        # Tokenize batch with left-padding for generation
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=False,
                generation_config=None,
            )
        print("Done", flush=True)

        # Decode and score each response in the batch
        for i, (output_ids, expected) in enumerate(zip(outputs, expected_answers)):
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            try:
                predicted = extract_answer_from_model_output(response)
                is_correct = check_correct(predicted, expected)
                if is_correct:
                    correct += 1
                print(expected)
                print(predicted)
                print("\nCorrect:", "✓" if is_correct else "✗")
            except Exception as e:
                print(f"\nFailed to parse output for example {batch_start + i + 1}: {e}")

    accuracy = (correct / total) * 100
    model.train()
    return accuracy


def evaluate_model_self_consistency(model, tokenizer, eval_examples, device, batch_size=8, num_samples=8, temperature=0.7):
    """
    Evaluates the model using self-consistency (majority voting over K sampled completions).

    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer for encoding inputs and decoding outputs.
        eval_examples (list): List of evaluation examples, each containing "prompt" and "answer".
        device: The device (CPU or GPU) to run evaluation on.
        batch_size (int): Number of examples to process in each batch.
        num_samples (int): Number of completions to sample per example (K).
        temperature (float): Sampling temperature.

    Returns:
        float: The accuracy percentage using majority voting.
    """
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "="*50)
    print(f"SELF-CONSISTENCY EVALUATION ON {total} EXAMPLES (K={num_samples}, T={temperature})")
    print("="*50)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = eval_examples[batch_start:batch_end]
        prompts = [ex["prompt"] for ex in batch]
        expected_answers = [ex["answer"] for ex in batch]
        current_batch_size = len(prompts)

        print(f"Evaluating batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} "
              f"(examples {batch_start+1}-{batch_end}/{total})...", end=" ", flush=True)

        # Repeat each prompt num_samples times for batched generation
        repeated_prompts = [p for p in prompts for _ in range(num_samples)]
        inputs = tokenizer(repeated_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=False,
                generation_config=None,
            )
        print("Done", flush=True)

        # Decode all outputs and group by example
        all_responses = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]

        for i, expected in enumerate(expected_answers):
            # Get the K responses for this example
            sample_responses = all_responses[i * num_samples : (i + 1) * num_samples]
            # Extract answers from each sample
            extracted = []
            for resp in sample_responses:
                try:
                    ans = extract_answer_from_model_output(resp)
                    if ans is not None:
                        extracted.append(str(ans).strip())
                except Exception:
                    pass

            if not extracted:
                print(f"  Example {batch_start + i + 1}: no valid answers extracted from {num_samples} samples")
                continue

            # Majority vote: normalize by trying numeric extraction for consistent comparison
            normalized = []
            for ans in extracted:
                num = extract_single_number(ans)
                if num is not None:
                    normalized.append(str(num))
                else:
                    num = extract_last_number(ans)
                    if num is not None:
                        normalized.append(str(num))
                    else:
                        normalized.append(ans)

            vote_counts = Counter(normalized)
            majority_answer = vote_counts.most_common(1)[0][0]
            is_correct = check_correct(majority_answer, expected)
            if is_correct:
                correct += 1
            print(f"  Expected: {expected} | Majority: {majority_answer} ({vote_counts.most_common(1)[0][1]}/{len(normalized)} votes) | {'✓' if is_correct else '✗'}")

    accuracy = (correct / total) * 100
    model.train()
    return accuracy


if __name__ == "__main__":
    # Determine the device: use GPU if available, else fallback to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 

    # Load the saved model and tokenizer
    saved_model_path = "outputs/kl_divergence/trial_1_seed123" #kl_divergence

    # Load the model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        saved_model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Model loaded successfully!")

    tokenizer = AutoTokenizer.from_pretrained(saved_model_path, padding_side="left", fix_mistral_regex=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Define test prompts — use full test set for lower variance
    eval_data = prepare_dataset("test")

    import json
    from datetime import datetime

    set_random_seed(BASE_SEED)
    num_gpus = torch.cuda.device_count()
    eval_batch_size = max(16, 8 * num_gpus)
    print(f"Using {num_gpus} GPU(s), eval batch size: {eval_batch_size}")

    # --- Greedy evaluation ---
    print("\n--- Greedy Evaluation ---")
    greedy_accuracy = evaluate_model(model, tokenizer, eval_data, device, batch_size=eval_batch_size)
    print(f"Greedy Accuracy: {greedy_accuracy:.2f}% ({len(eval_data)} examples)")

    # --- Self-consistency evaluation ---
    set_random_seed(BASE_SEED)
    num_samples = 5
    sc_temperature = 0.7
    # Smaller batch size since each example generates num_samples completions
    sc_batch_size = max(2, eval_batch_size // num_samples)
    print(f"\n--- Self-Consistency Evaluation (K={num_samples}, T={sc_temperature}) ---")
    sc_accuracy = evaluate_model_self_consistency(
        model, tokenizer, eval_data, device,
        batch_size=sc_batch_size, num_samples=num_samples, temperature=sc_temperature,
    )
    print(f"Self-Consistency Accuracy: {sc_accuracy:.2f}% ({len(eval_data)} examples)")

    # --- Save results ---
    results = {
        "model_path": saved_model_path,
        "eval_size": len(eval_data),
        "greedy_accuracy": greedy_accuracy,
        "self_consistency_accuracy": sc_accuracy,
        "self_consistency_K": num_samples,
        "self_consistency_temperature": sc_temperature,
        "timestamp": datetime.now().isoformat(),
    }
    results_path = os.path.join(saved_model_path, "full_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"\nSummary: Greedy={greedy_accuracy:.2f}%, Self-Consistency(K={num_samples})={sc_accuracy:.2f}%")