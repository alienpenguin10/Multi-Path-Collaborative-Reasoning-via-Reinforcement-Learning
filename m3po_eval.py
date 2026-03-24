###########################
# Step 4. LOAD AND TEST MODEL  #
###########################
import sys
import os
sys.path.insert(0, os.path.abspath("transformers/src"))
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import SYSTEM_PROMPT, build_prompt, extract_answer_from_model_output, extract_single_number, extract_last_number, prepare_dataset, set_random_seed
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


if __name__ == "__main__":
    # Determine the device: use GPU if available, else fallback to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the saved model and tokenizer
    saved_model_path = "outputs/baseline/trial_1_seed42"

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

    # Define test prompts
    eval_data = prepare_dataset("test")
    eval_data = eval_data[:len(eval_data)//4]  # 329 examples, matching training eval subset


    import json
    from datetime import datetime

    set_random_seed(BASE_SEED)
    num_gpus = torch.cuda.device_count()
    eval_batch_size = max(16, 8 * num_gpus)
    print(f"Using {num_gpus} GPU(s), eval batch size: {eval_batch_size}")

    print("\nFinal model evaluation after GRPO RL fine-tuning:")
    post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device, batch_size=eval_batch_size)
    print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")

    results = {
        "gating_type": "GRPO",
        "accuracy": post_grpo_accuracy,
        "eval_size": len(eval_data),
        "model_path": saved_model_path,
        "timestamp": datetime.now().isoformat(),
    }
    results_path = os.path.join(saved_model_path, "full_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")