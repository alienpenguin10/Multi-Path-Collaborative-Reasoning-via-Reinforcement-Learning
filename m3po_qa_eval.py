###########################
# Step 4. LOAD AND TEST MODEL  #
###########################
import sys
import os
sys.path.insert(0, os.path.abspath("transformers/src"))
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils_qa import (
    SYSTEM_PROMPT, build_prompt, extract_answer_from_model_output,
    normalize_answer, prepare_dataset, set_random_seed,
)
import torch

BASE_SEED = 42
set_random_seed(BASE_SEED)


def check_correct(predicted, expected_answers):
    """Check if predicted answer matches any of the expected answers (normalized)."""
    return normalize_answer(predicted) in set(expected_answers)


def evaluate_model(model, tokenizer, eval_examples, device, batch_size=16):
    """
    Evaluates the model on a set of examples using batched generation.

    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer for encoding inputs and decoding outputs.
        eval_examples (list): List of evaluation examples, each containing
                              "prompt" and "answers" (list of normalized strings).
        device: The device (CPU or GPU) to run evaluation on.
        batch_size (int): Number of examples to process in each batch.

    Returns:
        float: The accuracy percentage (correct predictions / total examples * 100).
    """
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "=" * 50)
    print("EVALUATION ON", total, "EXAMPLES")
    print("=" * 50)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = eval_examples[batch_start:batch_end]
        prompts = [ex["prompt"] for ex in batch]
        expected_answers_list = [ex["answers"] for ex in batch]

        print(f"Evaluating batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} "
              f"(examples {batch_start+1}-{batch_end}/{total})...", end=" ", flush=True)

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

        for i, (output_ids, expected_answers) in enumerate(zip(outputs, expected_answers_list)):
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            try:
                predicted = extract_answer_from_model_output(response)
                is_correct = check_correct(predicted, expected_answers)
                if is_correct:
                    correct += 1
                print(expected_answers[0] if expected_answers else "")
                print(predicted)
                print("\nCorrect:", "✓" if is_correct else "✗")
            except Exception as e:
                print(f"\nFailed to parse output for example {batch_start + i + 1}: {e}")

    accuracy = (correct / total) * 100
    model.train()
    return accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    saved_model_path = "outputs/bahdanau/trial_1_seed42"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        saved_model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Model loaded successfully!")

    tokenizer = AutoTokenizer.from_pretrained(saved_model_path, padding_side="left", fix_mistral_regex=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load TriviaQA validation split (up to 2000 examples)
    eval_data = prepare_dataset("validation", max_examples=2000)

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
        "dataset": "trivia_qa/rc.nocontext",
        "accuracy": post_grpo_accuracy,
        "eval_size": len(eval_data),
        "model_path": saved_model_path,
        "timestamp": datetime.now().isoformat(),
    }
    results_path = os.path.join(saved_model_path, "full_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
