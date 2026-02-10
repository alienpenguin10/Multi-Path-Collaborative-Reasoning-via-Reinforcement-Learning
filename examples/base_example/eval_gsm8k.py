# Fix shadowing of installed packages by local directories
import sys
import os
sys.path.insert(0, os.path.abspath("unsloth"))
sys.path.insert(0, os.path.abspath("transformers/src"))
sys.path.insert(0, os.path.abspath("trl"))

import json
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

from utils import *


def evaluate_model(
    model_name: str,
    temperature: float,
    batch_size: int = 4,
    num_samples: int = None,
    save_results: bool = True,
    output_dir: str = None,
):
    # Load model and tokenizer from HuggingFace
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('openai/gsm8k', 'main')['test']
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples")

    results = []
    correct = 0
    total = 0

    progress_bar = tqdm(
        total=total_samples,
        desc="Processing samples",
        unit="examples",
        dynamic_ncols=True,
    )
    progress_bar.set_postfix({'acc': '0.00%', 'correct': '0'})

    # Process samples in batches
    for i in range(0, total_samples, batch_size):
        batch_data = dataset[i:i + batch_size]
        current_batch_size = len(batch_data['question'])

        # Prepare prompts using the same format as training
        prompts = [
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': q.strip()},
            ]
            for q in batch_data['question']
        ]

        # Convert chat prompts to the required format
        formatted_prompts = [
            tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]

        prompt_inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.to(model.device)
        prompt_mask = prompt_mask.to(model.device)
        prompt_length = prompt_ids.size(1)

        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                prompt_ids, attention_mask=prompt_mask, 
                generation_config=GenerationConfig(
                    do_sample=True,  # for temperature, top-k, etc.
                    temperature=temperature,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.pad_token_id,
                ),
            )

        # Process each generated response
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)

            # Extract the generated answer using XML tags
            extracted = extract_from_response(response)
            generated_answer = process_gsm8k_answer(extracted)
            true_answer = extract_hash_answer(batch_data['answer'][j])
            true_answer = process_gsm8k_answer(true_answer)
            print(generated_answer, true_answer, generated_answer == true_answer)

            # Store the result
            result = {
                'question': batch_data['question'][j],
                'true_answer': true_answer,
                'generated_answer': generated_answer,
                'full_response': response,
                'correct': generated_answer == true_answer
            }
            results.append(result)

            if generated_answer == true_answer:
                correct += 1
            total += 1

        progress_bar.update(current_batch_size)
        progress_bar.set_postfix({
            'acc': f'{(correct/total)*100:.2f}%',
            'correct': f'{correct}/{total}',
        })

    progress_bar.close()
    accuracy = correct / total if total > 0 else 0
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'model_name': model_name,
        'timestamp': datetime.now().isoformat()
    }

    if save_results and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "eval_results.json")
        with open(save_path, 'w') as f:
            json.dump({'metrics': metrics, 'results': results}, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="wzzju/Qwen2.5-1.5B-GRPO-GSM8K")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    args = parser.parse_args()

    print(f"Model: {args.model_name}, Temperature: {args.temperature}")
    
    metrics = evaluate_model(
        model_name=args.model_name,
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        save_results=True,
        output_dir=args.output_dir,
    )
    
    print(f"\n{'='*50}")
    print(f"Final Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")
    print(f"{'='*50}")