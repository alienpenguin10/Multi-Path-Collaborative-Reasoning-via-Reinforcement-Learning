import unsloth
from unsloth import FastLanguageModel

import os
import json
import torch
import tempfile
from datetime import datetime
from datasets import load_dataset
from transformers import GenerationConfig
from tqdm import tqdm
from huggingface_hub import HfApi

from utils import *


def evaluate_model(
    model_path: str,
    temperature: float,
    is_inference: bool,
    batch_size: int = 4,
    num_samples: int = None,
    save_results: bool = True,
    output_repo: str = None,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 1024,
        load_in_4bit = False,
        fast_inference = False,
    )
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.for_inference(model)

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
            formatted_prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.to(model.device)
        prompt_mask = prompt_mask.to(model.device)
        prompt_length = prompt_ids.size(1)

        # Generate responses
        if is_inference:  # Use greedy decoding for evaluation
            outputs = model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                generation_config=GenerationConfig(
                    do_sample=False,  # Greedy decoding
                    max_new_tokens=512,
                ),
            )
        else:  # Use sampling (for analysis/diversity)
            outputs = model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                generation_config=GenerationConfig(
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=512,
                ),
            )

        # Process each generated response
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[prompt_length:])
            response = response.split(
                tokenizer.special_tokens_map['eos_token']
            )[0]

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
        'model_path': model_path,
        'timestamp': datetime.now().isoformat()
    }

    if save_results:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"eval_results_{timestamp}.json"
        data = {'metrics': metrics, 'results': results}

        # Determine where to save: output_repo overrides, then check if local path
        target_repo = output_repo if output_repo else model_path

        if os.path.isdir(target_repo):
            # Local path - save directly
            save_path = os.path.join(target_repo, filename)
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\nResults saved to {save_path}")
        else:
            # HuggingFace repo - upload
            api = HfApi()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(data, f, indent=2)
                temp_path = f.name

            api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=filename,
                repo_id=target_repo,
                repo_type="model",
            )
            os.unlink(temp_path)
            print(f"\nResults uploaded to https://huggingface.co/{target_repo}/{filename}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8k")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--greedy", type=str, default="True", help="Use greedy decoding (True/False)")
    parser.add_argument("--output_repo", type=str, default=None, help="HuggingFace repo to upload results (defaults to checkpoint_path)")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature (only used for metadata)")
    parser.add_argument("--force_eval", action="store_true", help="Run evaluation even if results exist")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (None for all)")
    args = parser.parse_args()

    model_path = args.checkpoint_path
    temperature = args.temperature
    is_greedy = args.greedy.lower() in ["true", "1", "yes"]

    # Try to extract temperature from model path if it contains -temp
    if '-temp' in model_path:
        try:
            temperature = float(model_path.split('-temp')[-1].split('/')[0].split('-')[0])
        except ValueError:
            pass

    print(f"Model: {model_path}, Temperature: {temperature}, Greedy: {is_greedy}")

    # Check if results already exist (only for local paths)
    should_run = args.force_eval
    if not should_run:
        if os.path.isdir(model_path):
            existing_results = [f for f in os.listdir(model_path) if f.startswith('eval_results')]
            should_run = len(existing_results) == 0
        else:
            should_run = True  # HuggingFace model, always run

    if should_run:
        print(f"Starting GSM8k evaluation on {model_path}")
        metrics = evaluate_model(
            model_path=model_path,
            temperature=temperature,
            is_inference=is_greedy,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            save_results=True,
            output_repo=args.output_repo,
        )
    else:
        print(f"Evaluation results already exist for {model_path}. Use --force_eval to re-run.")