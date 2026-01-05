import os
import sys

# Fix shadowing of installed packages by local directories
sys.path.insert(0, os.path.abspath("unsloth"))
sys.path.insert(0, os.path.abspath("transformers/src"))
sys.path.insert(0, os.path.abspath("trl"))

os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"


from unsloth import FastLanguageModel
import json
import torch
import re
from datetime import datetime
from datasets import load_dataset
from transformers import GenerationConfig
from tqdm import tqdm
import argparse

# ==========================================
# Utility Functions (Inlined)
# ==========================================

ANSWER_START = "####"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "final answer is provided after the " + ANSWER_START + " tag, i.e., "
    "{reasoning process} " + ANSWER_START + " {answer}."
)

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        try:
            n = eval(n)
        except:
            return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip("0")
        n = int(n.rstrip(".")) if n.endswith(".") else float(n)
        n=str(n)
        return n

def process_gsm8k_answer(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    pred = [delete_extra_zero(s.replace(",", ""))
            for s in re.findall(r"-?\d+/?\.?\d*", pred)]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1].rstrip(".").rstrip("/")
    return pred

def extract_from_response(text: str) -> str:
    try:
        answer = text.split(ANSWER_START)[-1].strip()
        if answer.endswith("."):
            answer = answer[:-1].strip()
        return answer
    except IndexError:
        return ""

# ==========================================
# Evaluation Logic
# ==========================================

def evaluate_model(
    model_path: str,
    temperature: float,
    is_inference: bool,
    batch_size: int = 4,
    num_samples: int = None,
    save_results: bool = True,
):
    print(f"Loading merged model from: {model_path}")
    
    # Load model and tokenizer via Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 1024,
        load_in_4bit = False,
        fast_inference = False, # Using False as safe default for merged models
    )
    model.answer_start = ANSWER_START

    
    
    # Ensure pad token is set
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Attempt to load missing thinking_residual weights from checkpoint-1245 if they exist
    # This addresses the issue where the merged model lost these specific weights
    checkpoint_path = "outputs/grpo-Qwen2.5-1.5B-Instruct-gsm8k-20260101_152454/checkpoint-1245/adapter_model.safetensors"
    if os.path.exists(checkpoint_path):
        print(f"Found recovery checkpoint at {checkpoint_path}")
        from safetensors.torch import load_file
        adapter_weights = load_file(checkpoint_path)
        
        # Filter and rename keys
        recovered_weights = {}
        for k, v in adapter_weights.items():
            if "thinking_residual" in k:
                # adapter key: base_model.model.model.thinking_residual_...
                # target key: model.thinking_residual_...
                # Check if model is wrapped in FastLanguageModel (which wraps it in 'model')
                # But 'model' here is likely 'FastLanguageModel' object which contains 'model' attribute?
                # Actually FastLanguageModel.from_pretrained returns a PeftModel or LlamaForCausalLM
                # Let's assume standard transformer structure: 'model.thinking_residual...'
                
                # The adapter keys are 'base_model.model.model.thinking_residual...'
                # We need to strip 'base_model.model.' to match 'model.thinking_residual...'
                new_key = k.replace("base_model.model.", "")
                recovered_weights[new_key] = v
        
        if recovered_weights:
            print(f"loading {len(recovered_weights)} thinking_residual weights into model...")
            # We need to access the underlying torch model. 
            # unsloth's 'model' object might be a wrapper. 
            # Trying to load directly into 'model' first.
            missing, unexpected = model.load_state_dict(recovered_weights, strict=False)
            print(f"Loaded weights. Missing keys: {len(missing)}. Unexpected keys: {len(unexpected)}")
        else:
            print("No thinking_residual weights found in adapter checkpoint.")
    else:
        print(f"No recovery checkpoint found at {checkpoint_path}")

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
        outputs = model.generate(
            prompt_ids, attention_mask=prompt_mask, 
            generation_config=GenerationConfig(
                do_sample=True,  # for temperature, top-k, etc.
                temperature=temperature,
                max_new_tokens=512,
            ),
            processing_class=tokenizer,
            # is_inference=is_inference,
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
            # print(generated_answer, true_answer, generated_answer == true_answer)

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
        save_path = os.path.join(model_path, "eval_results.json")
        with open(save_path, 'w') as f:
            json.dump({'metrics': metrics, 'results': results}, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged model directory")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (optional)")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (temperature=0)") # Kept for compatibility but temperature overrides
    
    args = parser.parse_args()
    
    temperature = args.temperature
    if args.greedy:
        temperature = 0.001 # Close to 0 for greedy
        
    print(f"Evaluating model: {args.model_path}")
    print(f"Temperature: {temperature}")

    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        exit(1)

    result_file = os.path.join(args.model_path, 'eval_results.json')
    if os.path.exists(result_file):
         print(f"Warning: {result_file} already exists. It will be overwritten.")

    metrics = evaluate_model(
        model_path=args.model_path,
        temperature=temperature,
        is_inference=True,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        save_results=True,
    )
    
    print("\nFinal Metrics:")
    print(json.dumps(metrics, indent=2))