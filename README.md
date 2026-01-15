### GRPO Training Script (Fastest with VLLM)
CUDA_VISIBLE_DEVICES=0 python grpo_baseline_gsm8k.py
  --model_name Qwen/Qwen2.5-1.5B-Instruct
  --group_size 8
  --per_device_train_batch_size 64
  --gradient_accumulation_steps 1
  --max_prompt_length 1024
  --max_completion_length 1024

CUDA_VISIBLE_DEVICES=0 python grpo_baseline_gsm8k.py
  --model_name Qwen/Qwen2.5-1.5B-Instruct
  --group_size 4
  --per_device_train_batch_size 8
  --gradient_accumulation_steps 4
  --max_prompt_length 1024
  --max_completion_length 1024

### Local Checkpoint Path
CUDA_VISIBLE_DEVICES=0 python eval_baseline_gsm8k.py --checkpoint_path ./experiments/Qwen2.5-1.5B-Instruct-gsm8k-baseline-group8-lora32-temp0.5 --batch_size 64 --greedy
### Hugging Face Checkpoint Path
CUDA_VISIBLE_DEVICES=0 python eval_baseline_gsm8k.py --model_path Alienpenguin10/Qwen2.5-1.5B-Instruct-gsm8k-baseline-group8-lora32-temp0.5 --batch_size 64 --greedy
### Base Model
CUDA_VISIBLE_DEVICES=0 python eval_baseline_gsm8k.py --model_path Qwen/Qwen2.5-1.5B-Instruct --batch_size 64 --greedy --output_repo Alienpenguin10/BASE
