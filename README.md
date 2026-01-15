# Latent Space Reasoning

## Running Training, Evaluation, and Shutdown

The `run_train_eval_shutdown.sh` script handles the full pipeline: training, evaluation, pushing to HuggingFace, and terminating the Lambda GPU instance.

### Prerequisites

Set the following environment variables:
```bash
export LAMBDA_API_KEY="your_lambda_api_key"
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_api_key"
```

### Configuration

Before running, edit `run_train_eval_shutdown.sh` and update:
```bash
LAMBDA_INSTANCE_ID="your_actual_instance_id"
```

### Make the Script Executable

```bash
chmod +x run_train_eval_shutdown.sh
```

### Execute the Script

```bash
./run_train_eval_shutdown.sh
```

### What the Script Does

1. Activates the virtual environment (`./lamda/bin/activate`)
2. Runs GRPO training (`grpo_gsm8k.py`)
3. Evaluates the model on GSM8k (`eval_baseline_gsm8k.py`)
4. Saves eval results to the local model folder
5. Pushes model and eval results to HuggingFace (`push_to_hf.py`)
6. Terminates the Lambda GPU instance

## Running Scripts Individually

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python grpo_gsm8k.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --run_name HRPO \
    --group_size 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --lora_rank 32 \
    --temperature 0.5
```

Additional training options:
- `--learning_rate`: Learning rate (default: 5e-6)
- `--beta`: KL penalty coefficient (default: 0.005)
- `--num_train_epochs`: Number of epochs (default: 1)
- `--test_run`: Use limited data for testing
- `--load_in_4bit`: Load model in 4-bit quantization

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python eval_baseline_gsm8k.py \
    --checkpoint_path ./HRPO \
    --batch_size 32 \
    --greedy True
```

Additional eval options:
- `--force_eval`: Run evaluation even if results exist
- `--num_samples`: Number of samples to evaluate (default: all)
- `--temperature`: Temperature for metadata (default: 0.5)

### Push to HuggingFace

```bash
python push_to_hf.py --model_path ./HRPO --hub_name Alienpenguin10/HRPO
```
