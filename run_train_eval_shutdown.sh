#!/bin/bash
set -e

# ============================================
# Configuration
# ============================================
LAMBDA_INSTANCE_ID="PLACEHOLDER_INSTANCE_ID"  # UPDATE THIS BEFORE RUNNING
LAMBDA_API_KEY="${LAMBDA_API_KEY}"  # Set in environment or update here

# Training Configuration
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
RUN_NAME="HRPO"
GROUP_SIZE=8
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=1024
LORA_RANK=32
TEMPERATURE=0.5

# Eval Configuration
EVAL_BATCH_SIZE=32

# Model paths
LOCAL_MODEL_PATH="./${RUN_NAME}"

# ============================================
# Activate virtual environment
# ============================================
echo "Activating virtual environment..."
source ./lamda/bin/activate

# ============================================
# Run Training
# ============================================
echo "=========================================="
echo "Starting GRPO Training..."
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python grpo_gsm8k.py \
    --model_name "${MODEL_NAME}" \
    --run_name "${RUN_NAME}" \
    --group_size ${GROUP_SIZE} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --max_completion_length ${MAX_COMPLETION_LENGTH} \
    --lora_rank ${LORA_RANK} \
    --temperature ${TEMPERATURE}

if [ $? -ne 0 ]; then
    echo "Training failed! Exiting without shutdown."
    exit 1
fi

echo "Training completed successfully!"

# ============================================
# Run Evaluation (Local Model Only)
# ============================================
echo "=========================================="
echo "Starting GSM8k Evaluation..."
echo "=========================================="

# Wait for model to be fully saved
sleep 5

# Check if model was saved locally
if [ ! -d "${LOCAL_MODEL_PATH}" ]; then
    echo "ERROR: Local model not found at ${LOCAL_MODEL_PATH}"
    echo "Training may have failed to save the model."
    exit 1
fi

echo "Found local model at ${LOCAL_MODEL_PATH}"
CUDA_VISIBLE_DEVICES=0 python eval_baseline_gsm8k.py \
    --checkpoint_path "${LOCAL_MODEL_PATH}" \
    --batch_size ${EVAL_BATCH_SIZE} \
    --greedy True \
    --force_eval

if [ $? -ne 0 ]; then
    echo "Evaluation failed! Exiting without shutdown."
    exit 1
fi

echo "Evaluation completed successfully!"

# ============================================
# Push Model and Eval Results to HuggingFace
# ============================================
echo "=========================================="
echo "Pushing model to HuggingFace..."
echo "=========================================="

python push_to_hf.py \
    --model_path "${LOCAL_MODEL_PATH}" \
    --hub_name "Alienpenguin10/${RUN_NAME}"

if [ $? -ne 0 ]; then
    echo "Push to HuggingFace failed! Exiting without shutdown."
    exit 1
fi

echo "HuggingFace push completed!"

# ============================================
# Terminate Lambda Instance
# ============================================
echo "=========================================="
echo "Terminating Lambda instance..."
echo "=========================================="

if [ -z "${LAMBDA_API_KEY}" ]; then
    echo "ERROR: LAMBDA_API_KEY not set. Cannot terminate instance."
    echo "Please set LAMBDA_API_KEY environment variable."
    exit 1
fi

if [ "${LAMBDA_INSTANCE_ID}" = "PLACEHOLDER_INSTANCE_ID" ]; then
    echo "ERROR: LAMBDA_INSTANCE_ID not updated. Please update the script with actual instance ID."
    exit 1
fi

# Terminate the instance using Lambda API
curl -s -X POST \
    -H "Authorization: Bearer ${LAMBDA_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"instance_ids\": [\"${LAMBDA_INSTANCE_ID}\"]}" \
    "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate"

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "Instance ${LAMBDA_INSTANCE_ID} termination requested."
echo "=========================================="
