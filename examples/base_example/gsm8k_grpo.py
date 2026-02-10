import os
import sys

# Fix shadowing of installed packages by local directories
sys.path.insert(0, os.path.abspath("unsloth"))
sys.path.insert(0, os.path.abspath("transformers/src"))
sys.path.insert(0, os.path.abspath("trl"))

from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported

import re
import torch
import subprocess
import string
from torch.nn.parallel import DistributedDataParallel
import transformers
print(f"DEBUG: transformers path: {transformers.__file__}")
print(f"DEBUG: sys.path: {sys.path}")

# [FIX] Patch DDP class to automatically expose .config from the internal module
@property
def ddp_config_patch(self):
    return getattr(self.module, "config", None)

DistributedDataParallel.config = ddp_config_patch


# Distributed Training Setup
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))
RANK = int(os.environ.get("RANK", LOCAL_RANK))
IS_MAIN_PROCESS = RANK in [-1, 0]

if IS_MAIN_PROCESS:
    print(f"GPUs available: {WORLD_SIZE}")

os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

# os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
# os.environ["UNSLOTH_DISABLE_CACHE"] = "1"
# os.environ["UNSLOTH_VLLM_STANDBY"] = "1" # [NEW] Extra 30% context lengths!
# os.environ["UNSLOTH_DISABLE_RL_PATCH"] = "0" # NOTE: We *want* the RL patch for GRPO speedups. Leaving this disabled can cause slower training and, depending on versions, unexpected behavior.

ANSWER_START = "####"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "final answer is provided after the " + ANSWER_START + " tag, i.e., "
    "{reasoning process} " + ANSWER_START + " {answer}."
)


if LOCAL_RANK >= 0:
    torch.cuda.set_device(LOCAL_RANK)

if IS_MAIN_PROCESS:
    logdir = "./logs"
    tb_port = 6007  # Different port from SFT
    try:
        tb_proc = subprocess.Popen(
            ["tensorboard", f"--logdir={logdir}", f"--port={tb_port}", "--bind_all"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ
        )
        print(f"View logs at: http://localhost:{tb_port}")
        print(f"To stop tensorboard: kill {tb_proc.pid}")
    except FileNotFoundError:
        print("Tensorboard not found, skipping...")


# Model Configuration
model_slug = "qwen/Qwen2.5-1.5B-Instruct"  # "qwen/Qwen2.5-3B-Instruct" "qwen/Qwen2.5-7B-Instruct"
test_run = False  # Set to True for quick smoke tests
max_seq_length = 2048  # Shorter for GRPO as it generates responses
dtype = None
load_in_4bit = False # Because of HRPO; otherwise True
load_in_8bit = False

# GRPO Training Hyperparameters
lora_rank = 32
lora_alpha = 64  # lora_rank * 2 as in hrpo

per_device_train_batch_size = 2
gradient_accumulation_steps = max(1, int(8 / per_device_train_batch_size / WORLD_SIZE))
num_train_epochs = 1  # GRPO typically needs fewer epochs
learning_rate = 5e-6  # Lower LR for RL fine-tuning

# GRPO specific
num_generations = 8  # Number of completions to generate per prompt (G in GRPO)
max_prompt_length = 1024
max_completion_length = 1024
beta = 0.005  # KL penalty coefficient
temperature = 0.5  # Sampling temperature


# Dataset Configuration
ft_dataset_name = "openai/gsm8k"
q_column = "question"
a_column = "answer"
chunk_size = 1000



PatchFastRL("GRPO", FastLanguageModel)  # Patch for faster GRPO training


import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

# Load Model with Unsloth
if IS_MAIN_PROCESS:
    print("Loading model...")

# Load the pretrained language model and tokenizer with specified configuration and quantization options
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_slug,
    max_seq_length=max_prompt_length + max_completion_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    device_map={"": LOCAL_RANK} if LOCAL_RANK != -1 else {"": 0}, #"auto" - doesn't work with multi-GPU training - manually setting a device_map can sometimes conflict with how Accelerate wants to handle the distribution
    use_gradient_checkpointing="unsloth",
    fast_inference=False,  # Set False for training - Enables vLLM fast inference
    random_state=42,
)
model.answer_start = ANSWER_START

# Set tokenizer properties (GRPO needs left padding for generation)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

if IS_MAIN_PROCESS:
    print(f"Model loaded: {model_slug}")
    print(f"Padding side: {tokenizer.padding_side}")
    print(model)

# Apply LoRA (Low-Rank Adaptation) PEFT to the loaded model with specified parameters.
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    # OR via specific list of modules - if you know what you're doing
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",  # don't train these if it's MoE
    ],
    # OR all linear layers (not recommended)
    # target_modules = ["all-linear"] #to train all linear layers
    # modules_to_save = ["lm_head","embed_tokens"] # to train embeddings

    lora_alpha=lora_alpha,
    lora_dropout=0,
    bias="none",  # Supports any, but 0 is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=True,  # rank stabilized LoRA
)


if IS_MAIN_PROCESS:
    print("Model with LoRA applied:")
    print(model.print_trainable_parameters())


# Load and Prepare Dataset
from datasets import load_dataset

#We directly leverage @willccbb for data prep and all reward functions. You are free to create your own!
# Load and prep dataset
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def process_gsm8k(batch):
    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},{"role": "user", "content": q.strip()},
    ] for q in batch["question"]]

    return {
        "prompt": prompts,
        "answer": [extract_hash_answer(a) for a in batch["answer"]]
    }

def preprocess_gsm8k(split="train", chunk_size=chunk_size):
    dataset = load_dataset(ft_dataset_name, "main")[split]
    return dataset.map(process_gsm8k, batched=True, batch_size=chunk_size, load_from_cache_file=False)

train_dataset = preprocess_gsm8k("train", chunk_size=chunk_size)

if test_run:
    train_dataset = train_dataset.select(range(1))
    if IS_MAIN_PROCESS:
        print(f"Test run: using {len(train_dataset)} examples")

if IS_MAIN_PROCESS:
    print(f"Training on {len(train_dataset)} examples.")
    print(train_dataset['question'][0])


if IS_MAIN_PROCESS:
    print("\nExample GRPO prompt:")
    print(train_dataset[0]["prompt"][:500])
    print(f"\nGround truth answer: {train_dataset[0]['answer']}")

# Reward functions
def extract_from_response(text: str) -> str:
    try:
        answer = text.split(ANSWER_START)[-1].strip()
        if answer.endswith("."):
            answer = answer[:-1].strip()
        return answer
    except IndexError:
        return ""

def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        try:
            n = eval(n)
        except:
            #print("Conversion to floating number fails: {}".format(n))
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

def get_reward_func(process_answer_func):
    def reward_func(completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]

        ans = [process_answer_func(a) for a in answer]
        extracted = [extract_from_response(r) for r in responses]
        predictions = [process_answer_func(r) for r in extracted]
        accuracy = [True if r == a else False for r, a in zip(predictions, ans)]

        escaped_answer_start = re.escape(ANSWER_START)
        pattern = f"^(?:(?!{escaped_answer_start}).)*{escaped_answer_start}(?:(?!{escaped_answer_start}).)*$"
        matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]

        rewards = [1.0 if a and m else 0.0 for a, m in zip(accuracy, matches)]

        # for completion_id in range(len(completions)):
        #     print(f"Completion {completion_id}")
        #     print(
        #         "=" * 50,
        #         f"\nBatch accuracy: " + "".join("Y" if r > 0 else "N" for r in rewards),
        #         f"\n1/{len(completions)} responses (answer: {ans[completion_id]}):\n{responses[completion_id]}",
        #         "\n" + "=" * 50,
        #     )
        return rewards 


    return reward_func


# GRPO Configuration and Trainer
from trl import GRPOTrainer, GRPOConfig
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"grpo-{model_slug.split('/')[-1]}-gsm8k-{current_timestamp}"

grpo_config = GRPOConfig(
    use_vllm=False, # use vLLM for fast inference!
    
    # Basic training args
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps, # Increase to 4 for smoother training
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,  # Learning rate for GRPO training
    # max_steps=10,  # uncomment for shorter run
    
    # Optimizer settings (matching hrpo)
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    max_grad_norm=0.1,
    
    # GRPO specific parameters
    num_generations=num_generations,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    beta=beta,
    temperature=temperature,
     
    # Logging
    logging_strategy="steps",
    logging_steps=1,
    logging_dir=f"./logs/{run_name}",
    report_to="tensorboard",
    
    # Precision
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    
    # Output
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    seed=3407,
    remove_unused_columns=False, # DOESN'T SEND THE ANSWER COLUMN TO THE REWARD FUNCTIONS
    
    # DDP args
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},  # Can set false for DDP if using LoRA and facing issues
    
    # Save
    save_strategy="steps",
    save_steps=250,
    save_total_limit=3,
)

"""And let's run the trainer! If you scroll up, you'll see a table of rewards. The goal is to see the `reward` column increase!

You might have to wait 150 to 200 steps for any action. You'll probably get 0 reward for the first 100 steps. Please be patient!

| Step | Training Loss | reward    | reward_std | completion_length | kl       |
|------|---------------|-----------|------------|-------------------|----------|
| 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
| 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
| 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |

"""

# Create trainer with reward function
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    # processing_class=tokenizer,
    reward_funcs= [get_reward_func(process_gsm8k_answer)],
)



if IS_MAIN_PROCESS:
    print(f"\n{'='*60}")
    print("GRPO Training Configuration")
    print(f"{'='*60}")
    print(f"Model: {model_slug}")
    print(f"Dataset: {ft_dataset_name}")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Batch size per device: {per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps * WORLD_SIZE}")
    print(f"Num generations per prompt: {num_generations}")
    print(f"Max prompt length: {max_prompt_length}")
    print(f"Max completion length: {max_completion_length}")
    print(f"Learning rate: {learning_rate}")
    print(f"Beta (KL penalty): {beta}")
    print(f"Temperature: {temperature}")
    print(f"Epochs: {num_train_epochs}")
    print(f"{'='*60}\n")

# Check memory
#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

if IS_MAIN_PROCESS:
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")


# Training
import torch.distributed as dist

if IS_MAIN_PROCESS:
    print("🔌 Dist available:", dist.is_available(), "initialized:", dist.is_initialized())
if dist.is_initialized():
    if IS_MAIN_PROCESS:
        print("🔗 backend:", dist.get_backend(), "🌍 world:", dist.get_world_size(), "🏅 rank:", dist.get_rank())

if IS_MAIN_PROCESS:
    print("\n🚀 Starting GRPO training...")

trainer_stats = trainer.train()

# Final memory results
#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)

if IS_MAIN_PROCESS:
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

if IS_MAIN_PROCESS:
    # # Manually shorten the run name and just re-run the push, if needed.
    # run_name = f"grpo-ddp-demo-{WORLD_SIZE}"
    # # Merge to 16bit (RECOMMENDED, should merge to a dequantized base model for best accuracy)
    org = "Trelis"
    print(f"Saving and/or pushing as {run_name} and {org}/{run_name}")
    
    # Save model
    print(f"📦 Saving model to {run_name}/...")
    # model.save_pretrained(f"{run_name}")
    # tokenizer.save_pretrained(f"{run_name}")
    print(f"✅ Model saved to {run_name}/")


if IS_MAIN_PROCESS:
    print("Training completed successfully!")