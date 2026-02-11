# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **GRPO (Group Relative Policy Optimization)** for reinforcement learning fine-tuning of language models, along with the **M3PO (Multi-Path Collaborative Reasoning)** extension. The project trains models on the GSM8K math reasoning dataset using a custom RL training loop.

## Key Files

- `grpo_train.py` - Main training script implementing GRPO algorithm with PPO-style clipping and KL divergence penalty. Uses DataParallel for multi-GPU training.
- `grpo_eval.py` - Evaluation script to test fine-tuned models on math problems. Note: Requires `SYSTEM_PROMPT` and helper functions from training script.
- `m3po_module.py` - Cross-path interaction module for M3PO that blends embeddings across multiple generation paths during reasoning.
- `transformers/` - Local fork of HuggingFace Transformers library (includes `modeling_llama.py` at root level for customization).

## Commands

### Running Training
```bash
# Requires 2+ GPUs with CUDA. Tested on 8xA100 (80GB)
python grpo_train.py
```

Training parameters are configured in `training_config` dict in `grpo_train.py`. Key settings:
- `batch_size`: 7 (reduce for fewer GPUs)
- `num_generations`: 12 (reduce for less VRAM)
- `max_completion_length`: 400 (reduce for less VRAM)

### Running Evaluation
```bash
python grpo_eval.py
```
Loads model from `grpo_finetuned_model/` directory.

### Transformers Library Commands
From `transformers/` directory:

```bash
# Install dev dependencies
pip install -e ".[quality]"
pip install -e ".[testing]"

# Style and code fixes (run after changes)
make fixup

# Run all tests
make test

# Run specific model tests
pytest tests/models/[name]/test_modeling_[name].py
```

## Architecture

### GRPO Training Loop
1. **Rollout generation**: Sample prompts, generate multiple completions per prompt using the policy model
2. **Reward computation**: Combined reward = correctness (0-2.0) + format compliance (0-0.8)
3. **Advantage estimation**: Standardize rewards within each prompt's generations
4. **Policy update**: PPO surrogate loss with clipping + KL penalty against reference model
5. **Reference model update**: Deep copy of policy after each outer iteration

### Reward Structure
- **Correctness**: 2.0 for exact match, 1.5 for numeric equivalence, 0.0 otherwise
- **Format**: 0.2 each for `<reasoning>`, `</reasoning>`, `<answer>`, `</answer>` tags

### M3PO Cross-Path Interaction
Applied during generation to blend hidden states across reasoning paths:
1. Compute pairwise cosine similarity between path embeddings
2. Apply temperature-scaled softmax attention (excluding self and inactive paths)
3. Blend: `h_i = (1 - λ) * e_i + λ * contextual_i`

Enable debug output with `M3PO_DEBUG=1` environment variable.

## Transformers Fork Notes

The `transformers/` directory is a fork of HuggingFace Transformers. Key conventions:
- Model files in `src/transformers/models/` should be self-contained
- Use `# Copied from` comments for shared code - `make fixup` propagates changes
- For new models, prefer `modular_*.py` style (auto-generates `modeling_*.py`)
- Code style: ruff with line length 119, double quotes
