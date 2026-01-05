"""
Visualize Hybrid Reasoning Mechanism in HRPO/H3PO

This script demonstrates how the model blends:
1. Discrete token embeddings (ê_{t+1}) from sampled token x̂_{t+1}
2. Projected hidden states (h_{t+1}) from previous step
Using gates r_t and i_t to create the hybrid input e_{t+1}
"""

import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath("unsloth"))
sys.path.insert(0, os.path.abspath("transformers/src"))
sys.path.insert(0, os.path.abspath("trl"))

from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# Disable warnings
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

ANSWER_START = "####"

def visualize_thinking_residual(model, prompt_text, max_new_tokens=50):
    """
    Generate text and visualize the hybrid reasoning mechanism.
    """
    
    # Prepare inputs
    tokenizer = model.tokenizer
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    print("="*80)
    print("HYBRID REASONING VISUALIZATION")
    print("="*80)
    print(f"\nPrompt: {prompt_text}\n")
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}\n")
    
    # Initialize generation state
    model.eval()
    generated_ids = input_ids.clone()
    
    # Get embedding layer
    embed_layer = model.get_input_embeddings()
    
    # Storage for visualization data
    step_data = []
    
    # Manually step through generation to capture intermediate values
    past_key_values = None
    last_thinking_states = None
    is_thinking = None
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            print(f"\n{'='*80}")
            print(f"GENERATION STEP {step + 1}")
            print(f"{'='*80}")
            
            # Prepare inputs for this step
            if past_key_values is None:
                # First step - full prompt
                current_input_ids = generated_ids
            else:
                # Subsequent steps - only last token
                current_input_ids = generated_ids[:, -1:]
            
            # Get token embedding (ê_t)
            token_embedding = embed_layer(current_input_ids)
            
            print(f"\n1. SAMPLED TOKEN (previous step):")
            if current_input_ids.shape[1] == 1:
                token_id = current_input_ids[0, 0].item()
                token_text = tokenizer.decode([token_id])
                print(f"   Token ID: {token_id}")
                print(f"   Token: '{token_text}'")
                print(f"   Embedding shape: {token_embedding.shape}")
                print(f"   Embedding norm: {token_embedding.norm().item():.4f}")
                print(f"   Embedding sample values: {token_embedding[0, 0, :5].float().cpu().numpy()}")
            
            # Model forward pass with thinking_residual monitoring
            model_kwargs = {
                "input_ids": current_input_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "return_dict": True,
            }
            
            # Add thinking states if available
            if last_thinking_states is not None and is_thinking is not None:
                model_kwargs["is_thinking"] = is_thinking
                model_kwargs["last_thinking_states"] = last_thinking_states
                
                print(f"\n2. PROJECTED HIDDEN STATE (h_t):")
                print(f"   Shape: {last_thinking_states.shape}")
                print(f"   Norm: {last_thinking_states.norm().item():.4f}")
                print(f"   Sample values: {last_thinking_states[0, :5].float().cpu().numpy()}")
                
                # Access the thinking_residual module to show gating
                if hasattr(model.model, 'thinking_residual'):
                    thinking_residual = model.model.thinking_residual
                    
                    # Get the current token embedding for blending
                    X = token_embedding
                    h = last_thinking_states.unsqueeze(1)  # Add sequence dimension
                    
                    # Compute gates (replicating the thinking_residual logic)
                    # r_t and i_t gates
                    gate_r = thinking_residual.gate_r(X)  # Reset gate
                    gate_i = thinking_residual.gate_i(torch.cat([X, h], dim=-1))  # Input gate
                    
                    r_t = torch.sigmoid(gate_r)
                    i_t = torch.sigmoid(gate_i)
                    
                    print(f"\n3. GATING MECHANISM:")
                    print(f"   Reset gate (r_t) - controls hidden state:")
                    print(f"     Shape: {r_t.shape}")
                    print(f"     Mean: {r_t.mean().item():.4f}")
                    print(f"     Std: {r_t.std().item():.4f}")
                    print(f"     Sample values: {r_t[0, 0, :5].float().cpu().numpy()}")
                    
                    print(f"\n   Input gate (i_t) - controls blending:")
                    print(f"     Shape: {i_t.shape}")
                    print(f"     Mean: {i_t.mean().item():.4f}")
                    print(f"     Std: {i_t.std().item():.4f}")
                    print(f"     Sample values: {i_t[0, 0, :5].float().cpu().numpy()}")
                    
                    # Compute Lambda (learnable scaling)
                    Lambda = thinking_residual.Lambda(torch.ones_like(X))
                    
                    # Compute h_tilde (modulated hidden state)
                    h_tilde = r_t * h
                    
                    # Compute X_hat (candidate hybrid state)
                    X_hat = torch.tanh(thinking_residual.W(torch.cat([X, h_tilde], dim=-1)))
                    
                    # Final blending with Lambda
                    a_t = i_t * Lambda  # This becomes embeds_ratio
                    X_out = a_t * X + torch.sqrt(1 - a_t**2) * X_hat
                    
                    embeds_ratio_scalar = a_t.mean().item()
                    hidden_ratio_scalar = torch.sqrt(1 - a_t**2).mean().item()
                    
                    print(f"\n4. HYBRID BLENDING:")
                    print(f"   Lambda (learnable scaling):")
                    print(f"     Mean: {Lambda.mean().item():.4f}")
                    print(f"   ")
                    print(f"   Embeds ratio (a_t = i_t * Λ):")
                    print(f"     Mean: {embeds_ratio_scalar:.4f}")
                    print(f"     This means {embeds_ratio_scalar*100:.1f}% token embedding")
                    print(f"   ")
                    print(f"   Hidden ratio (sqrt(1 - a_t²)):")
                    print(f"     Mean: {hidden_ratio_scalar:.4f}")
                    print(f"     This means {hidden_ratio_scalar*100:.1f}% hidden state")
                    print(f"   ")
                    print(f"   Hybrid output norm: {X_out.norm().item():.4f}")
                    print(f"   Token embedding norm: {X.norm().item():.4f}")
                    print(f"   Hidden state norm: {h.norm().item():.4f}")
                    
                    # Store for later visualization
                    step_data.append({
                        'step': step,
                        'token': token_text if current_input_ids.shape[1] == 1 else 'prompt',
                        'embeds_ratio': embeds_ratio_scalar,
                        'hidden_ratio': hidden_ratio_scalar,
                        'r_t_mean': r_t.mean().item(),
                        'i_t_mean': i_t.mean().item(),
                        'lambda_mean': Lambda.mean().item(),
                    })
            
            # Forward pass
            outputs = model(**model_kwargs)
            
            # Get logits and sample next token
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits / 1.0, dim=-1)  # temperature=1.0 for more diversity
            
            # Use greedy decoding for first few steps to get stable output
            if step < 3:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            
            next_token_id = next_token[0, 0].item()
            next_token_text = tokenizer.decode([next_token_id])
            
            print(f"\n5. NEXT TOKEN SAMPLING:")
            print(f"   Sampled token ID: {next_token_id}")
            print(f"   Sampled token: '{next_token_text}'")
            print(f"   Token probability: {probs[0, next_token_id].item():.4f}")
            
            # Update for next iteration
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values
            
            # Compute last_thinking_states for next step (weighted embedding based on probs)
            last_thinking_states = torch.einsum('bv,vd->bd', probs, embed_layer.weight)
            last_thinking_states /= torch.sqrt((probs ** 2).sum(-1, keepdim=True)).to(last_thinking_states.dtype)
            
            # Check if still thinking
            generated_text = tokenizer.decode(generated_ids[0, inputs["input_ids"].shape[1]:])
            is_thinking = [ANSWER_START not in generated_text]
            
            print(f"\n6. THINKING STATUS:")
            print(f"   Still thinking: {is_thinking[0]}")
            print(f"   Generated so far: {generated_text}")
            
            # Stop if EOS token, but continue even if we see "####" for demonstration
            if next_token_id == tokenizer.eos_token_id:
                print(f"\n{'='*80}")
                print("GENERATION COMPLETE (EOS)")
                print(f"{'='*80}")
                break
            
            # Stop after sufficient steps for visualization
            if step >= 12:
                print(f"\n{'='*80}")
                print("STOPPING AFTER 12 STEPS FOR VISUALIZATION")
                print(f"{'='*80}")
                break
    
    # Final output
    final_text = tokenizer.decode(generated_ids[0])
    print(f"\n\nFINAL GENERATED TEXT:")
    print(f"{final_text}")
    
    # Create visualization
    if len(step_data) > 0:
        plot_hybrid_ratios(step_data)
    
    return step_data


def plot_hybrid_ratios(step_data):
    """
    Plot the evolution of embedding vs hidden state ratios over generation steps.
    """
    steps = [d['step'] for d in step_data]
    embeds_ratios = [d['embeds_ratio'] for d in step_data]
    hidden_ratios = [d['hidden_ratio'] for d in step_data]
    r_t_means = [d['r_t_mean'] for d in step_data]
    i_t_means = [d['i_t_mean'] for d in step_data]
    lambda_means = [d['lambda_mean'] for d in step_data]
    tokens = [d['token'] for d in step_data]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Hybrid blending ratios
    ax1 = axes[0]
    ax1.plot(steps, embeds_ratios, 'o-', label='Token Embedding Ratio (a_t)', linewidth=2, markersize=8)
    ax1.plot(steps, hidden_ratios, 's-', label='Hidden State Ratio (√(1-a_t²))', linewidth=2, markersize=8)
    ax1.set_xlabel('Generation Step', fontsize=12)
    ax1.set_ylabel('Blending Ratio', fontsize=12)
    ax1.set_title('Hybrid Reasoning: Token Embedding vs Hidden State Contribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Add token labels
    for i, (step, token) in enumerate(zip(steps, tokens)):
        ax1.annotate(token, (step, embeds_ratios[i]), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=8, rotation=45)
    
    # Plot 2: Gate values
    ax2 = axes[1]
    ax2.plot(steps, r_t_means, 'o-', label='Reset Gate (r_t)', linewidth=2, markersize=8)
    ax2.plot(steps, i_t_means, 's-', label='Input Gate (i_t)', linewidth=2, markersize=8)
    ax2.set_xlabel('Generation Step', fontsize=12)
    ax2.set_ylabel('Gate Value (mean)', fontsize=12)
    ax2.set_title('Gating Mechanism: r_t (reset) and i_t (input) Gates', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Lambda scaling
    ax3 = axes[2]
    ax3.plot(steps, lambda_means, 'o-', color='purple', linewidth=2, markersize=8)
    ax3.set_xlabel('Generation Step', fontsize=12)
    ax3.set_ylabel('Lambda Value (mean)', fontsize=12)
    ax3.set_title('Learnable Scaling Factor (Λ)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hybrid_reasoning_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\n\nVisualization saved to: hybrid_reasoning_visualization.png")
    
    # Print summary statistics
    print(f"\n\nSUMMARY STATISTICS:")
    print(f"{'='*80}")
    print(f"Average token embedding ratio: {np.mean(embeds_ratios):.4f} ± {np.std(embeds_ratios):.4f}")
    print(f"Average hidden state ratio: {np.mean(hidden_ratios):.4f} ± {np.std(hidden_ratios):.4f}")
    print(f"Average reset gate (r_t): {np.mean(r_t_means):.4f} ± {np.std(r_t_means):.4f}")
    print(f"Average input gate (i_t): {np.mean(i_t_means):.4f} ± {np.std(i_t_means):.4f}")
    print(f"Average Lambda (Λ): {np.mean(lambda_means):.4f} ± {np.std(lambda_means):.4f}")


def main():
    """
    Main function to run the visualization.
    
    NOTE: For best results, use a model checkpoint that has been trained with HRPO/H3PO.
    The base model won't show meaningful hybrid reasoning since it hasn't learned to use
    the thinking_residual mechanism yet. This script demonstrates the mechanism even with
    an untrained model, but the ratios and gates will be more meaningful after training.
    """
    print("Loading model...")
    print("\n" + "="*80)
    print("NOTE: Using base model for demonstration.")
    print("For meaningful hybrid reasoning, use a trained HRPO/H3PO checkpoint.")
    print("The base model's thinking_residual is randomly initialized.")
    print("="*80 + "\n")
    
    model_name = "unsloth/Qwen2.5-1.5B-Instruct"
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    
    model.tokenizer = tokenizer
    model.answer_start = ANSWER_START
    
    # Example math problem - simpler prompt without priming "####"
    prompt = """User: If John has 5 apples and buys 3 more, how many apples does he have?
Assistant: Let me think about this step by step."""
    
    # Run visualization
    step_data = visualize_thinking_residual(model, prompt, max_new_tokens=20)
    
    print("\n\nVisualization complete!")
    print("Check 'hybrid_reasoning_visualization.png' for plots.")


if __name__ == "__main__":
    main()
