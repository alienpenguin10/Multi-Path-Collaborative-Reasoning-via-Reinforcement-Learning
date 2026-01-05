from safetensors.torch import load_file
import sys

model_path = "outputs/grpo-Qwen2.5-1.5B-Instruct-gsm8k-20260101_152454/checkpoint-1245/adapter_model.safetensors"
try:
    state_dict = load_file(model_path)
    keys = list(state_dict.keys())
    print(f"Total keys: {len(keys)}")
    
    residual_keys = [k for k in keys if "thinking_residual" in k]
    if residual_keys:
        print("Found thinking residual keys:")
        for k in residual_keys:
            print(k)
    else:
        print("NO thinking residual keys found.")
        print("First 10 keys:")
        for k in keys[:10]:
            print(k)

except Exception as e:
    print(f"Error loading safetensors: {e}")
