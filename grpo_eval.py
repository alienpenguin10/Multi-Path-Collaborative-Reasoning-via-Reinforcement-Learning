###########################
# Step 4. LOAD AND TEST MODEL  #
###########################
from transformers import AutoTokenizer, AutoModelForCausalLM
from grpo_train import SYSTEM_PROMPT, build_prompt, extract_answer_from_model_output, extract_single_number, extract_last_number, prepare_dataset
import torch
import os

def evaluate_model(model, tokenizer, eval_examples, device):
   """
   Evaluates the model on a set of examples and prints detailed results.

   Args:
       model: The language model to evaluate.
       tokenizer: The tokenizer for encoding inputs and decoding outputs.
       eval_examples (list): List of evaluation examples, each containing "prompt" and "answer".
       device: The device (CPU or GPU) to run evaluation on.

   Returns:
       float: The accuracy percentage (correct predictions / total examples * 100).

   Explanation:
       1. Sets the model to evaluation mode.
       2. For each example in the evaluation set:
          - Encodes the prompt and generates a response using the model.
          - Extracts the predicted answer from the generated response.
          - Compares the predicted answer with the expected answer using multiple methods:
            a. Exact string matching
            b. Single number extraction and comparison
            c. Last number extraction and comparison
          - Prints detailed information about each example.
       3. Calculates and returns the overall accuracy.
       4. Returns the model to training mode.
   """
   model.eval()
   correct = 0
   total = len(eval_examples)
   print("\n" + "="*50)
   print("EVALUATION ON", total, "EXAMPLES")
   print("="*50)

   for idx, example in enumerate(eval_examples):
       # Get the prompt and expected answer
       full_prompt = example["prompt"]
       expected = example["answer"]

       print(f"Evaluating example {idx+1}/{total}...", end=" ", flush=True)

       # Tokenize and generate response with attention mask
       inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
       with torch.no_grad():
           outputs = model.generate(
               inputs["input_ids"],
               attention_mask=inputs["attention_mask"],
               max_new_tokens=512,
               do_sample=True,
               temperature=0.7,
               pad_token_id=tokenizer.pad_token_id,
               eos_token_id=tokenizer.eos_token_id,
               early_stopping=False,
           )
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)
       print("Done", flush=True)

       try:
           # Extract answer and check correctness
           predicted = extract_answer_from_model_output(response)

           # Try different matching methods
           if predicted == expected:  # Exact match
               is_correct = True
           else:
               # Try single number matching
               pred_num = extract_single_number(str(predicted))
               exp_num = extract_single_number(str(expected))
               if pred_num is not None and exp_num is not None and pred_num == exp_num:
                   is_correct = True
               else:
                   # Try last number matching
                   pred_num = extract_last_number(str(predicted))
                   exp_num = extract_last_number(str(expected))
                   is_correct = (pred_num is not None and exp_num is not None and
                               pred_num == exp_num)

           # Update counter for correct answers
           if is_correct:
               correct += 1

           # Print evaluation details
        #    print("\nPrompt:")
        #    print(full_prompt)
        #    print("\nExpected Answer:")
           print(expected)
        #    print("\nExtracted Answer:")
           print(predicted)
        #    print("\nFull Generated Response:")
        #    print(response)
           print("\nCorrect:", "✓" if is_correct else "✗")
        #    print("-"*50)

       except Exception as e:
           print("\nFailed to parse model output for prompt:")
           print(full_prompt)
           print("Error:", e)
           print("-"*50)

   # Calculate and print final accuracy
   accuracy = (correct / total) * 100
   # print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
   # print("="*50)

   # Return model to training mode
   model.train()
   return accuracy

# def main():
#     """
#     Main function to load the fine-tuned model and test it on example math problems.

#     Explanation:
#         1. Determines the device (GPU if available, otherwise CPU).
#         2. Loads the fine-tuned model and tokenizer from the saved path.
#         3. Tests the model on predefined math problems.
#         4. Formats the prompt using the same SYSTEM_PROMPT and build_prompt function as training.
#         5. Generates and displays responses for each test prompt.
#     """
#     # Determine the device: use GPU if available, else fallback to CPU.
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Load the saved model and tokenizer
#     saved_model_path = "grpo_finetuned_model"


#     # Load the model
#     loaded_model = AutoModelForCausalLM.from_pretrained(
#         saved_model_path,
#         torch_dtype=torch.bfloat16,
#         device_map="auto"
#     )


#     loaded_tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
#     loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

#     # Define test prompts
#     prompts_to_test = [
#         "How much is 1+1?",
#         "I have 3 apples, my friend eats one and I give 2 to my sister, how many apples do I have now?",
#         "Solve the equation 6x + 4 = 40"
#     ]

#     # Test each prompt
#     for prompt in prompts_to_test:
#         # Prepare the prompt using the same format as during training
#         test_messages = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": prompt}
#         ]
#         test_prompt = build_prompt(test_messages)

#         # Tokenize the prompt and generate a response
#         test_input_ids = loaded_tokenizer.encode(test_prompt, return_tensors="pt").to(device)

#         # Generate response with similar parameters to those used in training
#         with torch.no_grad():
#             test_output_ids = loaded_model.generate(
#                 test_input_ids,
#                 max_new_tokens=400,
#                 temperature=0.7,
#                 num_return_sequences=1,
#                 pad_token_id=loaded_tokenizer.pad_token_id,
#                 eos_token_id=loaded_tokenizer.eos_token_id,
#                 do_sample=True,
#                 early_stopping=False
#             )

#         test_response = loaded_tokenizer.decode(test_output_ids[0], skip_special_tokens=True)

#         # Print the test prompt and the model's response
#         print("\nTest Prompt:")
#         print(test_prompt)
#         print("\nModel Response:")
#         print(test_response)

#         # Extract and display the answer part for easier evaluation
#         try:
#             extracted_answer = extract_answer_from_model_output(test_response)
#             print("\nExtracted Answer:")
#             print(extracted_answer)
#             print("-" * 50)
#         except Exception as e:
#             print(f"\nFailed to extract answer: {e}")
#             print("-" * 50)

if __name__ == "__main__":
    # Determine the device: use GPU if available, else fallback to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the saved model and tokenizer
    saved_model_path = "grpo_finetuned_model"

    # Load the model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        saved_model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Model loaded successfully!")

    tokenizer = AutoTokenizer.from_pretrained(saved_model_path, fix_mistral_regex=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Define test prompts
    eval_data = prepare_dataset("test") 


    print("\nFinal model evaluation after GRPO RL fine-tuning:")
    post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")