"""
Part 1: Basic Setup and Imports
In this first part, we install and import all necessary modules. 
We also set up our environment by configuring random seeds for reproducibility and initializing environment variables required for experiment tracking. 
In addition, we install and import libraries that provide optimized transformer attention mechanisms (FlashAttention2) and reporting (Weights and Biases):
"""

# Import necessary libraries
# Basic Python libraries for various operations
import random
import re
import numpy as np


# PyTorch and related libraries for deep learning
import torch

# Hugging Face libraries for transformer models
from datasets import load_dataset


def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None

    Explanation:
        1. Sets seed for Python's built-in random module for basic random operations.
        2. Sets seed for NumPy, ensuring consistent random number generation in array operations.
        3. Sets seed for PyTorch CPU operations.
        4. If CUDA is available, sets seed for all GPU devices.
        5. Configures cuDNN to ensure deterministic behavior:
           - Sets deterministic flag to True, ensuring reproducible results.
           - Disables benchmarking to prevent algorithm selection based on hardware.

    Note:
        Setting deterministic behavior may impact performance but ensures consistent results
        across multiple runs, which is crucial for debugging and research.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


"""
Part 2: Data Formatting and Answer Extraction

In this section, we define how our data is formatted and how to extract the answer segments from both the model's output and the dataset. 
To ensure that the model outputs its response in a consistent format, we define a system prompt. 
The prompt instructs the model to generate output in an XML-like format containing <reasoning> and <answer> tags. We then provide two functions:

1. extract_answer_from_model_output: This function takes the model's output text and extracts the content within the <answer> tags.
2. extract_answer_from_dataset: This function extracts the expected answer from the GSM8K dataset, which separates the answer using the "####" delimiter.
"""
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_answer_from_model_output(text):
   """
   Extracts the value from the last <answer> tag in the text.

   Args:
       text (str): The model-generated text containing XML-style <answer> tags.

   Returns:
       str or None: The content inside the <answer> tags, or None if no valid answer is found.

   Explanation:
       1. Splits the text on the <answer> tag to isolate content after the tag.
       2. Checks if at least one <answer> tag exists in the text.
       3. For the last <answer> segment:
          - Verifies it contains a closing </answer> tag.
          - Extracts only the content between the tags.
       4. Returns None if the answer is empty (just "...") or if tags are missing.
   """
   # Split on <answer> and take everything after the last occurrence
   parts = text.split("<answer>")
   if len(parts) < 2:  # No <answer> tag found
       return None
   last_part = parts[-1]

   # Extract content up to </answer>
   if "</answer>" not in last_part:
       return None
   answer = last_part.split("</answer>")[0].strip()
   return None if answer == "..." else answer

def extract_answer_from_dataset(text):
   """
   Extracts the answer from the GSM8K dataset examples.

   Args:
       text (str): The dataset example text containing a question and answer.

   Returns:
       str or None: The extracted answer part after the '####' delimiter, or None if not found.

   Explanation:
       1. Checks if the text contains the '####' delimiter that separates question from answer.
       2. If found, splits the text at this delimiter and returns the second part (the answer).
       3. The answer is stripped of leading/trailing whitespace.
       4. Returns None if no delimiter is present.
   """
   if "####" not in text:
       return None
   return text.split("####")[1].strip()

"""
Part 3: Dataset Preparation
We first load the dataset from Hugging Face and then format each example to include a system prompt and a user prompt. 
We also extract the expected answer from the dataset. 
Two helper functions are defined here:
1. prepare_dataset: Loads and prepares the GSM8K dataset by creating a prompt that includes a system prompt (with the formatting instructions) and a user message (the question). It also extracts the answer from the dataset.
2. build_prompt: Concatenates the list of message dictionaries into a single prompt string. This ensures consistency in how the prompt is constructed during both training and inference.
"""
def prepare_dataset(split="train"):
   """
   Load and prepare the GSM8K dataset for training with string prompts.

   Args:
       split (str): The dataset split to load ("train" or "test"). Defaults to "train".

   Returns:
       list: A list of formatted examples, each containing a prompt string and answer.

   Explanation:
       1. Loads the GSM8K dataset from the Hugging Face datasets hub.
       2. For each example in the dataset:
          - Creates a list of messages with system prompt and the question.
          - Converts this list into a single string prompt using build_prompt().
          - Extracts the answer from the dataset example.
          - Creates a formatted example dictionary with prompt and answer.
       3. Returns the list of formatted examples ready for model training or evaluation.
   """
   data = load_dataset('openai/gsm8k', 'main')[split]
   formatted_data = []
   for example in data:
       # Convert list of messages to a single string prompt.
       prompt_str = build_prompt([
           {"role": "system", "content": SYSTEM_PROMPT},
           {"role": "user", "content": example["question"]}
       ])
       formatted_example = {
           "prompt": prompt_str,  # Now a string rather than a list.
           "answer": extract_answer_from_dataset(example["answer"])
       }
       formatted_data.append(formatted_example)
   return formatted_data

def build_prompt(messages):
   """
   Build a single prompt string from a list of messages.

   Args:
       messages (list): A list of message dictionaries, each with 'role' and 'content' keys.

   Returns:
       str: A concatenated string of all message contents.

   Explanation:
       1. Takes a list of message dictionaries in the typical chat format.
       2. Extracts the 'content' field from each message and strips whitespace.
       3. Joins all content strings with newlines to create a single prompt.
       4. This preserves the training format while converting from structured messages to a string.
   """
   return "\n".join([msg["content"].strip() for msg in messages])

"""
Part 4: Evaluation Functions
The evaluation functions perform the following tasks:
1. Tokenize the prompt and generate a response: The model's output is generated given the tokenized prompt.
2. Extract the predicted answer: The answer is extracted from the generated response.
3. Compare the predicted answer with the expected answer: This comparison is done using exact matching as well as numeric equivalence checks.
Two helper functions, _extract_last_number and _extract_single_number, are used to extract numbers from text. 
The main evaluation function, evaluate_model, uses these helpers to determine if the predicted answer is correct:
"""
def extract_last_number(text):
   """
   Extracts the last number appearing in the text.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The last number in the text, or None if no number is found.

   Explanation:
       1. Removes dollar signs and percent symbols from the text.
       2. Uses regex to find a number that appears at the end of the text (possibly after whitespace).
       3. The pattern matches numbers that appear at the end of the string, with or without decimal points.
       4. Returns the found number as a float, or None if no match is found.
   """
   text = text.replace('$', '').replace('%', '')
   pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
   match = re.search(pattern, text)
   return float(match.group(1)) if match else None

def extract_single_number(text):
   """
   Extracts a single number from text if exactly one number is present.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The single number in the text, or None if zero or multiple numbers are found.

   Explanation:
       1. Uses regex to find all numbers in the text (including negative numbers and decimals).
       2. If exactly one number is found, returns it as a float.
       3. If zero or multiple numbers are found, returns None.
   """
   numbers = re.findall(r'-?\d*\.?\d+', text)
   return float(numbers[0]) if len(numbers) == 1 else None

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

   for example in eval_examples:
       # Get the prompt and expected answer
       full_prompt = example["prompt"]
       expected = example["answer"]

       # Tokenize and generate response
       inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
       with torch.no_grad():
           outputs = model.generate(
               inputs,
               max_new_tokens=512,
               temperature=0.7,
               num_return_sequences=1,
               pad_token_id=tokenizer.pad_token_id,
               eos_token_id=tokenizer.eos_token_id,
               forced_eos_token_id=tokenizer.eos_token_id,
               early_stopping=False,
           )
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)

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
        #    print(expected)
        #    print("\nExtracted Answer:")
        #    print(predicted)
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


"""
Part 5: Reward Functions
In our pipeline, we define two reward functions:
1. correctness_reward:
This function assigns rewards based on whether the generated answer is correct. 
It compares the extracted answer from the model output with the expected answer, using both exact string matching and numeric equivalence checks. A exact match earns a higher reward (2.0), while a match based on numeric equivalence receives a smaller reward (1.5).
2. format_reward:
This function encourages the model to adhere to the desired XML-like output format. 
It provides a small reward for the presence of the <reasoning>, </reasoning>, <answer>, and </answer> tags in the generated text. 
We use a relatively value of 0.05 for each of the four pieces because the model is already capable of using these tags from previous supervised finetuning step, so we give this small reward so that it doesn't forget to do that because of the RL updates.
"""
def correctness_reward(prompts, completions, answer, **kwargs):
   """
   Assigns a reward based on the correctness of the model's answer.

   Args:
       prompts (list): List of input prompts.
       completions (list): List of model completions, each containing content.
       answer (list): List of expected answers.
       **kwargs: Additional keyword arguments.

   Returns:
       list: List of numerical rewards for each completion.

   Explanation:
       1. Extracts the content from each completion.
       2. Extracts the answer portion from each response using extract_answer_from_model_output.
       3. Assigns rewards based on matching criteria:
          - 2.0 points for an exact match
          - 1.5 points for numeric equivalence (when values match but format differs)
          - 0.0 points for incorrect answers
       4. Tracks completion lengths for analysis.
   """
   responses = [completion[0]['content'] for completion in completions]
   extracted = [extract_answer_from_model_output(r) for r in responses]
   rewards = []
   for r, a in zip(extracted, answer):
       if r == a:  # Exact match case
           rewards.append(2.0)
       else:
           # Try numeric equivalence
           r_num = extract_single_number(str(r))
           a_num = extract_single_number(str(a))
           if r_num is not None and a_num is not None and r_num == a_num:
               rewards.append(1.5)
           else:
               rewards.append(0.0)
   # Log completion lengths
   completion_lengths = [len(response.split()) for response in responses]
   return rewards

def format_reward(completions, **kwargs):
   """
   Assigns a reward for adhering to the desired XML format.

   Args:
       completions (list): List of model completions, each containing content.
       **kwargs: Additional keyword arguments.

   Returns:
       list: List of format compliance scores for each completion.

   Explanation:
       1. Extracts the content from each completion.
       2. Evaluates format compliance by checking for required XML tags:
          - 0.2 points for each tag present (<reasoning>, </reasoning>, <answer>, </answer>)
          - Maximum score of 0.8 for perfect format compliance
       3. Stores and returns the format compliance scores.
   """
   responses = [completion[0]['content'] for completion in completions]
   rewards = []
   format_scores = []
   for response in responses:
       score = 0.0
       if "<reasoning>" in response: score += 0.2
       if "</reasoning>" in response: score += 0.2
       if "<answer>" in response: score += 0.2
       if "</answer>" in response: score += 0.2
       rewards.append(score)
       format_scores.append(score)
   return rewards


def eos_reward(completions, **kwargs):
   """
   Assigns a reward for properly terminating generation after </answer> tag.

   Args:
       completions (list): List of model completions, each containing content.
       **kwargs: Additional keyword arguments.

   Returns:
       list: List of EOS compliance scores for each completion.

   Explanation:
       1. Checks if the completion contains </answer> tag.
       2. Measures how much text appears after the </answer> tag.
       3. Rewards completions that stop soon after </answer>:
          - 0.5 points if </answer> is present and less than 20 chars follow
          - 0.3 points if </answer> is present and less than 50 chars follow
          - 0.1 points if </answer> is present but more text follows
          - 0.0 points if </answer> is not present
       4. This encourages the model to learn to stop generating after answering.
   """
   responses = [completion[0]['content'] for completion in completions]
   rewards = []
   for response in responses:
       if "</answer>" not in response:
           rewards.append(0.0)
           continue

       # Find position after the last </answer> tag
       last_answer_pos = response.rfind("</answer>")
       text_after_answer = response[last_answer_pos + len("</answer>"):].strip()

       # Reward based on how cleanly the model stopped
       if len(text_after_answer) == 0:
           # Perfect - stopped right after </answer>
           rewards.append(0.5)
       elif len(text_after_answer) < 20:
           # Good - only a few chars after (maybe whitespace or EOS token artifacts)
           rewards.append(0.4)
       elif len(text_after_answer) < 50:
           # Okay - some extra text but not too much
           rewards.append(0.2)
       else:
           # Has </answer> but continued generating significantly
           rewards.append(0.0)

   return rewards

def combined_reward(prompts, completions, answer):
   """
   Combines correctness, format, and EOS rewards.

   Args:
       prompts (list[str]): List of prompt texts
       completions (list[list[dict]]): List of completion dictionaries
       answer (list[str]): List of expected answers

   Returns:
       list[float]: Combined rewards for each prompt-completion pair

   Explanation:
       1. Calculates separate rewards for correctness, format compliance, and EOS behavior.
       2. Combines the rewards with the following weights:
          - Correctness score range: 0.0 to 2.0
          - Format score range: 0.0 to 0.8
          - EOS score range: 0.0 to 0.5
          - Total possible range: 0.0 to 3.3
       3. Returns the combined reward for each example.
   """
   # Get individual rewards
   correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
   format_scores = format_reward(completions=completions)
   eos_scores = eos_reward(completions=completions)

   # Combine rewards - correctness is weighted more heavily
   combined_rewards = []
   for c_score, f_score, e_score in zip(correctness_scores, format_scores, eos_scores):
       # Correctness score range: 0.0 to 2.0
       # Format score range: 0.0 to 0.8
       # EOS score range: 0.0 to 0.5
       # Total range: 0.0 to 3.3
       combined_rewards.append(c_score + f_score + e_score)

   return combined_rewards