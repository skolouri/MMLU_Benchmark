# Load the necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer # To load model
from datasets import load_dataset  # To load MMLU
from accelerate import Accelerator # For multi-GPU inference
from accelerate.utils import gather_object
import os
import torch
import re 
import numpy as np

# Load mmlu category names
mmlu_categories = np.load('mmlu_categories.npy', allow_pickle=True)
# Choose a specific category
category = mmlu_categories[-5]  # Category of interest
mmlu = load_dataset("tasksource/mmlu", category) # Load the category


#@title format_prompt - Formats an MMLU example into a customized prompt
def format_prompt(example, category):
    """
    Formats an MMLU example into a customized prompt
    Inputs
      example: An MMLU question
      category: The category of the question
    Outputs
      prompt: The formatted prompt
    """
    prompt = (
        f"You are an expert in the field of {category}. "
        f"Please answer the following multiple-choice question accurately:\n\n"
        f"Question: {example['question']}\n"
        f"Options:\n"
    )
    for i, choice in enumerate(example['choices'], start=1):
        prompt += f"{i}. {choice}\n"
    prompt += "\nAnswer by providing the number corresponding to the correct choice (e.g., '1', '2', etc.)."
    prompt += "\n Answer: \n"
    return prompt


# Generate the prompts
set = 'dev' # 'dev' or 'test'
mmlu_tests = mmlu[set] # I use 'dev' instead of 'test' for faster evaluation
                         # This must be changed to 'test' the actual eval.
prompts_all = [format_prompt(mmlu_test, category) for mmlu_test in mmlu_tests]

#@title find_last_boxed_content - Gets the last \boxed{item}
def find_last_boxed_content(response):
    last_index = response.rfind(r"Answer: ")
    if last_index == -1:
        return None  # No match found

    # Extract content using regex from the last occurrence
    # match = re.search(r"\\boxed\{(.*?)\}", response[last_index:])

    # if match:
    #     return match.group(1)  # Return only the content inside \boxed{}
    return response[last_index+8:last_index+9]
  
  
device = "cuda"
accelerator = Accelerator()

# Load a pretrained LLM and tokenizer
model_names = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
]
  

for model_name in model_names:
  # First lets create a results folder to collect the results
  folder_path = "./results/"+model_name+"/"+category+""
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  
  
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name,
                                              device_map={"": accelerator.process_index},
                                              torch_dtype=torch.bfloat16,)

  # Ensure the pad_token_id is set properly
  if tokenizer.pad_token_id is None:
      tokenizer.pad_token = tokenizer.eos_token
      tokenizer.pad_token_id = tokenizer.eos_token_id


  # sync all GPUs
  accelerator.wait_for_everyone()

  # divide the prompt list onto the available GPUs
  with accelerator.split_between_processes(prompts_all) as prompts:
    # store output of generations in dict
    results= []
    # have each GPU do inference, prompt by prompt
    for prompt in prompts:
      prompt_tokenized = tokenizer(prompt,
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True).to(device)

      output_tokenized = model.generate(prompt_tokenized.input_ids,
                                        max_length=10000,
                                        temperature=0.5,
                                        pad_token_id=tokenizer.eos_token_id,
                                        eos_token_id=tokenizer.eos_token_id)

      # store outputs and number of tokens in result{}
      results.append(tokenizer.decode(output_tokenized[0],
                                      skip_special_tokens=True))
  # collect results from all the GPUs
  results_gathered=gather_object(results)

  #@title Evaluate Model Predictions
  correct = 0
  for k,result in enumerate(results_gathered):
    predicted_answer = find_last_boxed_content(result)
    # print(len(result))
    if predicted_answer in ['1','2','3','4']:
      predicted_answer = int(predicted_answer)
    elif predicted_answer in ['A','B','C','D']:
      predicted_answer = ord(predicted_answer) - ord('A') + 1
    else:
      predicted_answer = None
    gt_answer = int(mmlu_tests[k]['answer'])+1
    correct += int(gt_answer == predicted_answer)
  print("Model:"+model_name+f", accuracy on {category}: {correct / len(results_gathered) * 100:.2f}%")
  np.save(folder_path+"/"+set+'.npy', results_gathered)
  del model
  del tokenizer
  