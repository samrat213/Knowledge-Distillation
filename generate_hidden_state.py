import os
HTTP_PROXY = 'http://10.10.78.61:3128'
HTTPS_PROXY = 'http://10.10.78.61:3128'

os.environ['http_proxy'] = HTTP_PROXY
os.environ['https_proxy'] = HTTPS_PROXY

# set path for locally downloaed models
student_path = "models/metaresearch/llama-3.2/transformer/1b"
teacher_path = "models/metaresearch/llama-3.1/transformers/8b/2"
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from evaluate import load
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn
import os
import numpy as np
import json
import threading
import datetime
# Load and process datasets
def load_datasets():
    # Summarization
    cnn_dm = load_dataset("cnn_dailymail", "3.0.0", split="train[:10000]")
    
    # Question Answering
    squad = load_dataset("squad_v2", split="train[:10000]")
    
    # Paraphrase Generation
    # quora = load_dataset("quora", split="train[:3]")
    # quora = load_dataset("quora", split="train[:363861]")
    quora = load_dataset("quora", split="train[:30000]")
    print("train dataset loaded")
    
    return {
        "summarization": cnn_dm,
        "qa": squad,
        "paraphrase": quora
    }
# Format datasets into consistent prompt structure
def format_datasets(datasets):
    formatted = {}
    
    # Summarization formatting
    summarization_data = []
    for example in datasets["summarization"]:
        prompt = f"Summarize the following article:\n{example['article']}"
        target = example['highlights']
        summarization_data.append({"prompt": prompt, "target": target, "task": "summarization"})
    formatted["summarization"] = summarization_data
    
    # QA formatting
    qa_data = []
    for example in datasets["qa"]:
        prompt = f"Context: {example['context']}\nQuestion: {example['question']}"
        target = example['answers']['text'][0] if len(example['answers']['text']) > 0 else "No answer available."
        qa_data.append({"prompt": prompt, "target": target, "task": "qa"})
    formatted["qa"] = qa_data
    
    # Paraphrase formatting
    paraphrase_data = []
    for example in datasets["paraphrase"]:
        if example['is_duplicate']:  # Only use duplicate pairs for paraphrasing
            prompt = f"Paraphrase the following:\n{example['questions']['text'][0]}"
            target = example['questions']['text'][1]
            paraphrase_data.append({"prompt": prompt, "target": target, "task": "paraphrase"})
    formatted["paraphrase"] = paraphrase_data
    print("format the data in desired form")
    return formatted
## teacher model setup

class TeacherModel:
    def __init__(self, model_path=teacher_path, gpu=3):
        print("*** Initializing teacher model ***")
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=self.device)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,

            device_map=self.device
        )
        # Set model config pad_token_id
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()  # Set to evaluation mode
        
    def generate_outputs(self, prompts, max_new_tokens=20):
        """Generate logits and outputs from the teacher model"""
        outputs = []
        logits_list = []
        # hidden_states_list = []
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            return_attention_mask=True
        ).to(self.device)
        
        with torch.no_grad():
            # Generate text outputs
            torch.cuda.empty_cache()

            model_output = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True # As in original code
            )
            # model_output = model_output.to(self.device)
            # Get logits for each generation step
            # Stack all scores (logits) from each generation step
            hidden_state = model_output.hidden_states[-1]
            
            hidden_state = hidden_state.to(self.device)
            # Split logits to match each prompt's output
            hidden_state_list = []
            # `attention_mask` is 1 for real tokens, 0 for padding. Summing gives actual length.
            actual_lengths = inputs.attention_mask.sum(dim=1)
            for i in range(hidden_state.size(0)): # Iterate over batch
                prompt_len = actual_lengths[i].item()
                # Slice to get logits only for the actual tokens of this prompt
                # Shape: (prompt_len, vocab_size)
                prompt_hidden_state = hidden_state[i, :prompt_len, :]
                # Add a leading batch dimension of 1 as per the original implied output structure
                hidden_state_list.append(prompt_hidden_state.unsqueeze(0))

            del model_output  
        return hidden_state_list
    
# Load and format datasets
datasets = load_datasets()
formatted_data = format_datasets(datasets)

# Flatten all tasks into one list for inference
all_data = formatted_data["summarization"] + formatted_data["qa"] + formatted_data["paraphrase"]

# Extract prompts only
prompts = [example["prompt"] for example in all_data]

# Initialize teacher model
# teacher1 = TeacherModel(model_path=teacher_path, gpu=0)
# teacher2 = TeacherModel(model_path=teacher_path, gpu=1)
# teacher3 = TeacherModel(model_path=teacher_path, gpu=2)
teacher4 = TeacherModel(model_path=teacher_path, gpu=3)

print("number of prompt: ", len(prompts))
class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None):
        super().__init__(group=group, target=target, name=name,
                         args=args, kwargs=kwargs or {}, daemon=daemon)
        self._return = None
        self._exception = None # To store exception if one occurs

    def run(self):
        if self._target is not None:
            try:
                self._return = self._target(*self._args, **self._kwargs)
            except Exception as e:
                self._exception = e # Store the exception

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self._exception:
            raise self._exception # Re-raise exception in the main thread
        return self._return

import threading
import time

from collections import defaultdict
class thread_out():
    def __init__(self):
        self.out= defaultdict()
        self.i=0
    def add_data(self, prompts, logits):
        self.i+=1
        for prompt, logits_tensor in zip(prompts, logits):
        # Convert logits tensor to list (removes batch dim)
            self.out[prompt] = logits_tensor.squeeze(0).tolist()
        if self.i%5==0:
            self.save()
    
    def save(self, path='./_logists.json'):
        with open(path, 'w') as f:
            json.dump(self.out, f)


# tokenizer=teacher1.tokenizer


active_threads = []

max_concurrent_threads = max(1, os.cpu_count() - 10) if os.cpu_count() else 4 # Leave one CPU for main/GPU tasks
# 
logits_dict = {}
logits=[]
batch_size = 4
max_new_tokens=100
# output_path="prompt_logits.json"
obj = thread_out()
from safetensors.torch import save_file

tensors = {}

for i in tqdm(range(0, len(prompts), batch_size), desc="Generating hidden_state"):
    batch_prompts = prompts[i:i+batch_size]
    batch_hidden_state=[]
    # _, batch_hidden_state = teacher1.generate_outputs(batch_prompts, tokenizer, max_new_tokens=max_new_tokens)
    # active_threads = [t for t in active_threads if t.is_alive()]
    # while len(active_threads) >= max_concurrent_threads:
    #     print(f"Max concurrent threads ({max_concurrent_threads}) reached. Waiting...") # Debug
    #     time.sleep(0.1) # Brief pause
    #     active_threads = [t for t in active_threads if t.is_alive()] # Re-check

    # t1 = ThreadWithReturnValue(target=teacher1.generate_outputs, args=(batch_prompts[:10], max_new_tokens))
    # t2 = ThreadWithReturnValue(target=teacher2.generate_outputs, args=(batch_prompts[10:20], max_new_tokens,))
    # t3 = ThreadWithReturnValue(target=teacher3.generate_outputs, args=(batch_prompts[20:30], max_new_tokens,))
    # t4 = ThreadWithReturnValue(target=teacher4.generate_outputs, args=(batch_prompts[30:], max_new_tokens,))
    # t1.start()
    # t2.start()
    # t3.start()
    # t4.start()

    # active_threads.append(t1)
    # active_threads.append(t2)
    # active_threads.append(t3)
    # active_threads.append(t4)

    # batch_hidden_state.extend(t1.join())
    # batch_hidden_state.extend(t2.join())
    # batch_hidden_state.extend(t3.join())
    # batch_hidden_state.extend(t4.join())
    # yen
    # thread1 = threading.Thread(target=obj.add_data, args=(batch_prompts,batch_hidden_state,))
    # thread1.start()
    # s = datetime.datetime.now()
    batch_hidden_state= teacher4.generate_outputs(batch_prompts)
    for prompt, hidden_state_tensor in zip(batch_prompts, batch_hidden_state):

        # top_k_values, top_k_indices = torch.topk(logits_tensor, 100, dim=-1)
        tensors[prompt] = hidden_state_tensor
        # tensors[prompt] = logits_tensor
        # Convert logits tensor to list (removes batch dim)
        # logits_dict[prompt] = logits_tensor.squeeze(0).tolist()
    # print(datetime.datetime.now()-s)
    if i%100==0:
        torch.save(tensors, "hidden_state.pt")

    del batch_hidden_state


# Save to JSON
# with open(output_path, "w") as f:
#     json.dump(logits_dict, f)
# obj.save()
# print(f"Saved logits for {len(logits_dict)} prompts to {output_path}")
# for t in tqdm(active_threads, desc="Joining threads"):
#     t.join()

torch.save(tensors, "hidden_state.pt")

# obj.save()
