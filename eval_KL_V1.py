import os
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
#from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from evaluate import load
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
HTTP_PROXY = 'http://10.10.78.61:3128'
HTTPS_PROXY = 'http://10.10.78.61:3128'

os.environ['http_proxy'] = HTTP_PROXY
os.environ['https_proxy'] = HTTPS_PROXY

# set path for locally downloaed models
# student_path = "models/metaresearch/llama-3.2/transformer/1b"
teacher_path = "models/metaresearch/llama-3.1/transformers/8b/2"
# adapter_path = "checkpoints/llama_new_student_logits_Temp_1/epoch_0"
result_path =     "results/teacher"
DEVICE = "cuda"

## Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from transformers.utils import logging
logging.set_verbosity_error()

def load_test_split():
    # Summarization
    cnn_dm = load_dataset("cnn_dailymail", "3.0.0", split="test[:500]")
    # Question Answering
    squad = load_dataset("squad_v2", split="validation[:500]")
    # Paraphrase Generation
    quora = load_dataset("quora", split="train[362000:363000]")
    print("test dataset loaded")
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

data = load_test_split()
test_data = format_datasets(data)

# class StudentSystem:
#     def __init__(self, model_path=student_path,adapters_root=adapter_path, device=None):
#         self.device = DEVICE if torch.cuda.is_available() else "cpu"
#         print("*** Initializing student model ***")

#         # Load the base model
#         self.base_model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             device_map=self.device,
#         )

#         # Initialize adapter storage
#         self.task_adapters = {}
#         self.adapters_root = adapters_root

#         # Load trained adapters
#         self.setup_task_adapters()

#     def setup_task_adapters(self):
#         """Load trained LoRA adapters for each task"""

#         # Define adapter subfolders (must match the names used when saving)
#         adapter_tasks = ["summarization", "qa", "paraphrase"]

#         # Wrap model with PEFT so we can load adapters
#         lora_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             r=8,
#             lora_alpha=32,
#             lora_dropout=0.1,
#             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
#         )
#         self.base_model = get_peft_model(self.base_model, lora_config)

#         for task in adapter_tasks:
#             path = f"{self.adapters_root}/{task}"
#             print(f"Loading adapter for task '{task}' from: {path}")
#             self.base_model.load_adapter(path, adapter_name=task)
#             self.task_adapters[task] = path  # optional, for reference

#         print("Adapters loaded:", list(self.task_adapters.keys()))

#     def task_router(self, prompt):
#         """Determine which task adapter to use based on the prompt"""
#         prompt_lower = prompt.lower()
#         if any(term in prompt_lower for term in ["summarize", "summary", "summarization"]):
#             return "summarization"
#         elif any(term in prompt_lower for term in ["paraphrase", "rephrase", "rewrite"]):
#             return "paraphrase"
#         elif any(term in prompt_lower for term in ["question", "answer", "context"]):
#             return "qa"
#         else:
#             return "summarization"  # default

#     def generate(self, prompt, tokenizer, max_new_tokens=512):
#         """Generate response using the appropriate task adapter"""
#         task = self.task_router(prompt)
#         self.base_model.set_adapter(task)

#         inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

#         with torch.no_grad():
#             outputs = self.base_model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=False
#             )

#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response, task
# student = StudentSystem(student_path,adapter_path)
# tokenizer = AutoTokenizer.from_pretrained(student_path,device_map=DEVICE)
# tokenizer.pad_token = tokenizer.eos_token

## teacher model setup

class TeacherModel:
    def __init__(self, model_path=teacher_path):
        print("*** Initializing teacher model ***")
        self.device = DEVICE if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_uant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map=DEVICE
        )
        # Set model config pad_token_id
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()  # Set to evaluation mode
        
    def generate(self, prompts, tokenizer, max_new_tokens):
        """Generate logits and outputs from the teacher model"""
        outputs = []
        logits_list = []
        # hidden_states_list = []
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            return_attention_mask=True
        ).to(self.device)
        
        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Get full output distribution (logits)
                model_output = self.model(**inputs, output_hidden_states=False)
                logits = model_output.logits
                # hidden_states = model_output.hidden_states[-1]  # Last layer hidden states
                
                # Generate text output
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
                
                outputs.append(decoded)
                logits_list.append(logits)
                # hidden_states_list.append(hidden_states)
                
        return outputs, logits_list
teacher = TeacherModel(teacher_path)
tokenizer = AutoTokenizer.from_pretrained(teacher_path,device_map=DEVICE)
tokenizer.pad_token = tokenizer.eos_token
rouge = load("rouge",device_map=DEVICE)
bert_score = load("bertscore",device_map=DEVICE)
sacrebleu = load("sacrebleu",device_map=DEVICE)
meteor = load("meteor",device_map=DEVICE)
results = {
        "summarization": {"rouge_l": []},
        "qa": {"rouge_l": [], "bert_score": []},
        "paraphrase": {"sacrebleu": [], "meteor": []}
    }
# 
for task, examples in test_data.items():
    if not examples:
        print("No examples")
        continue
    for example in tqdm(examples, desc=f"Evaluating {task}"):
        prompt = example["prompt"]
        reference = example["target"]
        if not reference.strip():
            continue

        # Generate prediction from the teacher model
        prediction_list, _ = teacher.generate([prompt], tokenizer, max_new_tokens=512)
        prediction = prediction_list[0]

        if not prediction.strip():
            continue

        # Evaluate using task-specific metrics
        if task == "summarization":
            rouge_scores = rouge.compute(predictions=[prediction], references=[reference])
            results[task]["rouge_l"].append(rouge_scores['rougeL'])

        elif task == "qa":
            rouge_scores = rouge.compute(predictions=[prediction], references=[reference])
            bert_scores = bert_score.compute(predictions=[prediction], references=[reference], lang="en")

            results[task]["rouge_l"].append(rouge_scores['rougeL'])
            results[task]["bert_score"].append(bert_scores["f1"][0])

        elif task == "paraphrase":
            sacrebleu_score = sacrebleu.compute(predictions=[prediction], references=[[reference]])
            meteor_score = meteor.compute(predictions=[prediction], references=[reference])

            results[task]["sacrebleu"].append(sacrebleu_score["score"])
            results[task]["meteor"].append(meteor_score["meteor"])

aggregated = {}
for task, metrics in results.items():
    aggregated[task] = {}
    for metric_name, scores in metrics.items():
        aggregated[task][metric_name] = sum(scores) / len(scores)

# Create the directory if it doesn't exist
os.makedirs(result_path, exist_ok=True)

# File name for the metrics
output_file = os.path.join(result_path, "aggregated_metrics.json")

# Save the dictionary
with open(output_file, "w") as f:
    json.dump(aggregated, f, indent=2)
print(f"Aggregated metrics saved to: {output_file}")

