import os
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
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.nn import MSELoss

HTTP_PROXY = 'http://10.10.78.61:3128'
HTTPS_PROXY = 'http://10.10.78.61:3128'

os.environ['http_proxy'] = HTTP_PROXY
os.environ['https_proxy'] = HTTPS_PROXY

# set path for locally downloaed models
student_path = "models/metaresearch/llama-3.2/transformer/1b"
teacher_path = "models/metaresearch/llama-3.1/transformers/8b/2"
import argparse

parser = argparse.ArgumentParser("simple_example")
parser.add_argument("--temp", default=1, type=int, required=False)
parser.add_argument("--device", default=3, type=int, required=False)
parser.add_argument("--run", default="llama_hidden_state", type=str, required=False)
args = parser.parse_args()

TEMP = args.temp
DEVICE = args.device
RUN = f"{args.run}_Temp_{TEMP}"
print(f"*** {RUN} ***")
from teacher_with_hidden_state import Teacher_logits

def load_test_split():
    # Summarization
    cnn_dm = load_dataset("cnn_dailymail", "3.0.0", split="test[:600]")
    
    # Question Answering
    squad = load_dataset("squad_v2", split="validation[:600]")
    
    # Paraphrase Generation
    quora = load_dataset("quora", split="train[363861:365861]")
    #quora = load_dataset("quora", split="train[363861:]")
    print("test dataset loaded")
    return {
        "summarization": cnn_dm,
        "qa": squad,
        "paraphrase": quora
    }

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

class StudentSystem:
    def __init__(self,model_path=student_path, device=1):
        self.device = f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu"
        print("*** Initializing student model ***")

        # Load the base model (shared backbone)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
        )
        
        # Initialize task-specific adapters
        self.setup_task_adapters()
        
    def setup_task_adapters(self):
        """Initialize LoRA adapters for each task"""
        
        # Define LoRA configurations for each task
        # Adjust r and alpha based on your parameter budget
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  # Low-rank dimension
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        # Create task-specific LoRA adapters
        self.base_model = get_peft_model(self.base_model, lora_config)

        self.base_model.add_adapter("summarization", lora_config)
        self.base_model.add_adapter("qa", lora_config)
        self.base_model.add_adapter("paraphrase", lora_config)
        
    def task_router(self, prompt):
        """Determine which task adapter to use based on the prompt"""
        prompt_lower = prompt.lower()
        # Simple keyword-based routing
        if any(term in prompt_lower for term in ["summarize", "summary", "summarization"]):
            return "summarization"
        elif any(term in prompt_lower for term in ["paraphrase", "rephrase", "rewrite"]):
            return "paraphrase"
        elif any(term in prompt_lower for term in ["question", "answer", "context"]):
            return "qa"
        else:
            # Default to the most general task or run a more sophisticated classifier
            return "summarization"
    
    def generate(self, prompt, tokenizer, max_new_tokens=100):
        """Generate response using the appropriate task adapter"""
        task = self.task_router(prompt)
        adapter = self.task_adapters[task]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = adapter.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, task
student = StudentSystem(student_path)

#### Knowledge Distillation Process
class KnowledgeDistillation:
    def __init__(self, teacher, student, tokenizer):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.device = student.device
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2048, 4096)
        ).to(self.device)
        self.mse_loss = nn.MSELoss() 
        
    def compute_kd_loss(self, student_logits, teacher_logits, temperature=2.0, attention_mask=None):
        """Compute knowledge distillation loss"""
         # Ensure teacher_logits is a tensor
        if not isinstance(teacher_logits, torch.Tensor):
            teacher_logits = torch.tensor(teacher_logits).to(self.device)
    
        # Add batch dimension if needed 
        if len(teacher_logits.shape) == 2:
            teacher_logits = teacher_logits.unsqueeze(0)
        # Both logits should have the same shape
        if student_logits.shape != teacher_logits.shape:
            # Truncate to the smallest length
            min_length = min(student_logits.shape[1], teacher_logits.shape[1])
            student_logits = student_logits[:, :min_length, :]
            teacher_logits = teacher_logits[:, :min_length, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :min_length]

        # Soften the distributions
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        # Compute KL divergence loss
        if attention_mask is not None:
            # Apply mask to focus on non-padding tokens
            mask = attention_mask.unsqueeze(-1).expand_as(soft_student)
            kd_loss = F.kl_div(
                soft_student * mask, 
                soft_teacher * mask, 
                reduction="sum"
            ) / (mask.sum() + 1e-8)  # Add small epsilon to avoid division by zero
        else:
            kd_loss = F.kl_div(
                soft_student, 
                soft_teacher, 
                reduction="batchmean"
            )
        return kd_loss * (temperature ** 2)
    
    def compute_task_loss(self, student_outputs, targets,attention_mask= None):
        """Compute task-specific loss (e.g., cross-entropy for next token prediction)"""
        # Standard language modeling loss
        shift_logits = student_outputs.logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()

        # Truncate both to the same minimum length
        min_len = min(shift_logits.size(1), shift_labels.size(1))
        shift_logits = shift_logits[:, :min_len, :]
        shift_labels = shift_labels[:, :min_len]
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction='none'
            )
            # print('shift_attention_mask: ',shift_attention_mask.size())
            # print("loss:",loss.size())
            shift_attention_mask = shift_attention_mask[:, :min_len]
            # print('shift_attention_mask changed: ',shift_attention_mask.size())

            loss = loss * shift_attention_mask.reshape(-1)
            task_loss = loss.sum() / shift_attention_mask.sum()
        else:
            task_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )
        return task_loss
    
    def compute_hidden_state_loss(self, student_HS, teacher_HS):
        # print(student_HS.size())
        # print(teacher_HS.size())
        total=0
        for i in range(len(teacher_HS)):
            
            st_HS1 = self.mlp(student_HS[i:i+1,:,:]).to(self.device)
            min_len = min(st_HS1.size(1), teacher_HS[i].size(1))
            t_HS = teacher_HS[i]
            print(st_HS1.size(), t_HS.size())
            loss = self.mse_loss(st_HS1[i:i+1,:min_len,:], t_HS[:,:min_len,:])
            total+= loss
        return total/i

    def train_step(self, batch, task, alpha=0.5):
        """Single training step combining KD and task losses"""
        prompts = batch["prompt"]
        targets = batch["target"]
        
        max_input_length = 1024
        # Prepare inputs for student

        inputs = self.tokenizer(prompts,
                                return_tensors="pt", 
                                padding=True,
                                truncation=True,
                                max_length=max_input_length).to(self.device)
        
        target_ids = self.tokenizer(targets,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=max_input_length).to(self.device)
        
        # Get teacher outputs and logits
        with torch.no_grad():
            teacher_logits_list = self.teacher.get_logits_batch(batch['prompt'])
            teacher_hidden_state_list = self.teacher.get_hidden_state_batch(batch['prompt'])

        # Forward pass through student
        self.student.base_model.set_adapter(task)
        student_outputs = self.student.base_model(**inputs,
                output_hidden_states=True)
        
        actual_lengths = inputs.attention_mask.sum(dim=1)
        # for i in range(all_prompt_logits.size(0)): # Iterate over batch
        #     prompt_len = actual_lengths[i].item()
        #     # Slice to get logits only for the actual tokens of this prompt
        #     # Shape: (prompt_len, vocab_size)
        #     prompt_specific_logits = all_prompt_logits[i, :prompt_len, :]
        #     # Add a leading batch dimension of 1 as per the original implied output structure
        #     logits_list.append(prompt_specific_logits.unsqueeze(0))
        HS_loss = self.compute_hidden_state_loss(student_outputs.hidden_states[-1], teacher_hidden_state_list)

        total_kd_loss = 0
        for i in range(len(prompts)):
            # Extract the individual student logits for this example
            student_logit = student_outputs.logits[i:i+1,:actual_lengths[i], :]  # Keep batch dimension
             # Extract the corresponding teacher logits - convert to tensor if not already
            # print(teacher_logits_list)
            teacher_logit = teacher_logits_list[i].to(self.device)

            if not isinstance(teacher_logit, torch.Tensor):
                teacher_logit = torch.tensor(teacher_logit).to(self.device)

            # Add batch dimension if needed
            if len(teacher_logit.shape) == 2:
                teacher_logit = teacher_logit.unsqueeze(0)

            # Extract attention mask for this example if available
            attention_mask = inputs["attention_mask"][i:i+1] if "attention_mask" in inputs else None
            # print(student_logit.size())
            # print(teacher_logit.size())
            # print(attention_mask.size())
            # Compute KD loss for this example
            example_kd_loss = self.compute_kd_loss(
                student_logit, 
                teacher_logit,
                attention_mask=None,
                temperature=TEMP
            )
        
            total_kd_loss += example_kd_loss
        
         # Average KD loss across batch
        kd_loss = total_kd_loss / len(prompts)
        
        # Compute task loss normally
        task_loss = self.compute_task_loss(
            student_outputs, 
            target_ids["input_ids"],
            attention_mask=target_ids["attention_mask"] if "attention_mask" in target_ids else None
        )
        
        # Combined loss with task-specific weighting
        alpha_map = {
            "summarization": 0.6,  # More weight on KD for summarization
            "qa": 0.5,             # Equal weight for QA
            "paraphrase": 0.4      # More weight on task loss for paraphrase
        }
        task_alpha = alpha_map.get(task, alpha)
        total_loss = task_alpha * kd_loss + (1 - task_alpha) * task_loss + HS_loss
        
        return total_loss, kd_loss, task_loss
    
    
    def validation_loss(self, test_data, writer, epoch):
        print("calculating validation loss")
        # Load metrics

        loss = []
        for task, examples in test_data.items():
            if not examples:
                continue
            for example in tqdm(examples, desc=f"Evaluating {task}"):
                prompt = example["prompt"]
                reference = example["target"]

                if not reference.strip():
                    continue
                
                # Generate prediction
                # prediction, detected_task = self.student.generate(prompt, tokenizer)
                
                inputs = self.tokenizer(prompt,
                                    return_tensors="pt", 
                                    padding=True,
                                    truncation=True,
                                    max_length=512).to(self.device)
                target_ids = self.tokenizer(reference,
                                            return_tensors="pt",
                                            padding=True,
                                            truncation=True,
                                            max_length=512).to(self.device)
                with torch.no_grad():
                    student_outputs = self.student.base_model(**inputs)

                task_loss = self.compute_task_loss(
                                student_outputs, 
                                target_ids["input_ids"],
                                )
                loss.append(task_loss.cpu().item())

        writer.add_scalar('validation loss', np.average(loss), epoch)



    def evaluate_model(self, test_data, tokenizer, writer, epoch):
        print("evaluating results")
        # Load metrics
        rouge = load("rouge")
        bert_score = load("bertscore")
        sacrebleu = load("sacrebleu")
        #meteor = load("meteor")
        
        # results = {
        #     "summarization": {"rouge_l": []},
        #     "qa": {"rouge_l": [], "bert_score": []},
        #     "paraphrase": {"sacrebleu": [], "meteor": []}
        # }
        results = {
            "summarization": {"rouge_l": []},
            "qa": {"rouge_l": [], "bert_score": []},
            "paraphrase": {"sacrebleu": [],}
        }
        loss = []
        for task, examples in test_data.items():
            if not examples:
                continue
            for example in tqdm(examples, desc=f"Evaluating {task}"):
                prompt = example["prompt"]
                reference = example["target"]

                if not reference.strip():
                    continue
                
                # Generate prediction
                prediction, detected_task = self.student.generate(prompt, tokenizer)
                
                inputs = self.tokenizer(prompt,
                                    return_tensors="pt", 
                                    padding=True,
                                    truncation=True,
                                    max_length=512).to(self.device)
                target_ids = self.tokenizer(reference,
                                            return_tensors="pt",
                                            padding=True,
                                            truncation=True,
                                            max_length=512).to(self.device)

                student_outputs = self.student.base_model(**inputs)

                task_loss = self.compute_task_loss(
                                student_outputs, 
                                target_ids["input_ids"],
                                )
                loss.append(task_loss.cpu().item())

                if not prediction.strip():
                    continue
                if task != detected_task:
                    print(task, detected_task)
                    print(prompt)
                # Calculate task-specific metrics
                if detected_task == "summarization":
                    rouge_scores = rouge.compute(predictions=[prediction], references=[reference])
                    results[detected_task]["rouge_l"].append(rouge_scores['rougeL'])
                    
                elif detected_task == "qa":
                    rouge_scores = rouge.compute(predictions=[prediction], references=[reference])
                    bert_scores = bert_score.compute(predictions=[prediction], references=[reference], lang="en")
                    
                    results[detected_task]["rouge_l"].append(rouge_scores['rougeL'])
                    results[detected_task]["bert_score"].append(bert_scores["f1"][0])
                    
                elif detected_task == "paraphrase":
                    sacrebleu_score = sacrebleu.compute(predictions=[prediction], references=[[reference]])
                    #meteor_score = meteor.compute(predictions=[prediction], references=[reference])
                    
                    results[detected_task]["sacrebleu"].append(sacrebleu_score["score"])
                    #results[task]["meteor"].append(meteor_score["meteor"])
        
        writer.add_scalar('loss', np.average(loss), epoch)

        # Aggregate results
        # print(results)
        aggregated = {}
        for task, metrics in results.items():
            aggregated[task] = {}
            for metric_name, scores in metrics.items():
                aggregated[task][metric_name] = sum(scores) / len(scores)
                writer.add_scalar(f'{task}_{metric_name}', aggregated[task][metric_name], epoch)

        
        return aggregated

# def train_student_system(teacher, student, tokenizer, formatted_data, epochs=3):
    # Create dataset and dataloader
tokenizer = AutoTokenizer.from_pretrained(teacher_path)
tokenizer.pad_token = tokenizer.eos_token

teacher_output = Teacher_logits(top_k_path='./logits.pt', device=DEVICE)
epochs = 3

BATCH_SIZE = 6 # Keep small for testing
train_dataloader = DataLoader(
    teacher_output,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4 # Set to 0 for easier debugging initially, then increase
    # The default collate_fn will stack the tensors from __getitem__
)
TASKS = ["summarization", "qa", "paraphrase"]

# Initialize distillation trainer
writer = SummaryWriter(f'./logs/{RUN}')

distiller = KnowledgeDistillation(teacher_output, student, tokenizer)

# Optimizer - one per task adapter
optimizers = torch.optim.AdamW(student.base_model.parameters(), lr=5e-4)

test_datasets = load_test_split()
test_data = format_datasets(test_datasets)
# optimizers = {
#     task: torch.optim.AdamW(adapter.parameters(), lr=5e-5)
#     for task, adapter in student.base_model.active_adapters.items()
# }
print("traing loop starts")
# Training loop
step=0
for epoch in range(epochs):
    total_loss = 0
    total_examples = 0
    i=0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        # Group examples by task
        task_examples = {}
        for i in range(len(batch['prompt'])):
            # print(batch)
            # print(len(batch), batch['task'])
            task = batch['task'][i]
            if task not in task_examples:
                task_examples[task] = {"prompt": [], "target": []}
            task_examples[task]["prompt"].append(batch["prompt"][i])
            task_examples[task]["target"].append(batch["target"][i])
        
        # Process each task separately
        batch_loss = 0
        batch_count =0
        for task, examples in task_examples.items():
            # Skip if no examples for this task in this batch
            if not examples["prompt"]:
                continue
            student.base_model.set_adapter(task)
            # Number of examples for this task
            num_examples = len(examples["prompt"])
            batch_count += num_examples
            # Compute loss
            loss, kd_loss, task_loss = distiller.train_step(examples, task)
        
        
            # Backward pass and optimization
            optimizers.zero_grad()
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(student.base_model.parameters(), max_norm=1.0)
            optimizers.step()

            # Accumulate weighted loss
            batch_loss += loss.item()*num_examples
        
        writer.add_scalar('loss', batch_loss, step)
        step+=1

        total_loss += batch_loss
        total_examples += batch_count
        # break
    # Calculate average loss properly (weighted by number of examples)
    avg_loss = total_loss / total_examples if total_examples > 0 else 0
    writer.add_scalar('avg_loss', avg_loss, epoch)

    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    distiller.validation_loss(test_data, writer, epoch)
    student.base_model.save_pretrained(f"./checkpoints/{RUN}/epoch_{epoch}")


writer.flush()

# Save trained adapters
# for task, adapter in student.task_adapters.items():
#     adapter.save_pretrained(f"./adapter_{task}")

