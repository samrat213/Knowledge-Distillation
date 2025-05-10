import torch
from datetime import datetime
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os
from tqdm import tqdm
from peft import prepare_model_for_kbit_training,LoraConfig,PeftModel,get_peft_model
from transformers import GenerationConfig
from accelerate import Accelerator
from time import perf_counter
HTTP_PROXY = 'http://10.10.78.61:3128'
HTTPS_PROXY = 'http://10.10.78.61:3128'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# os.environ["LD_LIBRARY_PATH"]="/home/naveeta/anaconda3/envs/nlp/lib:$LD_LIBRARY_PATH"
# os.environ["LIBRARY_PATH"]="/home/naveeta/anaconda3/envs/nlp/lib:$LIBRARY_PATH"
os.environ['http_proxy'] = HTTP_PROXY
os.environ['https_proxy'] = HTTPS_PROXY
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# model_id="meta-llama/Meta-Llama-3.1-8B"
# model_path = "models/metaresearch/llama-3.1/transformers/8b/2"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=False,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# # Loading the model and tokenizer

# model = AutoModelForCausalLM.from_pretrained(model_path,
#                                             quantization_config=bnb_config,
#                                             device_map="cuda:0")
# tokenizer = AutoTokenizer.from_pretrained(
#     model_path,
#     model_max_length=512,
#     padding_side="left",
#     add_eos_token=True)
# tokenizer.pad_token = tokenizer.eos_token

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def load_datasets():
    # Summarization
    cnn_dm = load_dataset("cnn_dailymail", "3.0.0", split="train[:3000]")
    
    # Question Answering
    squad = load_dataset("squad_v2", split="train[:3000]")
    
    # Paraphrase Generation
    # quora = load_dataset("quora", split="train[:3]")
    # quora = load_dataset("quora", split="train[:363861]")
    quora = load_dataset("quora", split="train[:10000]")
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


INSTRUCTION = {
    'qa'            :   "You are a helpful assistant, who answer for the question using context given.",
    'summarization' :   "You are a helpful assistant, who summarize the article given.",
    'paraphrase'    :   "You are a helpful assistant, who paraphrase the context given.",
}

# EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["task"]
    inputs       = examples["prompt"]
    outputs      = examples["target"]
    texts = []
    text = f'<|im_start|>system\n{INSTRUCTION[instructions]}<|im_end|>\n<|im_start|>user\n{inputs}<|im_end|>\n<|im_start|>assistant\n{outputs}<|im_end|>\n'
    # for instruction, input, output in zip(instructions, inputs, outputs):
    #     # Must add EOS_TOKEN, otherwise your generation will go on forever!
    #     text = text + EOS_TOKEN
    #     texts.append(text)
    # return { "text" : text, }
    # return tokenize(text)
    return text

datasets = load_datasets()
formatted_data = format_datasets(datasets)
all_data = formatted_data["summarization"] + formatted_data["qa"] + formatted_data["paraphrase"]


dataset = [formatting_prompts_func(example) for example in tqdm(all_data)]
# dataset = Dataset.from_list(dataset)

# device
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch import optim
model_path = "models/metaresearch/llama-3.1/transformers/8b/2"

class TeacherModel:
    def __init__(self, model_path=model_path, gpu=0):
        print("*** Initializing teacher model ***")
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.accelerator = Accelerator()
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
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
            device_map='balanced'
            # device_map="auto"
        )
        # Set model config pad_token_id
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()  # Set to evaluation mode
        
    def train_step(self, prompts, optimizer):
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
        

        model_output = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=False # As in original code,
        )

        all_prompt_logits = model_output.logits
        targets = inputs["input_ids"]
        shift_logits = all_prompt_logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()

        min_len = min(shift_logits.size(1), shift_labels.size(1))
        shift_logits = shift_logits[:, :min_len, :]
        shift_labels = shift_labels[:, :min_len]

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )
        
        # loss = outputs.lossn
        optimizer.zero_grad()
        self.accelerator.backward(loss)
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        # progress_bar.update(1)
        del all_prompt_logits
        del shift_logits
        del shift_labels
        return loss.item()

teacher = TeacherModel(model_path=model_path)



BATCH_SIZE = 4 # Keep small for testing
# train_dataloader = DataLoader(
#     teacher_output,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=4 # Set to 0 for easier debugging initially, then increase
#     # The default collate_fn will stack the tensors from __getitem__
# )
TASKS = ["summarization", "qa", "paraphrase"]
RUN = "llama8b_SFT"
# Initialize distillation trainer
writer = SummaryWriter(f'./logs/fintuning_8b')

# Optimizer - one per task adapter

# optimizers = {
#     task: torch.optim.AdamW(adapter.parameters(), lr=5e-5)
#     for task, adapter in student.base_model.active_adapters.items()
# }
epochs = 2
batch_size = 3
print("traing loop starts")
# Training loop
optimizer = torch.optim.AdamW(teacher.model.parameters(), lr=5e-4)

# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=epoch/10)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

my_dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)

my_dataloader, teacher.model, optimizer = teacher.accelerator.prepare(
    my_dataloader, teacher.model, optimizer
)
step=0

for epoch in range(epochs):
    total_loss = 0
    total_examples = 0
    i=0
    for batch in tqdm(my_dataloader,desc=f"Epoch {epoch+1}/{epochs}"):
        # Group examples by task
        batch_prompts = batch[i:i+batch_size]
        loss = teacher.train_step(batch_prompts, optimizer)
        step+=1
        writer.add_scalar('loss', loss, step)

        if step%500==0:
            teacher.model.save_pretrained(f"./checkpoints/{RUN}/step_{step}")

teacher.model.save_pretrained(f"./checkpoints/{RUN}/final")


writer.flush()