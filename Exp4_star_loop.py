#This is actually ipynb file

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training, get_peft_model, LoraConfig
)
from transformers.trainer_utils import set_seed
import random

# Set seed for reproducibility
set_seed(42)

# Load Phi-2 model + tokenizer with 4-bit quantization
model_name = "microsoft/phi-2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          padding_side="left",add_eos_token=True, # end of sequence token
                                          add_bos_token=True, # beginning of sequence token
                                          use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # Required for padding
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# Apply LoRA adapters
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[ "Wqkv", "fc1", "fc2" ],  # Phi-2 uses these
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("✅ Model and tokenizer are loaded with LoRA applied.")



#cell 2
import json

# Load your fine_tune_data_improved.jsonl
data_path = "fine_tune_dataset_improved.jsonl"


with open(data_path, "r") as f:
    raw_data = [json.loads(line) for line in f]

# === Reproducible shuffle before split
random.seed(42)
random.shuffle(raw_data)

# === 100 for star-style rationale generation
total_len = len(raw_data)
split = total_len - 100 
direct_data = raw_data[:split]  # used for supervised fine-tuning
star_data = raw_data[split:]    # used for STaR-style rationale generation

# === Save both splits
with open("star_direct.jsonl", "w") as f:
    for item in direct_data:
        json.dump(item, f)
        f.write("\n")

with open("star_remaining.jsonl", "w") as f:
    for item in star_data:
        json.dump(item, f)
        f.write("\n")
        
#cell 3:

from datasets import Dataset

with open("star_direct.jsonl", "r") as f:
    raw_data = [json.loads(line) for line in f]

# Convert to HuggingFace Dataset
dataset = Dataset.from_list([
    {"text": f'Question: {item["question"]}\nAnswer: {item["answer"]}'} for item in raw_data
])


print(f"✅ Loaded {len(dataset)} examples.")
dataset[0]  # Peek at one example



#cell 4
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])

print("✅ Tokenization complete.")
print(tokenized_dataset[0])

#cell 5

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./phi2-lora-star-sft",
    per_device_train_batch_size=6,               # safe for 12GB GPU
    gradient_accumulation_steps=2,               # simulates batch size 12
    num_train_epochs=2,                          # increase from 0.5
    learning_rate=1e-4,                          # conservative LR for stability
    fp16=True,                                   # enable mixed precision
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    optim="paged_adamw_8bit",
    # lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    evaluation_strategy="no",                    # can enable if dev set added
    report_to="none",                            # no W&B logs
    dataloader_num_workers=4                     # parallel data loading
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("✅ Trainer is ready.")

#cell 6
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
trainer.train()

#STaR Section
#cell 7
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Path where your fine-tuned model checkpoint was saved
adapter_path = "./phi2-lora-rationale/checkpoint-1030"

# Load base model and apply the LoRA adapter
base_model = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter_path)

model.eval()
print("✅ Direct Fine-tuned model loaded.")



#cell 8
import torch
import re

# Load the the star_remaining_50.jsonl
with open("star_remaining.jsonl", "r") as f:
    star_data =[json.loads(line) for line in f]

def format_star_prompt(review):
    return f'Question: {item["question"]}\nAnswer: '

def extract_star_sentiment(generated_text):
    sentiment_match = re.search(r"classify.*?as\s+(POSITIVE|NEGATIVE)", generated_text, re.IGNORECASE)
    sentiment = sentiment_match.group(1).upper() if sentiment_match else "UNKNOWN"
    return sentiment

def extract_star_rationale(generated_text):
    _, _, answer = generated_text.partition('Answer:')
    return answer.strip()

# Evaluate the model on the star_remaining_50.jsonl data
model.eval()

new_correct_dataset = []
new_incorrect_dataset = []

for item in star_data:
    prompt = format_star_prompt(item['question'])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,  
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id

        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sentiment = extract_star_sentiment(output_text)
    # use rationales of correct answer only
    if sentiment == extract_star_sentiment(item['answer']):
        new_correct_dataset.append({"question": item['question'], "answer": extract_star_rationale(output_text)})
    else:
        # rationalization 
        prompt = f"Question: {item['question']}. You know that this review is {extract_star_sentiment(item['answer'])}.\nAnswer: "
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,  
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        rationale = extract_star_rationale(output_text)
        new_incorrect_dataset.append({"question": item['question'], "answer": rationale})

# Save the new datasets
with open("new_correct_dataset.jsonl", "w") as f:
    for item in new_correct_dataset:
        json.dump(item, f)
        f.write("\n")
with open("new_incorrect_dataset.jsonl", "w") as f:
    for item in new_incorrect_dataset:
        json.dump(item, f)
        f.write("\n")


#Loop 2

from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# === Step 1: Define adapter path (root folder where you copied LoRA files)
adapter_path = "phi2-lora-star-sft"

# === Step 2: Load the adapter config locally
peft_config = PeftConfig.from_pretrained(adapter_path, is_local=True)

# === Step 3: Load the base model with 4-bit quantization (must match Loop 1)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# === Step 4: Prepare for LoRA fine-tuning
base_model = prepare_model_for_kbit_training(base_model)

# === Step 5: Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path, is_local=True)

# === Step 6: (Optional) Force LoRA layers to be trainable if needed
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True

# === Step 7: Verify
model.print_trainable_parameters()


# cell 

import json
from datasets import Dataset

with open("combined_star_data.jsonl", "r") as f:  #replaced, before it was star_direct
    raw_data = [json.loads(line) for line in f]

# Convert to HuggingFace Dataset
dataset = Dataset.from_list([
    {"text": f'Question: {item["question"]}\nAnswer: {item["answer"]}'} for item in raw_data
])


print(f"✅ Loaded {len(dataset)} examples.")
dataset[0]  # Peek at one example

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])

print("✅ Tokenization complete.")
print(tokenized_dataset[0])

# cell 
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./phi2-lora-star-combined",
    per_device_train_batch_size=6,               # safe for 12GB GPU
    gradient_accumulation_steps=2,               # simulates batch size 12
    num_train_epochs=4,                          # increase from 0.5
    learning_rate=1e-4,                          # conservative LR for stability
    fp16=True,                                   # enable mixed precision
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    optim="paged_adamw_8bit",
    # lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    evaluation_strategy="no",                    # can enable if dev set added
    report_to="none",                            # no W&B logs
    dataloader_num_workers=4                     # parallel data loading
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("✅ Trainer is ready.")

trainer.train() 

#cell
import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load model with Loop 2 adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "phi2-lora-star-combined", is_local=True) # for now, change later
model.eval()
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
with open("test-dataset.jsonl", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

# Prompt format
def format_prompt(review):
    return f"Question: Classify this movie review into positive or negative: [{review}]\nJustify your decision by giving a well-formed explanation describing your rationale.\nAnswer:"

def extract_sentiment_and_rationale(output_text):
    sentiment_match = re.search(r"(POSITIVE|NEGATIVE)", output_text) # shouldn't be upper() lemme know when this is done
    sentiment = sentiment_match.group(0) if sentiment_match else "UNKNOWN" # changed it from  1 to 0. If this doesn't work, we will print out outputs. 
    rationale = output_text.split("Answer:")[-1].strip()
    return sentiment, rationale

# Evaluate
predictions, true_labels, rationales = [], [], []

for example in tqdm(test_data):
    prompt = format_prompt(example["text"])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sentiment, rationale = extract_sentiment_and_rationale(output_text)

    predictions.append(1 if sentiment == "POSITIVE" else 0)
    true_labels.append(example["label"])
    rationales.append(rationale)

# Metrics
acc = accuracy_score(true_labels, predictions)
prec, rec, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="macro")

print("\n✅ Loop 2 Evaluation Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Save predictions + rationales
with open("loop2_eval_with_rationales.jsonl", "w") as f:
    for example, label, pred, rationale in zip(test_data, true_labels, predictions, rationales):
        f.write(json.dumps({
            "text": example["text"],
            "true_label": label,
            "predicted": pred,
            "rationale": rationale
        }) + "\n")


#✅ Loop 2 Evaluation Results:
#Accuracy:  0.9467
#Precision: 0.9460
#Recall:    0.9473
#F1 Score:  0.9465