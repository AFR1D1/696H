#cell 1

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training, get_peft_model, LoraConfig
)
from transformers.trainer_utils import set_seed

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
from datasets import Dataset

# Load your fine_tune_data improved one
data_path = "fine_tune_dataset_improved.jsonl"

with open(data_path, "r") as f:
    raw_data = [json.loads(line) for line in f]

# Convert to HuggingFace Dataset
dataset = Dataset.from_list([
    {"text": f'Question: {item["question"]}\nAnswer: {item["answer"]}'} for item in raw_data
])


print(f"✅ Loaded {len(dataset)} examples.")
dataset[0]  # Peek at one example


#cell 3
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
#cell 4
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./phi2-lora-rationale-improved",
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

#cell 5:
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
trainer.train()
model.save_pretrained("./phi2-lora-rationale-improved")


#cell 6:
import json
import re
import torch
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Paths
adapter_path = "./phi2-lora-rationale-improved"
base_model = "microsoft/phi-2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load base model and LoRA adapter
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base, adapter_path)

# Move to GPU if available
if torch.cuda.is_available():
    model = model.to("cuda")
model.eval()

# Load test data
with open("test-dataset.jsonl", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

# Prompt format
def format_prompt(review):
    return f"Question: Classify this movie review into positive or negative: [{review}]\nJustify your decision by giving a well-formed explanation describing your rationale.\nAnswer:"

# Extract prediction and rationale
def extract_sentiment_and_rationale(output_text):
    sentiment_match = re.search(r"POSITIVE|NEGATIVE", output_text)
    sentiment = sentiment_match.group(0) if sentiment_match else "Unknown"
    rationale_match = re.search(r"(rationale[:\-]*)(.*)", output_text, re.IGNORECASE | re.DOTALL)
    rationale = rationale_match.group(2).strip() if rationale_match else "[Not found]"
    return sentiment, rationale

# Run inference with timing
start_time = time.time()

predictions, true_labels, rationales = [], [], []

for example in tqdm(test_data, desc="🔍 Running Inference"):
    prompt = format_prompt(example["text"])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sentiment, rationale = extract_sentiment_and_rationale(output_text)
    print("SENTIMENT: " + sentiment + "\nAnswer: "+ rationale + "\n")
    predictions.append(1 if sentiment == "POSITIVE" else 0)
    true_labels.append(example["label"])
    rationales.append(rationale)

elapsed = time.time() - start_time

# Metrics
acc = accuracy_score(true_labels, predictions)
prec, rec, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="macro")

print("✅ Evaluation Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"\n⏱️ Total inference time: {elapsed:.2f} seconds")
print(f"⏱️ Avg per example: {elapsed / len(test_data):.2f} seconds")

# Save predictions + rationales
with open("imdb_exp3_predictions_with_rationales.jsonl", "w") as f:
    for example, label, pred, rationale in zip(test_data, true_labels, predictions, rationales):
        f.write(json.dumps({
            "text": example["text"],
            "true_label": label,
            "predicted": pred,
            "rationale": rationale
        }) + "\n")



---

#Accuracy:  0.9667
#Precision: 0.9684
#Recall:    0.9652
#F1 Score:  0.9664

#⏱️ Total inference time: 1090.46 seconds
#⏱️ Avg per example: 7.27 seconds