import torch, json, os, re
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training, get_peft_model, LoraConfig
)
from transformers.trainer_utils import set_seed

# Set seed
set_seed(42)

# Load Phi-2 with 4-bit quant
model_name = "microsoft/phi-2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# Apply LoRA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load and tokenize dataset
with open("fine_tune_data.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list([{"text": item["question"] + "\n" + item["answer"]} for item in data])
tokenized = dataset.map(
    lambda x: {**tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), "labels": tokenizer(x["text"], truncation=True, padding="max_length", max_length=512)["input_ids"]},
    remove_columns=["text"]
)

# Trainer setup
args = TrainingArguments(
    output_dir="./phi2-lora-finetuned",
    per_device_train_batch_size=6,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    save_strategy="epoch",
    save_total_limit=1,
    evaluation_strategy="no",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Train and save adapter
os.environ["TOKENIZERS_PARALLELISM"] = "false"
trainer.train()
model.save_pretrained("./phi2-lora-finetuned")


from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load model with LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, "./phi2-lora-finetuned")
model.eval()

# Load 150-example test set
with open("test-dataset.jsonl", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

# Prompt and extraction
def format_prompt(review):
    return f"Question: Classify this movie review into positive or negative: {review}\nAnswer: Sentiment:"

def extract_sentiment(text):
    match = re.search(r"Sentiment:\s*(Positive|Negative)", text, re.IGNORECASE)
    return match.group(1).capitalize() if match else "Unknown"

# Inference
predictions, true_labels = [], []

for ex in test_data:
    prompt = format_prompt(ex["text"])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False, temperature=0.7)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    sentiment = extract_sentiment(result)
    predictions.append(1 if sentiment == "Positive" else 0)
    true_labels.append(ex["label"])

# Metrics
acc = accuracy_score(true_labels, predictions)
prec, rec, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="macro")

print("✅ Evaluation Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")


#-----

#✅ Evaluation Results:
#Accuracy:  0.9467
#Precision: 0.9464
#Recall:    0.9464
#F1 Score:  0.9464