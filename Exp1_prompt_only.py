import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# Load your 150-example IMDb test set
test_data = []
with open("test-dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        test_data.append(json.loads(line))

# Simple classification prompt
def format_prompt(review):
    return f"Question: Classify this movie review into positive or negative: {review}\nAnswer: Sentiment:"

# Extract sentiment from output
def extract_sentiment(generated_text):
    match = re.search(r"Sentiment:\s*(Positive|Negative)", generated_text, re.IGNORECASE)
    return match.group(1).capitalize() if match else "Unknown"

# Inference
predictions = []
true_labels = []

for example in test_data:
    prompt = format_prompt(example["text"])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.7
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sentiment = extract_sentiment(output_text)

    predictions.append(1 if sentiment == "Positive" else 0)
    true_labels.append(example["label"])

# Evaluation
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="macro")

print("✅ Evaluation Results:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


----

#✅ Evaluation Results:
#ccuracy:  0.9467
#Precision: 0.9463
#Recall:    0.9482
#F1 Score:  0.9466