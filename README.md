# Self-Taught Reasoning Meets Structured Markup  
*A Hybrid Approach for Movie Review Sentiment Analysis*

This project explores interpretable sentiment classification using open-source language models. We combine the STaR (Self-Taught Reasoning) framework with structured rationale generation to improve classification and transparency on long movie reviews.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `exp1_prompt_only.py` | Zero-shot prompting with Phi-2 (no training) |
| `exp2_lora_qa.py` | LoRA fine-tuning on SST2 QA-style classification pairs |
| `exp3_lora_rationale.py` | LoRA fine-tuning on structured rationale-rich examples |
| `exp4_star_loop.py` | STaR-style rationale generation + retraining cycle |
| `dataset_maker.py` | Initial dataset generator using SST2 tree structure |
| `dataset_maker_improved.py` | Improved version with phrase numbering and better logic |
| `test-dataset.jsonl` | 150 long IMDb reviews used for evaluation |
| `fine_tune_dataset_improved.jsonl` | Final training data used in Exp3 and Exp4 |

---

## 🧠 Key Concepts

- **STaR Framework**: Bootstraps model reasoning by generating rationales, then fine-tuning on them.
- **Structured Rationales**: Each review is broken into labeled phrases with polarity and logic-based justification.
- **QLoRA**: Efficient 4-bit fine-tuning on Phi-2, using LoRA adapters for memory efficiency.

---

## 🧪 Experiments Summary

| Exp | Description | Accuracy |
|-----|-------------|----------|
| 1 | Prompt-only, no fine-tuning | 94.67% |
| 2 | QA fine-tuning (no rationale) | 94.67% |
| 3 | QA + Structured rationale | **96.67%** |
| 4 | STaR rationale generation + retraining | 94.67% |

---

## 📊 Evaluation

Models were evaluated on IMDb reviews (7× longer than SST2), using:
- Accuracy, Precision, Recall, F1
- Manual inspection of rationales for logical consistency and clarity

---

## 🛠 Requirements

- Python 3.10+
- PyTorch (with CUDA support)
- Transformers (Hugging Face)
- PEFT + BitsAndBytes (for QLoRA)
- `datasets`, `scikit-learn`, `tqdm`

---

## 📌 Authors

- Md Afridi Hasan  
- Aditya Kumar  
University of Arizona — CSC 696H

---

## 📄 License

This repository is for academic course use only.
