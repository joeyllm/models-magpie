# Models Magpie – Fine-Tuning

This repo is for **fine-tuning Mistral 7B** on cleaned data prepared in **data-magpie**.

---

## 🎯 Focus for This Sprint

- Build a **training pipeline** that runs on the **2× A100 environment**.  
- Connect the training run to **Weights & Biases (W&B)** for tracking.  
- Use the **cleaned + tokenized dataset** output from **data-magpie**.  

---

## 📦 Deliverable

A training script that:  
1. Loads the dataset from **data-magpie**.  
2. Sets up the **Mistral 7B model and tokenizer**.  
3. Runs training on the **2× A100 environment**.  
4. Logs metrics to **W&B**.  

---

## 🖥 Environment

- **Development environment:** VS Code server link  
- **Point of contact:** Pranav (he has the password)  

---

## 🔜 Coming Next Sprint

- Scale from a **single parquet file** to the **full FineWeb dataset**.  
- Experiment with **hyperparameters** and **checkpointing**.  
- Start preparing for **longer training runs**.  

---

## 📝 Notes

- Focus is on getting a **working training loop**, not tuning for performance yet.  
- Keep code **clean and testable**.  
- Use **W&B** so results are visible.  



# Magpie
## Mistral Fine-Tuning for Australian Context

This repository contains code, configurations, and documentation for fine-tuning the Mistral language model on Australian-specific data. The goal is to adapt the base model to better understand and generate content relevant to the Australian context — including language, culture, institutions, and regional knowledge.

## Objectives

- Fine-tune the Mistral model on high-quality Australian datasets
- Improve performance on tasks with localised terminology, cultural references, and spelling conventions (e.g., "organise" vs. "organize")
- Evaluate downstream improvements in generation and comprehension on Australian-specific tasks

