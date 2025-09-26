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

## Notes 26/09/25

building off of main_fsdp.py 

when running torchrun --nproc_per_node=3 --nnodes=1 --node_rank=0 main_fsdp.py

get the error
requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/mistralai/Mistral-7B-v0.3/resolve/main/config.json

either lacking authentication or dont have model access on huggingface

running 
TRANSFORMERS_OFFLINE=1 torchrun --nproc_per_node=3 --nnodes=1 --node_rank=0 main_fsdp.py
does not resolve the issue.

the model seems to be located in .hf_cache/hub/models--mistralai--Mistral-7B-v0.3 but am unsure how to configure the project to use it.

export HUGGINGFACE_HUB_TOKEN=hf_XXXXXX

get warning \`torch_dtype\` is depricated! Use \`dtype\` instead!

line 149 model = FSDP(...
ValueError: Cannot flatten integer dtype tensors

export HYDRA_FULL_ERROR=1 // for more detailed logs

need to modify auto_wrap_policy to ignore the KV caches because the FSDP wrapper seems to be interpreting those values as trainable weights. leading to error with trying to flatten dtype tensors. 

after fixing that there seems to be an issue with permissions 

Cannot access gated repo for url https://huggingface.co/mistralai/Mistral-7B-v0.3/resolve/main/config.json.
Access to model mistralai/Mistral-7B-v0.3 is restricted. You must have access to it and be authenticated to access it. Please log in. - silently ignoring the lookup for the file config.json in mistralai/Mistral-7B-v0.3.