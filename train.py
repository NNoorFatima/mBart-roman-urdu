# train_mbart.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ACCELERATE_BACKEND"] = "gloo"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" 

import random
import torch
import pandas as pd
import evaluate
import torch
from datasets import Dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
import shutil
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# --- GPU Check ---
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"GPUs detected: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs detected. Training will use CPU.")

# Example: force model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ------------------------------
# Load CSV and prepare dataset
# ------------------------------
df = pd.read_csv("ERUPD_NMT.csv")  # update path if needed
df = df[["English", "Roman Urdu"]]
dataset = Dataset.from_pandas(df)
print("First 5 rows:\n", dataset[:5])

# ------------------------------
# Load Model & Tokenizer
# ------------------------------
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Add new language token for Roman Urdu
new_lang_token = "<roman_ur>"
tokenizer.add_tokens([new_lang_token])
tokenizer.tgt_lang = new_lang_token
model.resize_token_embeddings(len(tokenizer))

# ------------------------------
# Preprocessing function
# ------------------------------
def preprocess(batch):
    sources = ["<en_XX> " + str(t) for t in batch["English"]]
    targets = [f"{new_lang_token} " + str(t) for t in batch["Roman Urdu"]]

    model_inputs = tokenizer(
        sources,
        max_length=128,
        truncation=True,
        padding="max_length",
        text_target=targets,
        return_tensors=None
    )
    return model_inputs
    # return {k: v.tolist() for k, v in model_inputs.items()}

tokenized_dataset = dataset.map(preprocess, batched=True)

# ------------------------------
# Split train and eval
# ------------------------------
eval_dataset = tokenized_dataset.select(range(1000))
train_dataset = tokenized_dataset.select(range(1000, len(tokenized_dataset)))

# ------------------------------
# Metrics
# ------------------------------
bleu = evaluate.load("sacrebleu")

def compute_metrics(pred):
    preds = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
    labels = [[l] for l in labels]   # sacrebleu format
    result = bleu.compute(predictions=preds, references=labels)
    return {"bleu": result["score"]}

# ------------------------------
# Progress Callback
# ------------------------------
class TranslationProgressCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n Epoch {int(state.epoch)} completed")
        print(f"➡ Training Loss: {state.log_history[-1]['loss'] if 'loss' in state.log_history[-1] else 'N/A'}")

        if int(state.epoch) % 5 == 0:
            example_text = "For how much is this coat? I want to buy it for my brother."
            inputs = tokenizer("<en_XX> " + example_text, return_tensors="pt").to(model.device)
            output_tokens = model.generate(**inputs, max_new_tokens=60)
            translation = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            print(f" Test English Input: {example_text}")
            print(f" Model Output (Roman Urdu): {translation}")
            print("----------------------------------------------------------")

# ------------------------------
# Data Collator
# ------------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ------------------------------
# Training Arguments
# ------------------------------
use_cuda = torch.cuda.is_available()

# training_args = TrainingArguments(
#     output_dir="./mbart-roman-ur-finetune",
#     per_device_train_batch_size=2,          # batch per GPU
#     per_device_eval_batch_size=2,
#     gradient_accumulation_steps=8,          # accumulates gradients to increase effective batch size
#     bf16=False,                             # use bf16 if supported
#     fp16=True,                               # enable mixed precision
#     gradient_checkpointing=True,            # save memory
#     num_train_epochs=10,
#     learning_rate=5e-5,
#     lr_scheduler_type="cosine",
#     warmup_steps=500,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     save_total_limit=2,
#     logging_strategy="steps",
#     logging_steps=100,
#     report_to="none",
#     load_best_model_at_end=True,
#     metric_for_best_model="loss",
#     greater_is_better=False,
#     seed=42,
#     dataloader_num_workers=8,
#     ddp_find_unused_parameters=False,      # required for DDP multi-GPU                         # will be set automatically by torchrun
# )
training_args = TrainingArguments(
    output_dir="./mbart-roman-ur-finetune",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    fp16=True,
    bf16=False,                     # H100 supports BF16, better performance
    gradient_checkpointing=True,
    torch_compile=True,
    num_train_epochs=10,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=50,
    report_to="none",
    seed=42,
    ddp_backend="nccl",
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
    remove_unused_columns=False,

)

# ------------------------------
# Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[TranslationProgressCallback]
)

# ------------------------------
# Start Training
# ------------------------------
trainer.train()

# ------------------------------
# Save final model + tokenizer
# ------------------------------
output_dir = "./mbart-roman-ur-final"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model & tokenizer saved to {output_dir}")

# ------------------------------
# Zip the saved model
# ------------------------------
shutil.make_archive(output_dir, 'zip', output_dir)
print(f"Model & tokenizer zipped as {output_dir}.zip")
