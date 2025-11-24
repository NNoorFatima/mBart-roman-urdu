import os
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torch.distributed as dist

# -------------------------------
# Configuration
# -------------------------------
model_name = "Helsinki-NLP/opus-mt-en-ur"  # English → Urdu
dataset_path = "ERUPD_NMT.csv"  # CSV dataset: English,Roman Urdu
output_dir = "checkpoints"
batch_size = 4
num_epochs = 20
lr = 5e-5
max_len = 128

os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# DDP Setup
# -------------------------------
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# -------------------------------
# Dataset class
# -------------------------------
class TranslationDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        import pandas as pd
        self.data = pd.read_csv(path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = str(self.data.iloc[idx, 0])
        tgt_text = str(self.data.iloc[idx, 1])
        src = self.tokenizer(src_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
        tgt = self.tokenizer(tgt_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
        input_ids = src["input_ids"].squeeze()
        attention_mask = src["attention_mask"].squeeze()
        labels = tgt["input_ids"].squeeze()
        labels[labels == tokenizer.pad_token_id] = -100  # ignore padding
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# -------------------------------
# Load tokenizer and model
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

# -------------------------------
# Dataset and DataLoaders
# -------------------------------
dataset = TranslationDataset(dataset_path, tokenizer, max_len)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

# -------------------------------
# Optimizer & Scheduler
# -------------------------------
optimizer = AdamW(model.parameters(), lr=lr)
num_training_steps = num_epochs * len(train_loader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# -------------------------------
# Training loop
# -------------------------------
for epoch in range(1, num_epochs + 1):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = 0

    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training (Rank {local_rank})")):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

        # Evaluate every iteration (on rank 0 only)
        if dist.get_rank() == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_input_ids = val_batch["input_ids"].to(device)
                    val_attention_mask = val_batch["attention_mask"].to(device)
                    val_labels = val_batch["labels"].to(device)
                    val_outputs = model(val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
                    val_loss += val_outputs.loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch} Iter {i+1} — Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f}")
            model.train()

    # Save checkpoint per epoch (rank 0 only)
    if dist.get_rank() == 0:
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
        model.module.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)

# -------------------------------
# Save final model and zip (rank 0 only)
# -------------------------------
if dist.get_rank() == 0:
    final_model_dir = "final_model"
    model.module.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    with zipfile.ZipFile("final_model.zip", 'w') as zipf:
        for root, dirs, files in os.walk(final_model_dir):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=os.path.join(os.path.relpath(root, final_model_dir), file))

    print("Training complete. Final model saved as final_model.zip")

# -------------------------------
# Cleanup
# -------------------------------
dist.destroy_process_group()
