import os
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler

from tqdm import tqdm
import torch.distributed as dist


# ======================================================
# Configuration
# ======================================================
model_name = "Helsinki-NLP/opus-mt-en-ur"      # Pretrained English → Urdu
dataset_path = "ERUPD_NMT.csv"                 # CSV: English, RomanUrdu
output_dir = "checkpoints"
batch_size = 8
num_epochs = 20
lr = 5e-5
max_len = 128

os.makedirs(output_dir, exist_ok=True)


# ======================================================
# DDP Setup
# ======================================================
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


# ======================================================
# Dataset
# ======================================================
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

        src = self.tokenizer(
            src_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        tgt = self.tokenizer(
            tgt_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = src["input_ids"].squeeze()
        attention_mask = src["attention_mask"].squeeze()
        labels = tgt["input_ids"].squeeze()

        # Mask pad tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ======================================================
# Load tokenizer & model
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(device)

# DDP - faster without "find_unused_parameters"
# model = DDP(model, device_ids=[local_rank], output_device=local_rank)


# ======================================================
# Dataset and Loaders
# ======================================================
dataset = TranslationDataset(dataset_path, tokenizer, max_len)
train_size = int(0.9 * len(dataset))   # larger train split
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset, shuffle=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    sampler=val_sampler
)


# ======================================================
# Optimizer & Scheduler
# ======================================================
optimizer = AdamW(model.parameters(), lr=lr)
num_training_steps = num_epochs * len(train_loader)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# scaler = GradScaler()
scaler = GradScaler(device="cuda")



# ======================================================
# TRAINING LOOP
# ======================================================
def validate():
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type="cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()

    return total_loss / len(val_loader)


for epoch in range(1, num_epochs + 1):
    model.train()
    train_sampler.set_epoch(epoch)
    running_loss = 0

    progress = tqdm(
        train_loader,
        desc=f"EPOCH {epoch} (Rank {local_rank})",
        disable=(local_rank != 0)
    )

    for batch in progress:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(device_type="cuda"):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        if local_rank == 0:
            progress.set_postfix({"loss": loss.item()})

    # ------------------------------
    # Validation (Rank 0 only)
    # ------------------------------
    if dist.get_rank() == 0:
        val_loss = validate()
        print(f"[Epoch {epoch}] Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved → {checkpoint_dir}")


# ======================================================
# SAVE FINAL MODEL
# ======================================================
if dist.get_rank() == 0:
    final_dir = "final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with zipfile.ZipFile("final_model.zip", "w") as zipf:
        for root, dirs, files in os.walk(final_dir):
            for file in files:
                zipf.write(
                    os.path.join(root, file),
                    arcname=os.path.relpath(os.path.join(root, file), final_dir)
                )

    print("Training complete. Final model saved as final_model.zip")

dist.destroy_process_group()
