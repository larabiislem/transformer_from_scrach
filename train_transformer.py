"""
train_transformer.py

Script for training the transformer model on Cornell Movie Dialogs Corpus.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
from settings import Settings
from tokenizer_utils import build_tokenizer_from_texts
from conversation_dataset import MovieDialogsDataset
from transformer_model import DecoderOnlyTransformer
from conversation_dataset import load_dataset

def lr_scheduler(optimizer, warmup_steps, total_steps):
    def schedule(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))
    return LambdaLR(optimizer, schedule)

def compute_perplexity(loss):
    return torch.exp(loss)

def train():
    # Build tokenizer
    raw_data = load_dataset("cornell_movie_dialog", split="train", download_mode="force_redownload")
    all_utterances = []
    for dialog in raw_data["conversations"]:
        for utter in dialog["utterances"]:
            all_utterances.append(utter["text"])
    tokenizer = build_tokenizer_from_texts(all_utterances)
    tokenizer.save_pretrained(Settings.TOKENIZER_PATH)

    # Prepare dataset
    full_dataset = MovieDialogsDataset(tokenizer, split="train")
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=Settings.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=Settings.BATCH_SIZE)

    # Model, optimizer, loss, scheduler
    model = DecoderOnlyTransformer().to(Settings.DEVICE)
    optimizer = AdamW(model.parameters(), lr=Settings.LEARNING_RATE, weight_decay=Settings.WEIGHT_DECAY)
    total_steps = Settings.EPOCHS * len(train_loader)
    scheduler = lr_scheduler(optimizer, Settings.WARMUP_STEPS, total_steps)
    criterion = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    best_val_loss = float("inf")

    for epoch in range(1, Settings.EPOCHS + 1):
        model.train()
        running_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            input_ids = batch["input_ids"].to(Settings.DEVICE)
            attention = batch["attention_mask"].to(Settings.DEVICE)
            logits = model(input_ids, attention)
            loss = criterion(
                logits[:, :-1].reshape(-1, Settings.VOCABULARY_SIZE),
                input_ids[:, 1:].reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_train_loss += loss.item()
        avg_train_loss = running_train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(Settings.DEVICE)
                attention = batch["attention_mask"].to(Settings.DEVICE)
                logits = model(input_ids, attention)
                loss = criterion(
                    logits[:, :-1].reshape(-1, Settings.VOCABULARY_SIZE),
                    input_ids[:, 1:].reshape(-1)
                )
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch} | train loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}, val ppl: {compute_perplexity(torch.tensor(avg_val_loss)):.2f}")

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Settings.CHECKPOINT_PATH)
            print(f"Saved best model (val_loss={avg_val_loss:.4f})\n")

if __name__ == "__main__":
    train()