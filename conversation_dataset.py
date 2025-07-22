import torch
from torch.utils.data import Dataset
from .conversation_dataset import load_dataset
from .config import Settings

class MovieDialogsDataset(Dataset):
    """
    Loads and processes the Cornell Movie Dialogs Corpus as input/response pairs.
    """

    def __init__(self, tokenizer, split="train"):
        self.tokenizer = tokenizer
        self.max_length = Settings.MAX_SEQUENCE_LENGTH
        ds = load_dataset("cornell_movie_dialogs", split=split)
        self.pairs = []
        for dialog in ds["conversations"]:
            utterances = dialog["utterances"]
            for i in range(len(utterances) - 1):
                prompt = utterances[i]["text"]
                response = utterances[i + 1]["text"]
                self.pairs.append((prompt, response))
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prompt, response = self.pairs[idx]
        tokens = [self.tokenizer.bos_token_id]
        tokens += self.tokenizer.encode(prompt, add_special_tokens=False)
        tokens += [self.tokenizer.sep_token_id]
        tokens += self.tokenizer.encode(response, add_special_tokens=False)
        tokens += [self.tokenizer.eos_token_id]

        tokens = tokens[:self.max_length]
        attention = [1] * len(tokens)

        pad_amount = self.max_length - len(tokens)
        if pad_amount > 0:
            tokens += [self.tokenizer.pad_token_id] * pad_amount
            attention += [0] * pad_amount

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long)
        }