"""
infer_transformer.py

Script for running inference with the trained transformer model.
"""

import torch
from transformers import PreTrainedTokenizerFast
from settings import Settings
from transformer_model import DecoderOnlyTransformer

def load_model_and_tokenizer():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(Settings.TOKENIZER_PATH)
    model = DecoderOnlyTransformer().to(Settings.DEVICE)
    model.load_state_dict(torch.load(Settings.CHECKPOINT_PATH, map_location=Settings.DEVICE))
    model.eval()
    return model, tokenizer

@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_length=50):
    ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(Settings.DEVICE)
    bos = torch.tensor([[tokenizer.bos_token_id]], device=Settings.DEVICE)
    input_ids = torch.cat([bos, ids], dim=1)
    for _ in range(max_length):
        mask = (input_ids != tokenizer.pad_token_id).long()
        logits = model(input_ids, attention_mask=mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    while True:
        prompt = input("\nUser: ")
        print("Bot:", generate_response(model, tokenizer, prompt))