"""
transformer_model.py

Implementation of a decoder-only transformer for sequence modeling.
"""

import torch
import torch.nn as nn
from settings import Settings

class DecoderOnlyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(Settings.VOCABULARY_SIZE, Settings.MODEL_DIM)
        self.position_embedding = nn.Embedding(Settings.MAX_SEQUENCE_LENGTH, Settings.MODEL_DIM)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=Settings.MODEL_DIM,
            nhead=Settings.NUM_HEADS,
            dim_feedforward=Settings.FEEDFORWARD_DIM,
            dropout=Settings.DROPOUT_RATE,
            activation="gelu",
            norm_first=Settings.USE_NORM_FIRST
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=Settings.NUM_LAYERS)
        self.output_projection = nn.Linear(Settings.MODEL_DIM, Settings.VOCABULARY_SIZE, bias=False)
        self.output_projection.weight = self.token_embedding.weight

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        embedded = self.token_embedding(input_ids) + self.position_embedding(positions)
        embedded = embedded.transpose(0, 1)  # [seq_len, batch, model_dim]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        decoded = self.decoder(
            embedded, embedded,
            tgt_mask=mask,
            tgt_key_padding_mask=pad_mask
        )
        decoded = decoded.transpose(0, 1)
        return self.output_projection(decoded)