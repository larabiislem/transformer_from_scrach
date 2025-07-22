"""
settings.py

Configuration module for the transformer training and inference.
"""

class Settings:
    MODEL_DIM = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    FEEDFORWARD_DIM = 2048
    DROPOUT_RATE = 0.1
    USE_NORM_FIRST = True

    MAX_SEQUENCE_LENGTH = 128
    BATCH_SIZE = 64
    EPOCHS = 5
    VOCABULARY_SIZE = 8000
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_STEPS = 1000

    SPECIAL_TOKENS = ["<BOS>", "<EOS>", "<PAD>", "<UNK>", "<SEP>"]

    DEVICE = "cuda"
    CHECKPOINT_PATH = "transformer_checkpoint.pth"
    TOKENIZER_PATH = "./movie_tokenizer/"
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"