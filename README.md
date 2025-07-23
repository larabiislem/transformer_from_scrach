# Decoder-Only Transformer Implementation from Scratch

## Abstract

This repository presents a complete implementation of a decoder-only transformer architecture for conversational AI, trained on the Cornell Movie Dialogs Corpus. The implementation follows modern transformer design principles with pre-layer normalization, GELU activation, and byte-level BPE tokenization. The model is designed for sequence-to-sequence dialogue generation tasks.

## Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Training Details](#training-details)
- [Results](#results)
- [References](#references)
- [Citation](#citation)
- [License](#license)

## Introduction

The Transformer architecture, introduced by Vaswani et al. (2017), has revolutionized natural language processing through its self-attention mechanism. This implementation focuses on a decoder-only variant, similar to GPT models, which has proven highly effective for autoregressive language generation tasks.

Our implementation features:
- **Decoder-only architecture** with causal masking
- **Pre-layer normalization** for improved training stability
- **Byte-level BPE tokenization** for robust text processing
- **Configurable hyperparameters** for experimentation
- **Conversational dialogue training** on movie scripts

## Architecture

### Model Specifications

The transformer model consists of the following components:

```
Model Dimension (d_model): 512
Number of Layers: 6
Attention Heads: 8
Feed-Forward Dimension: 2048
Dropout Rate: 0.1
Maximum Sequence Length: 128
Vocabulary Size: 8000
```

### Technical Implementation

The model implements the standard transformer decoder with:

1. **Token and Positional Embeddings**: Convert input tokens to dense representations
2. **Multi-Head Self-Attention**: Compute attention weights across sequence positions
3. **Feed-Forward Networks**: Apply non-linear transformations
4. **Layer Normalization**: Stabilize training with pre-norm architecture
5. **Causal Masking**: Ensure autoregressive generation properties

### Mathematical Formulation

The attention mechanism follows:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where Q, K, V are query, key, and value matrices respectively.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/larabiislem/transformer_from_scrach.git
cd transformer_from_scrach
```

2. **Create virtual environment:**
```bash
python -m venv transformer_env
source transformer_env/bin/activate  # On Windows: transformer_env\Scripts\activate
```



## Usage

### Training the Model

To train the transformer model on the Cornell Movie Dialogs dataset:

```bash
python train_transformer.py
```

This will:
- Download and process the Cornell Movie Dialogs Corpus
- Build a BPE tokenizer from the training data
- Train the transformer for the specified number of epochs
- Save model checkpoints and tokenizer

### Configuration

Modify hyperparameters in `settings.py`:

```python
# Model Architecture
MODEL_DIM = 512
NUM_LAYERS = 6
NUM_HEADS = 8
FEEDFORWARD_DIM = 2048

# Training Parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 5
WARMUP_STEPS = 1000
```

### Inference

```python
from transformer_model import DecoderOnlyTransformer
from transformers import PreTrainedTokenizerFast
import torch

# Load trained model and tokenizer
model = DecoderOnlyTransformer()
model.load_state_dict(torch.load('transformer_checkpoint.pth'))
tokenizer = PreTrainedTokenizerFast.from_pretrained('./movie_tokenizer/')

# Generate response
def generate_response(prompt, max_length=50):
    tokens = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(tokens, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

## Dataset

### Cornell Movie Dialogs Corpus

The model is trained on the Cornell Movie Dialogs Corpus, which contains:
- **220,579 conversational exchanges** between characters
- **617 movies** spanning various genres
- **304,713 total utterances**
- Rich conversational context and diverse dialogue patterns

### Data Processing

The dataset preprocessing pipeline:
1. Extract conversation pairs from movie dialogues
2. Create input-response pairs from consecutive utterances
3. Apply BPE tokenization with special tokens
4. Pad sequences to fixed length with attention masking

## Configuration

### Hyperparameter Details

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MODEL_DIM` | 512 | Transformer hidden dimension |
| `NUM_LAYERS` | 6 | Number of decoder layers |
| `NUM_HEADS` | 8 | Multi-head attention heads |
| `FEEDFORWARD_DIM` | 2048 | Feed-forward network dimension |
| `DROPOUT_RATE` | 0.1 | Dropout probability |
| `MAX_SEQUENCE_LENGTH` | 128 | Maximum input sequence length |
| `VOCABULARY_SIZE` | 8000 | Tokenizer vocabulary size |
| `LEARNING_RATE` | 1e-4 | Adam optimizer learning rate |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization coefficient |
| `WARMUP_STEPS` | 1000 | Learning rate warmup steps |

### Special Tokens

```python
SPECIAL_TOKENS = ["<BOS>", "<EOS>", "<PAD>", "<UNK>", "<SEP>"]
```

## File Structure

```
transformer_from_scrach/
├── train_transformer.py      # Main training script
├── transformer_model.py      # Decoder-only transformer implementation
├── conversation_dataset.py   # Dataset loading and processing
├── tokenizer_utils.py        # BPE tokenizer utilities
├── settings.py              # Configuration parameters
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Core Components

- **`train_transformer.py`**: Orchestrates the complete training pipeline including data loading, model initialization, and training loop
- **`transformer_model.py`**: Implements the DecoderOnlyTransformer class with PyTorch nn.Module
- **`conversation_dataset.py`**: Handles Cornell Movie Dialogs dataset loading and preprocessing
- **`tokenizer_utils.py`**: Provides utilities for building and configuring BPE tokenizers
- **`settings.py`**: Centralizes all hyperparameters and configuration options

## Training Details

### Training Procedure

1. **Data Preparation**: Load Cornell Movie Dialogs and create conversation pairs
2. **Tokenizer Training**: Build BPE tokenizer on training corpus
3. **Model Initialization**: Initialize transformer with specified architecture
4. **Training Loop**: 
   - Forward pass with teacher forcing
   - Cross-entropy loss computation
   - Gradient clipping and optimization
   - Learning rate scheduling with warmup

### Optimization Strategy

- **Optimizer**: AdamW with weight decay
- **Learning Rate Schedule**: Linear warmup followed by linear decay
- **Gradient Clipping**: Max norm of 1.0
- **Loss Function**: Cross-entropy with padding token masking

### Training Metrics

The training script reports:
- Training loss per epoch
- Validation loss and perplexity
- Learning rate progression
- Training time statistics

## Results

### Performance Metrics

Training typically achieves:
- **Training Loss**: Convergence to ~2.5-3.0
- **Validation Perplexity**: ~15-25 after 5 epochs
- **Training Time**: ~30-60 minutes on modern GPU

### Qualitative Assessment

The model demonstrates:
- Coherent short-term dialogue responses
- Basic conversational flow understanding
- Appropriate use of special tokens
- Reasonable grammatical structure

## References

1. Vaswani, A., et al. (2017). "Attention is All You Need." *Advances in Neural Information Processing Systems*.

2. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI Blog*.

3. Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation of Rare Words with Subword Units." *ACL*.

4. Danescu-Niculescu-Mizil, C., & Lee, L. (2011). "Chameleons in Imagined Conversations: A New Approach to Understanding Coordination of Linguistic Style in Dialogs." *Workshop on Cognitive Modeling and Computational Linguistics, ACL*.


```

