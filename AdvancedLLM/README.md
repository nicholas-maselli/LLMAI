# Advanced decoder-only language model

This project builds a modern decoder-only transformer strictly for base-model pretraining. It intentionally excludes instruction tuning, preference optimization, agents, tool use, retrieval, and serving.

## Architecture target

The model will use:

- A learned subword tokenizer
- Token embeddings with tied output weights
- Rotary position embeddings (RoPE)
- Grouped-query causal self-attention (GQA)
- RMSNorm with pre-normalization
- SwiGLU feed-forward networks
- Residual connections
- Flash/scaled-dot-product attention when supported by PyTorch
- Mixed-precision distributed pretraining
- Gradient accumulation, clipping, warmup, cosine decay, and checkpoint resume

## Pretraining data

Development will use streamed FineWeb data so the entire corpus does not need to fit on disk. FineWeb is useful for general English, but a coding-focused base model will eventually need a carefully licensed code-data component and an explicit mixture policy.

## Build order

We will implement only one or two concepts at a time:

1. Configuration
2. Dataset streaming and document boundaries
3. Tokenizer training and encoding
4. Packed token batches
5. RMSNorm
6. Rotary embeddings
7. Grouped-query causal attention
8. SwiGLU
9. Transformer block
10. Full decoder model and tied language-model head
11. Initialization and parameter accounting
12. Pretraining loss and optimizer
13. Learning-rate schedule and gradient accumulation
14. Mixed precision and distributed execution
15. Logging, validation loss, and resumable checkpoints

## Current step

Only the model and data configuration classes exist. No architecture or training behavior has been implemented yet.

## Structure

```text
AdvancedLLM/
├── configs/                  # Saved experiment configurations
├── outputs/                  # Checkpoints and pretraining metrics
├── scripts/                  # Future command-line entry points
├── src/advanced_llm/
│   ├── config.py             # Current substantive implementation
│   ├── data/                 # Streaming, tokenization, packing, collators
│   ├── model/                # Transformer architecture components
│   ├── training/             # Pretraining and resumable training state
│   └── utils/                # Shared infrastructure
└── tests/
```

All named pretraining modules now exist as purpose-only placeholders. We will replace those markers with implementation one or two concepts at a time. No SFT, preference training, RL, agent runtime, serving, or post-training files are included.
