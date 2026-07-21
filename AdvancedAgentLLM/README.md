# Advanced agent language model

This project continues the learning path from `AdvancedLLM`: first build a
modern decoder-only language model, then teach it to follow instructions with
supervised fine-tuning (SFT), and finally place that model inside an agent
runtime that can call tools.

The project began by matching the reusable block from `AdvancedLLM`:

- model and data configuration
- native PyTorch RMSNorm
- rotary position embeddings (RoPE)
- grouped-query causal self-attention (GQA)
- SwiGLU feed-forward network
- pre-normalized transformer block with residual connections

It now additionally includes:

- stacked decoder-only language model with tied token/output embeddings
- streamed or local-document token packing
- next-token cross-entropy loss
- AdamW, warmup plus cosine decay, accumulation, and gradient clipping
- mixed-precision training and validation
- metrics and resumable checkpoints

The complete decoder model and resumable pretraining pipeline are now
implemented. The SFT pipeline and agent runtime remain deliberate placeholders
for the next learning stages.

## Learning path

1. Assemble the full decoder-only language model. (complete)
2. Add the language-model loss and finish pretraining infrastructure. (complete)
3. Define a conversation format and special tokens. (next)
4. Convert conversations into SFT examples.
5. Mask prompt tokens so loss is learned from assistant responses.
6. Fine-tune and evaluate instruction following.
7. Define structured tool schemas and tool-call messages.
8. Build the agent loop: model -> tool call -> tool result -> model.

SFT teaches behavior; the agent runtime executes that behavior. Tool execution
does not happen inside the transformer itself.

## Structure

```text
AdvancedAgentLLM/
|-- configs/                 # Future experiment configurations
|-- outputs/                 # Generated checkpoints and metrics
|-- scripts/                 # Future command-line entry points
|-- src/advanced_agent_llm/
|   |-- config.py            # Model, data, and training configuration
|   |-- model/               # Complete decoder-only language model
|   |-- data/                # Pretraining and future SFT data pipeline
|   |-- training/            # Pretraining and SFT training infrastructure
|   `-- agent/               # Future tool schemas and runtime loop
`-- tests/                   # Component tests
```

## Current next step

Pretraining is runnable. The next lesson is the conceptual boundary between
pretraining examples and conversations: a message schema and chat template.
