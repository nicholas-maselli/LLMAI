# Training outputs

Training artifacts are organized by model name and timestamped run:

```text
train/
└── tiny_language_model/
    ├── checkpoints/
    │   └── 2026-07-19_12-30-45/
    │       ├── step_000200.pt
    │       └── step_000400.pt
    └── logs/
        └── 2026-07-19_12-30-45/
            └── training_metrics.csv
```

Each checkpoint contains model weights, configuration, tokenizer vocabulary, and its training step. The metrics file records loss, gradient norm, learning rate, and elapsed time.
