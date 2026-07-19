# Pretraining outputs

Generated artifacts will use this structure:

```text
pretrain/
└── advanced_small/
    └── 2026-07-19_18-30-00/
        ├── checkpoints/
        │   └── step_001000/
        │       ├── model.pt
        │       ├── optimizer.pt
        │       ├── scheduler.pt
        │       └── trainer_state.json
        ├── logs/
        │   ├── metrics.jsonl
        │   └── training.log
        ├── tokenizer/
        └── resolved_config.yaml
```

The folders below `outputs/pretrain/` will be generated during training and ignored by Git.
