# Generated outputs

Training artifacts will be written beneath this directory, grouped by stage:

```text
outputs/
|-- pretrain/<run-name>/
`-- sft/<run-name>/
```

Each pretraining run now contains:

```text
outputs/pretrain/<run-name>/
|-- config.json
|-- metrics.jsonl
`-- checkpoints/
    `-- step_00000500.pt
```

Each checkpoint contains model weights, optimizer state, mixed-precision
scaler state, completed step, token count, configuration, and random-number
generator state. Generated run directories are ignored by Git.
