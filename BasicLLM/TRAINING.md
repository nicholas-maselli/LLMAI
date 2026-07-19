# Training and generating text

For a more detailed inference guide, see [GENERATION.md](GENERATION.md).

Run every command below from the `BasicLLM` directory.

## Train with the defaults

```bash
uv run src/train.py
```

The defaults come from `src/config.py`. The default model uses a context length of 32, embedding size of 64, four attention heads, a batch size of 16, and 500 training steps.

## Train with command-line parameters

```bash
uv run src/train.py \
  --model-name code_model \
  --context-length 64 \
  --embedding-size 128 \
  --heads 4 \
  --batch-size 32 \
  --steps 5000 \
  --learning-rate 0.001 \
  --log-every 200 \
  --checkpoint-every 1000
```

Git Bash uses the backslash above to continue a command on the next line. Do not place spaces after the backslash. The same command can be written on one line:

```bash
uv run src/train.py --model-name code_model --context-length 64 --embedding-size 128 --heads 4 --batch-size 32 --steps 5000 --learning-rate 0.001 --log-every 200 --checkpoint-every 1000
```

### Training parameters

| Parameter | Meaning |
| --- | --- |
| `--model-name` | Directory name used for this model's outputs. |
| `--context-length` | Maximum number of tokens the model can examine at once. |
| `--embedding-size` | Length of each token vector. Must be divisible by the number of heads. |
| `--heads` | Number of parallel attention heads. |
| `--batch-size` | Number of training examples processed per step. |
| `--steps` | Number of optimizer updates. |
| `--learning-rate` | Size of each optimizer update. |
| `--log-every` | Number of steps between metric records. |
| `--checkpoint-every` | Number of steps between saved checkpoints. |

View the built-in command help with:

```bash
uv run src/train.py --help
```

## Training outputs

A run produces this structure:

```text
outputs/train/code_model/
├── checkpoints/
│   └── 2026-07-19_12-30-45/
│       ├── step_001000.pt
│       ├── step_002000.pt
│       └── step_005000.pt
└── logs/
    └── 2026-07-19_12-30-45/
        └── training_metrics.csv
```

The metrics CSV records the step, loss, global gradient norm, learning rate, and elapsed training time. A final checkpoint is always saved, even when the final step does not match the checkpoint interval.

## Generate from the newest checkpoint

```bash
uv run src/generate.py
```

By default, this finds the most recently modified checkpoint and generates 200 characters beginning with `def `.

Provide a prompt and output length with:

```bash
uv run src/generate.py --prompt "def add" --tokens 300
```

Load a particular checkpoint with:

```bash
uv run src/generate.py \
  --checkpoint outputs/train/code_model/checkpoints/2026-07-19_12-30-45/step_005000.pt \
  --prompt "def multiply" \
  --tokens 300
```

View generation options with:

```bash
uv run src/generate.py --help
```

The prompt can contain only characters found in the training dataset because this project uses a character-level tokenizer.
