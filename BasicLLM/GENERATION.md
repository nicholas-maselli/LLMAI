# Generating text

Run every command below from the `BasicLLM` directory.

Train the model before generating text. See [TRAINING.md](TRAINING.md) for training commands.

## Generate from the newest checkpoint

```bash
uv run src/generate.py
```

This searches `outputs/train/` for the most recently modified checkpoint. By default, generation:

- Starts with the prompt `def `.
- Generates the number of characters configured by `generation_length`.
- Samples each character from the model's predicted probability distribution.

## Choose a prompt

```bash
uv run src/generate.py --prompt "def add"
```

Because this is a character-level model, every character in the prompt must exist in the checkpoint's vocabulary. That vocabulary was created from the training dataset.

## Choose the generation length

```bash
uv run src/generate.py --prompt "def add" --tokens 300
```

`--tokens 300` generates 300 new character tokens after the prompt. For this model, one token is one character.

## Choose a checkpoint

```bash
uv run src/generate.py \
  --checkpoint outputs/train/code_model/checkpoints/2026-07-19_16-32-32/step_002000.pt \
  --prompt "def multiply" \
  --tokens 300
```

The checkpoint supplies:

- The trained model weights.
- The model configuration.
- The tokenizer vocabulary.
- The training step at which it was saved.

This ensures generation reconstructs the same model architecture used during training.

## View all options

```bash
uv run src/generate.py --help
```

Available options:

| Parameter      | Meaning                                                                            |
| -------------- | ---------------------------------------------------------------------------------- |
| `--checkpoint` | Path to a particular `.pt` checkpoint. The newest checkpoint is used when omitted. |
| `--prompt`     | Starting text given to the model. Defaults to `def `.                              |
| `--tokens`     | Number of new characters to generate. Defaults to the saved configuration.         |

## How generation works

For each new character, the program:

1. Keeps only the most recent `context_length` token IDs.
2. Runs those token IDs through the language model.
3. Selects the logits produced at the final sequence position.
4. Applies softmax to produce next-character probabilities.
5. Samples one character from that distribution.
6. Appends the character and repeats.

The generated text may look imperfect because the model and dataset are intentionally tiny. More training can improve how closely it reproduces the dataset's patterns, but a small dataset also makes overfitting likely.

## Common errors

### No checkpoint found

```text
No checkpoint found. Run train.py first.
```

Train a model first:

```bash
uv run src/train.py
```

### Unknown prompt character

A `KeyError` during prompt encoding usually means the prompt contains a character that did not occur in the training dataset. Use characters from `datasets/dataset.txt` or add the missing characters and retrain.

### Checkpoint architecture mismatch

Always generate using the configuration stored inside the checkpoint. The generation script handles this automatically; do not manually replace the checkpoint's configuration or vocabulary.
