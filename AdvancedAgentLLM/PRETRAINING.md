# Pretraining guide

Run all commands from `AdvancedAgentLLM` in Git Bash.

## Install the environment

```bash
uv sync --dev
```

## 1. Train a tokenizer

FineWeb streaming:

```bash
uv run python scripts/train_tokenizer.py \
  --output artifacts/tokenizer.json \
  --vocabulary-size 50304 \
  --documents 100000
```

For a quick local experiment, place one document on each line of a UTF-8 text
file and use:

```bash
uv run python scripts/train_tokenizer.py \
  --local-text datasets/pretrain.txt \
  --output artifacts/tokenizer.json \
  --vocabulary-size 8000 \
  --documents 10000
```

## 2. Run a small local smoke training

```bash
uv run python scripts/pretrain.py \
  --tokenizer artifacts/tokenizer.json \
  --local-text datasets/pretrain.txt \
  --run-name local-smoke \
  --context-length 128 \
  --hidden-size 128 \
  --layers 4 \
  --attention-heads 4 \
  --kv-heads 2 \
  --batch-size 4 \
  --gradient-accumulation 2 \
  --steps 1000 \
  --learning-rate 0.0003 \
  --warmup-steps 100 \
  --log-every 10 \
  --validate-every 100 \
  --checkpoint-every 100 \
  --precision fp32
```

Use `--validation-batches 0` if a tiny local corpus is too small to produce a
validation batch.

## 3. Stream FineWeb

Omit `--local-text` to use the configured Hugging Face dataset:

```bash
uv run python scripts/pretrain.py \
  --tokenizer artifacts/tokenizer.json \
  --run-name fineweb-base \
  --context-length 2048 \
  --hidden-size 768 \
  --layers 12 \
  --attention-heads 12 \
  --kv-heads 4 \
  --batch-size 8 \
  --gradient-accumulation 4 \
  --steps 10000 \
  --precision bf16 \
  --checkpoint-every 500
```

The effective token batch per optimizer step is:

```text
batch_size * gradient_accumulation * context_length
```

## Resume a checkpoint

Use the same model and tokenizer arguments, and set a larger total step count:

```bash
uv run python scripts/pretrain.py \
  --tokenizer artifacts/tokenizer.json \
  --context-length 2048 \
  --hidden-size 768 \
  --layers 12 \
  --attention-heads 12 \
  --kv-heads 4 \
  --steps 20000 \
  --resume outputs/pretrain/fineweb-base_DATE/checkpoints/step_00010000.pt
```

Model, optimizer, scaler, step, token count, and RNG state are restored. A
streamed dataset restarts its iterator, so this educational implementation does
not yet restore the exact remote dataset cursor.
