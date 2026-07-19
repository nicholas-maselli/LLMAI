# Tiny Transformer — One Step at a Time

We are building a tiny language model incrementally. The code intentionally contains only the concepts discussed so far.

Current step:

1. Load the text in `dataset.txt`.
2. Discover the unique characters that make up its vocabulary.
3. Encode the dataset and create training batches.
4. Convert token IDs into learnable embedding vectors.

The project is organized by concept:

- `src/tokenizer.py` converts between characters and token IDs.
- `src/data.py` loads the dataset and creates training examples and batches.
- `src/model.py` contains the model components.
- `datasets/dataset.txt` contains the raw training text.

There is no training or transformer yet. We will add each piece only after discussing the current code.
