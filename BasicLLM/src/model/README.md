# Model components

Each major part of the transformer lives in a focused module:

- `embeddings.py` contains token, position, and combined input embeddings.
- `transformer_block/` contains the smaller pieces used by a transformer block.
- `language_model.py` stacks four transformer blocks and produces vocabulary logits.

We add each component only when we reach it in the lesson.
