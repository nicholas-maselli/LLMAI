# Transformer block

A transformer block will be assembled from several smaller components:

1. Self-attention lets tokens gather information from other tokens.
2. A feed-forward network processes each token's resulting vector.
3. Residual connections preserve earlier information.
4. Layer normalization keeps the values stable.

We are currently building self-attention one step at a time.

- `attention.py` contains the operations for one attention head.
- `multi_head_attention.py` runs several attention heads in parallel.
- `feed_forward.py` processes each token with a small neural network.
- `block.py` assembles attention, normalization, residual connections, and the feed-forward network.
