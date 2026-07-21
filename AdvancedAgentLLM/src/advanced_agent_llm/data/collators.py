"""Packed pretraining examples already have equal length, so PyTorch's
default collator can stack them directly. SFT will add a masked collator later.
"""
