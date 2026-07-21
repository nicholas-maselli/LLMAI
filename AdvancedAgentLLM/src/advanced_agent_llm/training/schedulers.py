import math


def cosine_learning_rate(
    step: int,
    maximum_steps: int,
    warmup_steps: int,
    maximum_learning_rate: float,
    minimum_learning_rate_ratio: float,
) -> float:
    minimum_learning_rate = maximum_learning_rate * minimum_learning_rate_ratio
    if warmup_steps > 0 and step < warmup_steps:
        return maximum_learning_rate * (step + 1) / warmup_steps
    if step >= maximum_steps:
        return minimum_learning_rate

    decay_steps = max(1, maximum_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_learning_rate + cosine * (
        maximum_learning_rate - minimum_learning_rate
    )
