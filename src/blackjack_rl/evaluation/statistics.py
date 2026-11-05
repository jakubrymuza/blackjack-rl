from statistics import NormalDist

import numpy as np

from blackjack_rl.config import Config


def mean_interval(values, config: Config, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, dtype=float)
    means = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(config.evaluation.bootstrap_resamples)
    ]
    alpha = (1 - config.evaluation.confidence) / 2
    return np.quantile(means, [alpha, 1 - alpha]).tolist()


def proportion_interval(successes: int, total: int, confidence: float) -> list[float]:
    """Wilson interval remains informative when no failures were observed."""
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    width = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0, center - width), min(1, center + width)]
