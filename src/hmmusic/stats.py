"""Small statistics for the listening test"""

from __future__ import annotations

import numpy as np


def permutation_test(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> float:
    """Two sided p value for a difference of means between a and b

    Shuffle the group labels n times; p = share of shuffles with a difference at least as large as the
    observed one. No normality assumption, suited to small samples of 1 to 5 ratings
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    observed = abs(a.mean() - b.mean())
    pooled = np.r_[a, b]
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        rng.shuffle(pooled)
        hits += abs(pooled[: len(a)].mean() - pooled[len(a) :].mean()) >= observed - 1e-12
    return (hits + 1) / (n + 1)  # +1: the observed labelling is one of the possible shuffles
