"""Markov chain of order k over a small alphabet

Two smoothing methods
  additive     P(x | ctx) = (count(ctx, x) + alpha) / (count(ctx) + alpha * V)
               an unseen context gives a uniform distribution
  witten_bell  P_k(x | ctx) = lam * count(ctx, x) / count(ctx) + (1 - lam) * P_{k-1}(x | shorter ctx)
               lam = count(ctx) / (count(ctx) + number of distinct symbols seen after ctx)
               a context seen often with few different followers is trusted more,
               an unseen context falls back entirely to the shorter one
At the start of a sequence the missing context is filled with a START symbol
"""

from __future__ import annotations

import numpy as np


class MarkovChain:
    def __init__(self, n_symbols: int, order: int, alpha: float = 0.1, smoothing: str = "additive"):
        if smoothing not in ("additive", "witten_bell"):
            raise ValueError(f"unknown smoothing {smoothing!r}")
        self.n_symbols = n_symbols
        self.order = order
        self.alpha = alpha
        self.smoothing = smoothing
        self.start = n_symbols  # extra symbol, only used in contexts
        # counts[k] has shape (V + 1,) * k + (V,): k context symbols, oldest first, then the next symbol
        self.counts = [np.zeros((n_symbols + 1,) * k + (n_symbols,)) for k in range(order + 1)]
        self._probs: np.ndarray | None = None

    def _contexts(self, seq: np.ndarray, k: int) -> tuple[np.ndarray, ...]:
        """For each position t, the k previous symbols, padded with START"""
        padded = np.r_[np.full(k, self.start), seq]
        return tuple(padded[j : j + len(seq)] for j in range(k))

    def fit(self, sequences: list[np.ndarray]) -> MarkovChain:
        for k in range(self.order + 1):
            self.counts[k][:] = 0
            for seq in sequences:
                np.add.at(self.counts[k], self._contexts(seq, k) + (seq,), 1)
        self._probs = None
        return self

    def probs(self) -> np.ndarray:
        """Smoothed conditional table of the full order, sums to 1 over the last axis"""
        if self._probs is None:
            self._probs = self._additive() if self.smoothing == "additive" else self._witten_bell()
        return self._probs

    def _additive(self) -> np.ndarray:
        smoothed = self.counts[self.order] + self.alpha
        return smoothed / smoothed.sum(axis=-1, keepdims=True)

    def _witten_bell(self) -> np.ndarray:
        # order 0: add alpha so that no symbol has probability 0
        unigram = self.counts[0] + self.alpha
        probs = unigram / unigram.sum()
        for k in range(1, self.order + 1):
            c = self.counts[k]
            total = c.sum(axis=-1, keepdims=True)
            distinct = (c > 0).sum(axis=-1, keepdims=True)
            lam = np.divide(total, total + distinct, out=np.zeros_like(total), where=total > 0)
            ml = np.divide(c, total, out=np.zeros_like(c), where=total > 0)
            # probs has one context axis fewer: it ignores the oldest symbol, so a leading
            # axis of size 1 broadcasts it over every value of that oldest symbol
            probs = lam * ml + (1 - lam) * probs[np.newaxis]
        return probs

    def log2_prob(self, seq: np.ndarray) -> float:
        """Total log2 probability of a sequence"""
        p = self.probs()[self._contexts(seq, self.order) + (seq,)]
        return float(np.log2(p).sum())

    def bits_per_symbol(self, sequences: list[np.ndarray]) -> float:
        total = sum(self.log2_prob(s) for s in sequences)
        return -total / sum(len(s) for s in sequences)

    def sample(self, length: int, rng: np.random.Generator) -> np.ndarray:
        probs = self.probs()
        context = [self.start] * self.order
        out = np.empty(length, dtype=int)
        for t in range(length):
            out[t] = rng.choice(self.n_symbols, p=probs[tuple(context)])
            if self.order:
                context = context[1:] + [int(out[t])]
        return out
