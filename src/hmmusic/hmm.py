"""Discrete hidden Markov model, written from scratch in NumPy

K hidden states, V observation symbols
  pi[i]    = P(z_1 = i)                      initial distribution
  A[i, j]  = P(z_t+1 = j | z_t = i)          transition matrix
  B[i, v]  = P(x_t = v | z_t = i)            emission matrix, plain HMM

Autoregressive variant (emission order 1): the symbol also depends on the previous symbol
  B[i, u, v] = P(x_t = v | z_t = i, x_t-1 = u)   with u = V meaning "no previous symbol"
Inference is identical once we have E[t, i] = P(x_t | z_t = i, ...), so every algorithm
below works on that (T, K) emission matrix

P(x_1..x_T) = sum over all K^T state paths of pi[z_1] B[z_1, x_1] A[z_1, z_2] B[z_2, x_2] ...
The forward algorithm computes this sum in O(T K^2) instead of O(K^T)

Numerical stability: probabilities of long sequences underflow (0.1^1000 = 0 in float64)
We use scaling (Rabiner 1989): after each step the forward vector is normalized to sum 1
and the normalizer c_t is kept; log P(x) = sum_t log c_t
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Posteriors:
    """Result of forward-backward on one sequence"""

    log_likelihood: float  # natural log
    gamma: np.ndarray  # (T, K) P(z_t = i | x)
    xi_sum: np.ndarray  # (K, K) sum over t of P(z_t = i, z_t+1 = j | x)


class HMM:
    def __init__(self, pi: np.ndarray, A: np.ndarray, B: np.ndarray):
        self.pi = np.asarray(pi, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        K, V = self.B.shape[0], self.B.shape[-1]
        if self.pi.shape != (K,) or self.A.shape != (K, K):
            raise ValueError("pi must be (K,), A (K, K), B (K, V) or (K, V + 1, V)")
        if self.B.ndim == 3 and self.B.shape[1] != V + 1:
            raise ValueError("autoregressive B must be (K, V + 1, V)")

    @property
    def n_states(self) -> int:
        return self.B.shape[0]

    @property
    def n_obs(self) -> int:
        return self.B.shape[-1]

    @property
    def emission_order(self) -> int:
        return self.B.ndim - 2  # 0 plain, 1 autoregressive

    @classmethod
    def random(cls, n_states: int, n_obs: int, rng: np.random.Generator, emission_order: int = 0) -> HMM:
        """Random rows drawn from a flat Dirichlet, a common starting point for Baum-Welch"""
        shape = (n_states,) if emission_order == 0 else (n_states, n_obs + 1)
        return cls(
            rng.dirichlet(np.ones(n_states)),
            rng.dirichlet(np.ones(n_states), size=n_states),
            rng.dirichlet(np.ones(n_obs), size=shape),
        )

    @classmethod
    def fit_supervised(
        cls, obs_seqs, state_seqs, n_states: int, n_obs: int, alpha: float = 1.0, emission_order: int = 0
    ) -> HMM:
        """Maximum likelihood with known states: count and normalize, plus alpha pseudo counts"""
        pi = np.full(n_states, alpha)
        A = np.full((n_states, n_states), alpha)
        B = np.full((n_states,) + (n_obs + 1,) * emission_order + (n_obs,), alpha)
        for x, z in zip(obs_seqs, state_seqs):
            pi[z[0]] += 1
            np.add.at(A, (z[:-1], z[1:]), 1)
            np.add.at(B, (z,) + _previous(x, n_obs, emission_order) + (x,), 1)
        return cls(pi / pi.sum(), A / A.sum(1, keepdims=True), B / B.sum(-1, keepdims=True))

    def emissions(self, x: np.ndarray) -> np.ndarray:
        """E[t, i] = probability of x_t in state i (given x_t-1 for the autoregressive variant)"""
        return self.B[(slice(None),) + _previous(x, self.n_obs, self.emission_order) + (x,)].T

    # ---- inference ----

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scaled forward pass

        alpha_hat[t, i] = P(z_t = i | x_1..x_t)
        c[t] = P(x_t | x_1..x_t-1), so log P(x) = sum log c
        """
        E = self.emissions(x)
        T, K = len(x), self.n_states
        alpha = np.empty((T, K))
        c = np.empty(T)
        a = self.pi * E[0]
        for t in range(T):
            if t > 0:
                # predict with A, then weight by how well each state explains x_t
                a = (alpha[t - 1] @ self.A) * E[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        return alpha, c

    def backward(self, x: np.ndarray, c: np.ndarray, E: np.ndarray | None = None) -> np.ndarray:
        """Scaled backward pass with the forward scales

        beta_hat[t, i] = P(x_t+1..x_T | z_t = i) / P(x_t+1..x_T | x_1..x_t)
        so that alpha_hat[t] * beta_hat[t] = P(z_t | x), the posterior
        """
        E = self.emissions(x) if E is None else E
        T, K = len(x), self.n_states
        beta = np.empty((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (E[t + 1] * beta[t + 1]) / c[t + 1]
        return beta

    def log_likelihood(self, x: np.ndarray) -> float:
        """Natural log P(x)"""
        _, c = self.forward(x)
        return float(np.log(c).sum())

    def bits_per_symbol(self, seqs: list[np.ndarray]) -> float:
        total = sum(self.log_likelihood(x) for x in seqs)
        return -total / np.log(2) / sum(len(x) for x in seqs)

    def posteriors(self, x: np.ndarray) -> Posteriors:
        E = self.emissions(x)
        alpha, c = self.forward(x)
        beta = self.backward(x, c, E)
        gamma = alpha * beta
        # xi_t[i, j] = alpha_hat[t, i] A[i, j] E[t+1, j] beta_hat[t+1, j] / c[t+1]
        # summed over t in one matrix product: (K, T-1) @ (T-1, K), then times A elementwise
        weighted_next = E[1:] * beta[1:] / c[1:, None]  # (T-1, K)
        xi_sum = (alpha[:-1].T @ weighted_next) * self.A
        return Posteriors(float(np.log(c).sum()), gamma, xi_sum)

    def posterior_decode(self, x: np.ndarray) -> np.ndarray:
        """Most likely state at each time separately (argmax of gamma)

        Maximizes the expected number of correct states, which is what per-note accuracy
        measures; the result need not be a valid path (it may use transitions with A = 0)
        """
        return self.posteriors(x).gamma.argmax(axis=1)

    def viterbi(self, x: np.ndarray) -> np.ndarray:
        """Most likely state path, computed in log space (max instead of sum, no underflow)"""
        T, K = len(x), self.n_states
        with np.errstate(divide="ignore"):
            log_A, log_E, log_pi = np.log(self.A), np.log(self.emissions(x)), np.log(self.pi)
        delta = log_pi + log_E[0]
        back = np.empty((T, K), dtype=int)
        for t in range(1, T):
            scores = delta[:, None] + log_A  # scores[i, j]: best path ending in i, then i -> j
            back[t] = scores.argmax(axis=0)
            delta = scores.max(axis=0) + log_E[t]
        path = np.empty(T, dtype=int)
        path[-1] = delta.argmax()
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    # ---- learning ----

    def baum_welch(self, seqs: list[np.ndarray], n_iter: int = 50, tol: float = 1e-4, alpha: float = 0.0) -> list[float]:
        """EM on the training sequences, updates the parameters in place

        E step: posteriors of states and transitions with the current parameters
        M step: expected counts, normalized (alpha adds pseudo counts, 0 = pure maximum likelihood)
        Returns the total log likelihood before each M step; it never decreases when alpha = 0
        Stops when the improvement per symbol falls below tol (in nats)
        """
        n_symbols = sum(len(x) for x in seqs)
        history: list[float] = []
        for _ in range(n_iter):
            pi = np.full(self.n_states, alpha)
            A = np.full((self.n_states, self.n_states), alpha)
            B = np.full(self.B.shape, alpha)
            total = 0.0
            for x in seqs:
                post = self.posteriors(x)
                total += post.log_likelihood
                pi += post.gamma[0]
                A += post.xi_sum
                # add gamma[t] to the emission entry used at time t, for every state at once
                # moving the state axis last lets one index tuple address all K entries
                np.add.at(np.moveaxis(B, 0, -1), _previous(x, self.n_obs, self.emission_order) + (x,), post.gamma)
            history.append(total)
            # rows that received no mass (unused states) keep their old values
            self.pi = pi / pi.sum()
            self.A = _normalize_rows(A, self.A)
            self.B = _normalize_rows(B, self.B)
            if len(history) > 1 and (history[-1] - history[-2]) / n_symbols < tol:
                break
        return history

    # ---- generation ----

    def sample(self, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        z = np.empty(length, dtype=int)
        x = np.empty(length, dtype=int)
        z[0] = rng.choice(self.n_states, p=self.pi)
        for t in range(length):
            if t > 0:
                z[t] = rng.choice(self.n_states, p=self.A[z[t - 1]])
            if self.emission_order == 0:
                p = self.B[z[t]]
            else:
                p = self.B[z[t], x[t - 1] if t > 0 else self.n_obs]
            x[t] = rng.choice(self.n_obs, p=p)
        return z, x


def _previous(x: np.ndarray, n_obs: int, emission_order: int) -> tuple[np.ndarray, ...]:
    """Index tuple for the conditioning symbol: () for a plain HMM, (x_t-1,) otherwise"""
    if emission_order == 0:
        return ()
    return (np.r_[n_obs, x[:-1]],)


def _normalize_rows(counts: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Normalize over the last axis; rows with no counts keep their old values"""
    sums = counts.sum(axis=-1, keepdims=True)
    return np.where(sums > 0, counts / np.where(sums > 0, sums, 1), fallback)
