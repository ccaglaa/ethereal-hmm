"""HMM tests: every inference result is checked against brute force over all K^T state paths"""

import itertools

import numpy as np
import pytest

from hmmusic.hmm import HMM


def _small_model(seed=0, K=3, V=4, order=0):
    return HMM.random(K, V, np.random.default_rng(seed), emission_order=order)


def _emit(model, z, x, t):
    """Emission probability written out by hand, independent of HMM.emissions"""
    if model.emission_order == 0:
        return model.B[z[t], x[t]]
    previous = x[t - 1] if t > 0 else model.n_obs
    return model.B[z[t], previous, x[t]]


def _path_prob(model, x, z):
    p = model.pi[z[0]] * _emit(model, z, x, 0)
    for t in range(1, len(x)):
        p *= model.A[z[t - 1], z[t]] * _emit(model, z, x, t)
    return p


def _all_paths(model, T):
    return list(itertools.product(range(model.n_states), repeat=T))


X = np.array([0, 2, 1, 3, 3])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("order", [0, 1])
def test_likelihood_matches_brute_force(seed, order):
    model = _small_model(seed, order=order)
    brute = sum(_path_prob(model, X, z) for z in _all_paths(model, len(X)))
    assert model.log_likelihood(X) == pytest.approx(np.log(brute))


@pytest.mark.parametrize("order", [0, 1])
def test_likelihood_sums_to_one_over_sequences(order):
    model = _small_model(3, K=2, V=3, order=order)
    total = sum(np.exp(model.log_likelihood(np.array(x))) for x in itertools.product(range(3), repeat=4))
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("order", [0, 1])
def test_gamma_and_xi_match_brute_force(seed, order):
    model = _small_model(seed, order=order)
    K, T = model.n_states, len(X)
    gamma = np.zeros((T, K))
    xi = np.zeros((K, K))
    for z in _all_paths(model, T):
        p = _path_prob(model, X, z)
        gamma[np.arange(T), z] += p
        for t in range(T - 1):
            xi[z[t], z[t + 1]] += p
    total = gamma[0].sum()
    post = model.posteriors(X)
    np.testing.assert_allclose(post.gamma, gamma / total)
    np.testing.assert_allclose(post.xi_sum, xi / total)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("order", [0, 1])
def test_viterbi_matches_brute_force(seed, order):
    model = _small_model(seed, order=order)
    best = max(_all_paths(model, len(X)), key=lambda z: _path_prob(model, X, z))
    assert model.viterbi(X).tolist() == list(best)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_posterior_decode_matches_brute_force_marginals(seed):
    model = _small_model(seed, order=1)
    T = len(X)
    gamma = np.zeros((T, model.n_states))
    for z in _all_paths(model, T):
        gamma[np.arange(T), z] += _path_prob(model, X, z)
    assert model.posterior_decode(X).tolist() == gamma.argmax(1).tolist()


def test_long_sequence_does_not_underflow():
    model = _small_model(0)
    x = np.random.default_rng(0).integers(0, 4, 20000)
    ll = model.log_likelihood(x)
    assert np.isfinite(ll) and ll < -1000
    post = model.posteriors(x)
    np.testing.assert_allclose(post.gamma.sum(1), 1.0)


@pytest.mark.parametrize("order", [0, 1])
def test_baum_welch_never_decreases(order):
    rng = np.random.default_rng(0)
    truth = _small_model(5, order=order)
    seqs = [truth.sample(50, rng)[1] for _ in range(20)]
    model = HMM.random(3, 4, np.random.default_rng(1), emission_order=order)
    history = model.baum_welch(seqs, n_iter=30, tol=0.0)
    assert len(history) == 30
    assert all(b >= a - 1e-9 for a, b in zip(history, history[1:]))
    for m in (model.pi, model.A, model.B):
        np.testing.assert_allclose(np.atleast_2d(m).sum(-1), 1.0)


def test_baum_welch_recovers_a_clear_model():
    # two states that emit disjoint symbols and rarely switch: EM should find them (up to relabeling)
    truth = HMM([0.5, 0.5], [[0.95, 0.05], [0.05, 0.95]], [[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]])
    rng = np.random.default_rng(0)
    seqs = [truth.sample(100, rng)[1] for _ in range(30)]
    model = HMM.random(2, 4, np.random.default_rng(2))
    model.baum_welch(seqs, n_iter=100)
    order = np.argsort(model.B[:, 0])[::-1]  # state emitting symbol 0 first
    np.testing.assert_allclose(model.B[order], truth.B, atol=0.05)
    np.testing.assert_allclose(model.A[np.ix_(order, order)], truth.A, atol=0.05)


def test_baum_welch_pseudo_counts_keep_unseen_pairs_possible():
    # symbol 2 never follows symbol 0 in training; with alpha > 0 it still has probability > 0
    seqs = [np.array([0, 1, 0, 1, 2, 2])] * 5
    ml = HMM.random(2, 3, np.random.default_rng(0), emission_order=1)
    ml.baum_welch(seqs, n_iter=50, tol=0.0)
    smoothed = HMM.random(2, 3, np.random.default_rng(0), emission_order=1)
    smoothed.baum_welch(seqs, n_iter=50, tol=0.0, alpha=0.1)
    unseen = np.array([0, 2])
    assert np.isfinite(smoothed.log_likelihood(unseen))
    assert (smoothed.B > 0).all()
    assert np.allclose(ml.B[:, 0, 2], 0.0)


def test_fit_supervised_counts():
    x = [np.array([0, 1, 1]), np.array([1, 0])]
    z = [np.array([0, 0, 1]), np.array([1, 1])]
    model = HMM.fit_supervised(x, z, n_states=2, n_obs=2, alpha=0.0)
    np.testing.assert_allclose(model.pi, [0.5, 0.5])
    np.testing.assert_allclose(model.A, [[0.5, 0.5], [0.0, 1.0]])  # 0->0, 0->1, 1->1
    np.testing.assert_allclose(model.B, [[0.5, 0.5], [1 / 3, 2 / 3]])


def test_fit_supervised_autoregressive_counts():
    x = [np.array([0, 1, 1])]
    z = [np.array([0, 0, 0])]
    model = HMM.fit_supervised(x, z, n_states=1, n_obs=2, alpha=0.0, emission_order=1)
    # state 0 sees: start -> 0, 0 -> 1, 1 -> 1 (each previous symbol once)
    assert model.B[0, 2].tolist() == [1.0, 0.0]
    assert model.B[0, 0].tolist() == [0.0, 1.0]
    assert model.B[0, 1].tolist() == [0.0, 1.0]


def test_baum_welch_autoregressive_matches_one_manual_em_step():
    # one EM step by brute force: expected counts from the exact path posteriors
    model = _small_model(7, K=2, V=3, order=1)
    seqs = [np.array([0, 2, 2, 1]), np.array([1, 1, 0])]
    expected_B = np.zeros(model.B.shape)
    for x in seqs:
        paths = _all_paths(model, len(x))
        probs = np.array([_path_prob(model, x, z) for z in paths])
        probs /= probs.sum()
        for z, p in zip(paths, probs):
            for t in range(len(x)):
                previous = x[t - 1] if t > 0 else model.n_obs
                expected_B[z[t], previous, x[t]] += p
    sums = expected_B.sum(-1, keepdims=True)
    old_B = model.B.copy()
    model.baum_welch(seqs, n_iter=1)
    np.testing.assert_allclose(model.B, np.where(sums > 0, expected_B / np.where(sums > 0, sums, 1), old_B))


def test_sample_autoregressive_uses_previous_symbol():
    # one state; after 0 always 1, after 1 always 0, first symbol always 0
    B = np.array([[[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]])
    model = HMM([1.0], [[1.0]], B)
    _, x = model.sample(6, np.random.default_rng(0))
    assert x.tolist() == [0, 1, 0, 1, 0, 1]


def test_sample_frequencies():
    model = HMM([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.2, 0.8]])
    z, x = model.sample(20000, np.random.default_rng(0))
    assert z[:4].tolist() == [0, 1, 0, 1]
    assert x[z == 0].mean() == 0
    assert x[z == 1].mean() == pytest.approx(0.8, abs=0.02)


def test_bad_shapes():
    with pytest.raises(ValueError):
        HMM(np.ones(2) / 2, np.eye(3), np.ones((2, 2)) / 2)
    with pytest.raises(ValueError):
        HMM(np.ones(2) / 2, np.eye(2), np.ones((2, 2, 2)) / 2)  # needs (K, V + 1, V)
