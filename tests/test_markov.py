import itertools

import numpy as np
import pytest

from hmmusic.markov import MarkovChain


def _random_sequences(rng, n_symbols=4, n=20, length=30):
    return [rng.integers(0, n_symbols, length) for _ in range(n)]


@pytest.mark.parametrize("smoothing", ["additive", "witten_bell"])
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_probs_sum_to_one(smoothing, order):
    rng = np.random.default_rng(0)
    model = MarkovChain(4, order, 0.5, smoothing).fit(_random_sequences(rng))
    np.testing.assert_allclose(model.probs().sum(axis=-1), 1.0)
    assert (model.probs() > 0).all()


def test_additive_known_values():
    # sequence 0 1 0 1 with order 1: after START comes 0, after 0 always 1, after 1 always 0
    model = MarkovChain(2, 1, alpha=1.0).fit([np.array([0, 1, 0, 1])])
    p = model.probs()
    assert p[2, 0] == pytest.approx(2 / 3)  # START -> 0 seen once: (1 + 1) / (1 + 2)
    assert p[0, 1] == pytest.approx(3 / 4)  # 0 -> 1 seen twice: (2 + 1) / (2 + 2)
    assert p[1, 0] == pytest.approx(2 / 3)


def test_log2_prob_matches_product_of_conditionals():
    rng = np.random.default_rng(1)
    model = MarkovChain(3, 2, 0.3, "witten_bell").fit(_random_sequences(rng, 3))
    seq = np.array([2, 0, 1, 1, 0])
    p, start = model.probs(), model.start
    padded = [start, start] + seq.tolist()
    expected = sum(np.log2(p[padded[t], padded[t + 1], padded[t + 2]]) for t in range(len(seq)))
    assert model.log2_prob(seq) == pytest.approx(expected)


def test_log2_prob_sums_to_one_over_all_sequences():
    # a proper model gives total probability 1 to all sequences of a fixed length
    rng = np.random.default_rng(2)
    model = MarkovChain(3, 2, 0.2, "witten_bell").fit(_random_sequences(rng, 3))
    total = sum(2 ** model.log2_prob(np.array(s)) for s in itertools.product(range(3), repeat=4))
    assert total == pytest.approx(1.0)


def test_witten_bell_unseen_context_backs_off():
    model = MarkovChain(3, 2, 0.1, "witten_bell").fit([np.array([0, 1, 2, 0, 1, 2])])
    lower = MarkovChain(3, 1, 0.1, "witten_bell").fit([np.array([0, 1, 2, 0, 1, 2])])
    # context (2, 2) never occurs, so the order 2 model uses the order 1 row for context 2
    np.testing.assert_allclose(model.probs()[2, 2], lower.probs()[2])


def test_witten_bell_lambda():
    # after 0 we saw 1 twice and 2 once: count 3, distinct 2, lam = 3 / 5
    model = MarkovChain(3, 1, 1.0, "witten_bell").fit([np.array([0, 1, 0, 1, 0, 2])])
    unigram = (np.array([3, 2, 1]) + 1.0) / 9
    expected = 0.6 * np.array([0, 2 / 3, 1 / 3]) + 0.4 * unigram
    np.testing.assert_allclose(model.probs()[0], expected)


def test_sample_follows_deterministic_chain():
    model = MarkovChain(3, 1, alpha=1e-9).fit([np.array([0, 1, 2] * 50)])
    out = model.sample(9, np.random.default_rng(0))
    assert out.tolist() == [0, 1, 2] * 3


def test_more_data_order_helps_on_structured_data():
    seqs = [np.array([0, 1, 2, 3] * 20) for _ in range(5)]
    bits = [MarkovChain(4, k, 0.01).fit(seqs).bits_per_symbol(seqs) for k in (0, 1)]
    assert bits[1] < bits[0]


def test_unknown_smoothing():
    with pytest.raises(ValueError):
        MarkovChain(3, 1, smoothing="kneser")
