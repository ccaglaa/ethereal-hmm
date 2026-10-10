import numpy as np

from hmmusic.stats import permutation_test


def test_identical_groups_give_p_one():
    assert permutation_test(np.array([3, 3, 3]), np.array([3, 3, 3]), n=200) == 1.0


def test_clear_difference_gives_small_p():
    p = permutation_test(np.array([5, 5, 5, 5, 5, 4]), np.array([1, 1, 2, 1, 1, 2]), n=2000)
    assert p < 0.01


def test_p_value_is_never_zero():
    # with 6 + 6 values there are 924 splits; even the most extreme one keeps p >= 1 / (n + 1)
    p = permutation_test(np.array([5.0] * 6), np.array([1.0] * 6), n=500)
    assert 0 < p < 0.01


def test_exact_value_small_case():
    # groups of 2 and 2 from [1, 2, 3, 4]: 6 equally likely splits, 2 are as extreme as (1, 2) vs (3, 4)
    p = permutation_test(np.array([1, 2]), np.array([3, 4]), n=20000, seed=1)
    assert abs(p - 2 / 6) < 0.02
