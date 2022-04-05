import numpy as np

from dnn_tip.neuron_coverage import NAC, KMNC, NBC, SNAC, TKNC

ACTIVATIONS_1 = [
    np.array([[0.1, 0.4, 0.9, 0.4], [0.1, 0.9, 0.9, 0.4]]),
    np.array([[0.3, 0.2, 0.1, 0.6, 0.8], [0.3, 0.9, 0.1, 0.6, 0.8]]),
    np.array([[0.2, 0.3, 0.4, 0.4], [0.2, 0.9, 0.4, 0.4]]),
]


def test_nac():
    score, profile = NAC(cov_threshold=0.55)(ACTIVATIONS_1)
    assert np.all(score == np.array([3, 6]))
    assert np.all(
        profile[0]
        == np.concatenate(
            [
                [
                    False,
                    False,
                    True,
                    False,
                ],  # Layer 1
                [
                    False,
                    False,
                    False,
                    True,
                    True,
                ],  # Layer 2
                [False, False, False, False],  # Layer 3
            ]
        )
    )


def test_kmnc():
    mins = [np.array([0] * 4), np.array([0] * 5), np.array([0.1] * 4)]
    maxs = [np.array([1] * 4), np.array([1] * 5), np.array([0.95] * 4)]
    score, profile = KMNC(mins, maxs, 2)(ACTIVATIONS_1)
    assert np.all(score == np.array([13, 13]))
    assert np.all(
        profile[0]
        == np.concatenate(
            [
                [[True, False], [True, False], [False, True], [True, False]],  # Layer 1
                [
                    [True, False],
                    [True, False],
                    [True, False],
                    [False, True],
                    [False, True],
                ],  # Layer 2
                [[True, False], [True, False], [True, False], [True, False]],  # Layer 3
            ]
        )
    )

    outside_boundary = [a.copy() for a in ACTIVATIONS_1]
    outside_boundary[0][0][0] = -0.5
    outside_boundary[1][0][0] = 1.5
    score, profile = KMNC(mins, maxs, 2)(outside_boundary)
    assert np.all(score == np.array([11, 13]))


def test_nbc():
    #  TEST CONSTANTS
    mins = [np.array([0] * 4), np.array([0] * 5), np.array([0.1] * 4)]
    maxs = [np.array([1] * 4), np.array([1] * 5), np.array([0.95] * 4)]
    zero_std = [np.array([0] * 4), np.array([0] * 5), np.array([0] * 4)]
    point_two_std = [np.array([0.2] * 4), np.array([0.2] * 5), np.array([0.2] * 4)]

    # Check case where nothing is outside of boundary
    score, profile = NBC(mins, maxs, zero_std, scaler=1)(ACTIVATIONS_1)

    assert np.all(score == np.array([0, 0]))
    assert np.all(
        profile[0]
        == np.concatenate(
            [
                [
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                ],  # Layer 1
                [
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                ],  # Layer 2
                [
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                ],  # Layer 3
            ]
        )
    )

    # Check with items outside boundary, but without std
    outside_boundary = [a.copy() for a in ACTIVATIONS_1]
    outside_boundary[0][0][0] = -0.1
    outside_boundary[1][0][0] = 1.5
    score, profile = NBC(mins, maxs, zero_std, scaler=1)(outside_boundary)
    assert np.all(score == np.array([2, 0]))

    # Check with std
    score, profile = NBC(mins, maxs, point_two_std, scaler=1)(outside_boundary)
    assert np.all(score == np.array([1, 0]))

    # Check with std and scaler
    score, profile = NBC(mins, maxs, point_two_std, scaler=6)(outside_boundary)
    assert np.all(score == np.array([0, 0]))


def test_snac():
    #  TEST CONSTANTS
    maxs = [np.array([1] * 4), np.array([1] * 5), np.array([0.95] * 4)]
    zero_std = [np.array([0] * 4), np.array([0] * 5), np.array([0] * 4)]
    point_two_std = [np.array([0.2] * 4), np.array([0.2] * 5), np.array([0.2] * 4)]

    # Check case where nothing is outside of boundary
    score, profile = SNAC(maxs, zero_std, scaler=1)(ACTIVATIONS_1)

    assert np.all(score == np.array([0, 0]))
    assert np.all(
        profile[0]
        == np.concatenate(
            [
                [False] * 4,
                [False] * 5,
                [False] * 4,
            ]
        )
    )

    # Check with items outside boundary, but without std
    outside_boundary = [a.copy() for a in ACTIVATIONS_1]
    outside_boundary[0][0][0] = -0.1
    outside_boundary[1][0][0] = 1.5
    score, profile = SNAC(maxs, zero_std, scaler=1)(outside_boundary)
    assert np.all(score == np.array([1, 0]))

    # Check with std
    score, profile = SNAC(maxs, point_two_std, scaler=1)(outside_boundary)
    assert np.all(score == np.array([1, 0]))

    # Check with std and scaler
    score, profile = SNAC(maxs, point_two_std, scaler=6)(outside_boundary)
    assert np.all(score == np.array([0, 0]))


def test_tknc():
    score, profile = TKNC(2)(ACTIVATIONS_1)
    assert np.all(score == np.array([6, 6]))
    # Layer one (two possible valid outcomes)
    assert np.all(profile[0][:4] == np.array([False, True, True, False])) or np.all(
        profile[0][:4] == np.array([False, False, True, True])
    )
    # Layer two
    assert np.all(profile[0][4:9] == np.array([False, False, False, True, True]))
    # Layer three
    assert np.all(profile[0][9:] == np.array([False, False, True, True]))
