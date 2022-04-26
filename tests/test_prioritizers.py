import random
from typing import List, Tuple

import numpy as np
import pytest

from dnn_tip import prioritizers


def get_example(seed) -> Tuple[np.ndarray, List[str]]:
    """Returns the example given in the DeepGini Paper

    A seed allows to shuffle the entries (order should not matter)."""
    examples_from_paper = [
        [True, True, True, False, False, True, True, True],
        [True, True, True, False, False, False, True, True],
        [True, True, True, True, False, False, False, False],
        [False, False, False, False, True, True, True, True],
    ]
    re_indexes = ["A", "B", "C", "D"]
    random.Random(seed).shuffle(examples_from_paper)
    random.Random(seed).shuffle(re_indexes)
    return np.array(examples_from_paper, dtype=bool), re_indexes


@pytest.mark.parametrize(
    "seed",
    [i for i in range(10)],
)
def test_ctm(seed: int):
    profile, idxs = get_example(seed=seed)
    scores = np.sum(profile, axis=1)

    predicted_order = [idxs[i] for i in prioritizers.ctm(scores)]

    assert predicted_order == ["A", "B", "C", "D"] or predicted_order == [
        "A",
        "B",
        "D",
        "C",
    ]


@pytest.mark.parametrize("seed", [i for i in range(10)])
@pytest.mark.parametrize(
    "shape", [(4, 8), (4, 8, 1), (4, 4, 2), (4, 2, 2, 2), (-1, 2, 4)]
)
def test_cam(seed: int, shape: Tuple[int]):
    profile, idxs = get_example(seed=seed)
    scores = np.sum(profile, axis=1)
    # The shape (number of dimensions) of the profile should not influence output
    profile = np.reshape(profile, shape)

    predicted_order = [idxs[i] for i in prioritizers.cam(scores, profile)]

    # Note: The DeepGini papers mentions only ["A", "D", "C", "B"] as valid solution,
    #       which is wrong.
    assert predicted_order == ["A", "D", "C", "B"] or predicted_order == [
        "A",
        "C",
        "D",
        "B",
    ]


@pytest.mark.parametrize(
    "seed, shape, prob",
    [
        (1, (20, 100), 0.1),
        (1, (200, 1000), 0.0001),
        (1, (2000, 10000), 0.01),
    ],
)
def test_cam_fuzzer(seed: int, shape: Tuple[int], prob: float):
    # Generate a random profile with the given shape and probability of a True value
    profile = np.random.random(shape) < prob
    scores = np.sum(profile, axis=1)

    # The orginal array will be modified
    profiles_copy, scores_copy = profile.copy(), scores.copy()
    predicted_order = [i for i in prioritizers.cam(scores, profile)]

    covered_nodes = np.zeros(profile.shape[1], dtype=bool)
    yielded_samples = np.zeros(profile.shape[0], dtype=bool)

    # Check a couple of consistency checks for the CAM algorithm
    last_coverage_increment = np.inf
    previous_coverage_sum = 0
    for i in predicted_order:
        assert not yielded_samples[i]
        yielded_samples[i] = True
        covered_nodes = np.logical_or(covered_nodes, profiles_copy[i])
        new_coverage_sum = np.sum(covered_nodes)
        # Check that the coverage sum increments are weakly monotonic decreasing
        assert new_coverage_sum - previous_coverage_sum <= last_coverage_increment
