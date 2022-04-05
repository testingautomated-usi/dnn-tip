"""Prioritizers, such as Coverage-Total Method and Coverage-Additional Method"""
from typing import Generator

import numpy as np


def ctm(scores: np.ndarray) -> Generator[int, None, None]:
    """Indexes according to Coverage-Total Method"""
    assert len(scores.shape) == 1
    # Sort negative scores to achieve decreasing order
    idxs = np.argsort(-scores)
    for x in idxs:
        yield x


def cam(scores: np.ndarray, profiles: np.ndarray) -> Generator[int, None, None]:
    """Indexes according to Coverage-Total Method (i.e., greedily increasing overall coverage)"""
    scores = scores.copy()
    profiles = profiles.reshape((profiles.shape[0], -1))
    uncovered = np.ones_like(profiles[0])
    num_coverable = np.sum(profiles, axis=1).flatten()
    remaining = np.sum(uncovered)
    yielded = np.zeros_like(scores)
    while True:
        next = np.argmax(num_coverable)
        covering_columns = profiles[next].nonzero()[0]
        newly_covered = num_coverable[next]

        if newly_covered == 0:
            break

        yield next
        yielded[next] = 1

        # Update uncovered
        remaining -= newly_covered  # Faster than np.sum(uncovered) after updating
        num_coverable_deductions = np.sum(profiles[:, covering_columns], axis=1)
        num_coverable = num_coverable - num_coverable_deductions
        uncovered[covering_columns] = 0
        profiles[:, covering_columns] = 0

        if remaining == 0:
            break

    # Sort remaining according to original scores and return
    min_score = np.min(scores) - 1
    # Make sure already yealed samples have a very low score and are at the and of the ordering
    scores[yielded.nonzero()[0]] = min_score - 1
    idxs = np.argsort(-scores)
    for x in idxs:
        # a score < min_score stands for a sample that was already yielded
        #   (and so will all further ones), so we end the loop
        if scores[x] < min_score:
            break
        else:
            yield x
            yielded[x] = True  # Unneeded, just for assertion below

    assert np.all(yielded)
