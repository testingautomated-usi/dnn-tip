"""Neuron-Coverage Methods"""
import abc
from typing import Tuple, List

import numpy as np


def sum_score(profiles: np.ndarray) -> np.ndarray:
    """Reduces a boolean profile array to a sum of covered profile sections."""
    assert profiles.dtype == bool
    # Choose a dtype that can hold the maximum possible sum_score
    maxval = np.prod(profiles[0].shape)
    if maxval <= np.iinfo(np.int16).max:
        dtype = np.int16
    elif maxval <= np.iinfo(np.int32).max:
        dtype = np.int32
    else:
        dtype = np.int64
    # Calculate the sum scores as the simple sum of scores
    score = np.sum(profiles.reshape((profiles.shape[0], -1)), axis=1, dtype=dtype)
    assert np.all(score >= 0)
    return score


def flatten_layers(layers: List[np.ndarray]) -> np.ndarray:
    """Flatten a list of layers into a single array, for each sample."""
    layers = [np.reshape(layer, (layer.shape[0], -1)) for layer in layers]
    return np.concatenate(layers, axis=1)


class CoverageMethod(abc.ABC):
    """Abstract class to be used when implementing Coverage Criteria"""

    def __init__(self):
        """Setup of the coverage method.

        If, e.g., the coverage method needs to be fit according to some
         distribution, this is done in this init method.
        """
        pass

    @abc.abstractmethod
    def __call__(self, activations: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate coverages for a badge of activations

        The first dimension of the input and outputs reflects
        (scores and profiles) the number of samples in the badge.
        """
        pass


class NAC(CoverageMethod):
    """Neuron-Activation Coverage"""

    def __init__(self, cov_threshold: float):
        super().__init__()
        self.cov_threshold = cov_threshold

    def __call__(self, activations: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        activations = flatten_layers(activations)
        profiles = activations > self.cov_threshold
        return sum_score(profiles=profiles), profiles


class KMNC(CoverageMethod):
    """K-Multisection Neuron Coverage"""

    def __init__(self, mins: List[np.ndarray], maxs: List[np.ndarray], sections: int):
        super().__init__()
        self.sections = sections

        # Calculate the threshold defining the buckets
        min_arr = np.concatenate([l.flatten() for l in mins])
        max_arr = np.concatenate([l.flatten() for l in maxs])
        jumps = (max_arr - min_arr) / sections
        # assert np.all(jumps > 0) # For early convolutional layers, this is not true,
        #   the only consequence it that some bits will never be true, but
        #   this does not matter for the coverage calculation.
        self.thresh = [min_arr + jumps * i for i in range(sections + 1)]

    # docstr_coverage:inherited
    def __call__(self, activations: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        activations = flatten_layers(activations)

        # Calculate the shape of the profiles created by KMNC
        #   (badge_size x nodes x sections)
        profile_shape = (activations.shape[0], activations.shape[1], self.sections)

        profiles = np.zeros(shape=profile_shape, dtype=bool)
        for i in range(self.sections):
            profiles[..., i] = np.logical_and(
                self.thresh[i] <= activations, activations < self.thresh[i + 1]
            )
        return sum_score(profiles=profiles), profiles


class NBC(CoverageMethod):
    """Neuron Boundary Coverage"""

    def __init__(
        self,
        mins: List[np.ndarray],
        maxs: List[np.ndarray],
        stds: List[np.ndarray],
        scaler: float,
    ):
        super().__init__()

        min_arr = np.concatenate([l.flatten() for l in mins])
        max_arr = np.concatenate([l.flatten() for l in maxs])
        std_arr = np.concatenate([l.flatten() for l in stds])

        # Calculate the min and max boundaries
        self.min_boundaries = min_arr - scaler * std_arr
        self.max_boundaries = max_arr + scaler * std_arr

    # docs_coverage:inherited
    def __call__(self, activations: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        activations = flatten_layers(activations)

        # Calculate the shape of the profiles created by NBC
        #   (badge_size x nodes x sections)
        profile_shape = (activations.shape[0], activations.shape[1], 2)

        profiles = np.zeros(shape=profile_shape, dtype=bool)
        profiles[..., 0] = activations <= self.min_boundaries
        profiles[..., 1] = activations >= self.max_boundaries
        return sum_score(profiles=profiles), profiles


class SNAC(CoverageMethod):
    """Strong Neuron Activation Coverage"""

    def __init__(self, maxs: List[np.ndarray], stds: List[np.ndarray], scaler: float):
        super().__init__()
        max_arr = np.concatenate([l.flatten() for l in maxs])
        std_arr = np.concatenate([l.flatten() for l in stds])
        self.max_boundaries = max_arr + scaler * std_arr

    # docstr_coverage:inherited
    def __call__(self, activations: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        activations = flatten_layers(activations)
        profiles = activations >= self.max_boundaries
        return sum_score(profiles=profiles), profiles


class TKNC(CoverageMethod):
    """Top-k Neuron Coverage"""

    def __init__(self, top_neurons: int):
        super().__init__()
        self.top_neurons = top_neurons

    # docstr_coverage:inherited
    def __call__(self, activations: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        profiles = []
        for i, layer in enumerate(activations):
            layer = layer.reshape((layer.shape[0], -1))
            layer_top_nodes = np.argsort(layer, axis=1)[..., -self.top_neurons :]
            profile = np.zeros_like(layer, dtype=bool)
            np.put_along_axis(profile, layer_top_nodes, True, axis=1)
            profiles.append(profile)

        flattend_profiles = flatten_layers(profiles)

        # Sum score is the same for all inputs, but we still calculate it for simplicity
        return sum_score(profiles=flattend_profiles), flattend_profiles
