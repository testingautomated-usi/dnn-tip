"""Various variants of surprise adequacy."""

import abc
import gc
import logging
import math
import os
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Union, Callable, Iterable, Optional, Dict

import numpy as np
import psutil
import sklearn
import tqdm
from packaging import version
from psutil._common import bytes2human
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

# =============================================================================
# Custom types used as inputs for our implementations
# =============================================================================
from dnn_tip.stable_kde import StableGaussianKDE

Activations = Union[List[np.ndarray], np.ndarray]
"""Type Alias for activation traces. Shape: (samples x neurons). 

Typically a two dimensional float numpy array.
For convenience, we allow the user to pass in a higher dimensional array,
or a list of activations. In both cases, we flatten the input to the required
two dimensional format."""

Predictions = Union[List[Union[int, float]], np.ndarray]
"""Type Alias for predictions corresponding to a set of activation traces.

A one-dimensional array or a list of float or ints.
When used for classification (e.g. for DSA or per-class LSA/MSA),
values are expected to reflect the predicted classes, starting at index 0
(gained e.g. using `np.argmax(softmax_outputs, axis=1)`."""

Discriminator = Callable[[Activations, Predictions], np.ndarray]
"""Type Alias for an activation discriminator function.

The returned list is expected to have the same length as the passed
activations and predictions, stating for every entry an integer cluster id
in [0, 1, ..., n_clusters-1] that should be used to discriminate the samples."""


def _subsample_array(
    subsampling: Union[int, float], array: np.ndarray, seed: int
) -> np.ndarray:
    """Subsample a single array."""
    return _subsample_arrays(subsampling, (array,), seed=seed)[0]


def _subsample_arrays(
    subsampling: Union[int, float], arrays: Tuple[np.ndarray], seed: int
) -> Tuple[np.ndarray]:
    """Subsample multiple arrays using the sample sampling indexes for all."""

    array_lengths = arrays[0].shape[0]
    assert all(
        a.shape[0] == arrays[0].shape[0] for a in arrays
    ), "All arrays must have the same number of samples"

    if subsampling == 1.0:
        return arrays
    elif isinstance(subsampling, int) and subsampling > 0:
        num_samples = min(subsampling, array_lengths)
    elif 0 < subsampling < 1:
        num_samples = int(subsampling * array_lengths)
    else:
        raise ValueError(
            "subsampling must be a float between 0 and 1"
            " (share of training data),"
            "or a positive int declaring the number of samples"
        )
    rng = np.random.RandomState(seed)
    indexes = rng.choice(np.arange(array_lengths), num_samples, replace=False)
    sub_arrays: List[np.ndarray] = [a[indexes] for a in arrays]
    return tuple(sub_arrays)


def _by_class_discriminator(
    activations: Activations, predictions: Predictions
) -> np.ndarray:
    """Discriminator that assigns each sample to its class.

    This is useful for classification, where we want to see the
    class distribution of the data.
    """
    predictions = _class_predictions(predictions)
    return predictions


class _KmeansDiscriminator:
    def __init__(
        self,
        training_data: Activations,
        potential_k: Iterable[int],
        subsampling: Union[int, float] = 1.0,
        subsampling_seed: int = 0,
        n_init: int = 10,
        max_iter: int = 300,
    ):
        training_data = _flatten_layers(training_data)
        training_data = _subsample_array(
            subsampling, training_data, seed=subsampling_seed
        )

        self.best_score = -np.inf
        self.best_k = None
        self.best_clusterer = None

        for i in potential_k:
            kmeans = KMeans(n_clusters=i, n_init=n_init, max_iter=max_iter)
            cluster_labels = kmeans.fit_predict(training_data)
            silhouette_avg = silhouette_score(training_data, cluster_labels)
            if silhouette_avg > self.best_score:
                self.best_score = silhouette_avg
                self.best_k = i
                self.best_clusterer = kmeans

    def __call__(
        self, activations: Activations, predictions: Predictions
    ) -> np.ndarray:
        return self.best_clusterer.predict(_flatten_layers(activations))


def _class_predictions(predictions: Predictions, num_classes: int = None) -> np.ndarray:
    """Convert a list of class predictions to a one-dimensional array.

    Along the way, makes sure that the predictions are integers
    and, if provided, within the range [0, num_classes).
    """
    if isinstance(predictions, List):
        predictions = np.array(predictions)

    assert predictions.ndim == 1, (
        "Class predictions must be one-dimensional. "
        "If your predictions are one_hot encoded, use "
        "eg `np.argmax(softmax_outputs, axis=1)`"
    )

    if not predictions.dtype == int:
        np.testing.assert_almost_equal(
            predictions,
            predictions.astype(int),
            decimal=5,
            err_msg="Predictions must be integers",
        )
        predictions = predictions.astype(int)

    assert np.all(predictions >= 0), "Class predictions must be >= 0"
    assert num_classes is None or np.all(
        predictions < num_classes
    ), "Class predictions must be < num_classes"

    return predictions


def _flatten_layers(layers: Activations) -> np.ndarray:
    """Flatten a list of layers into a single array, for each sample."""
    if isinstance(layers, np.ndarray):
        if layers.ndim == 2:
            return layers
        else:
            return layers.reshape((layers.shape[0], -1))

    layers = [np.reshape(layer, (layer.shape[0], -1)) for layer in layers]
    return np.concatenate(layers, axis=1)


def _flatten_predictions(predictions: Predictions) -> Optional[np.ndarray]:
    if predictions is None:
        return None
    return predictions if isinstance(predictions, np.ndarray) else np.array(predictions)


class SurpriseCoverageMapper:
    """Calculates Coverage Profiles from an array of Suprise Adequacies"""

    def __init__(
        self, sections: int, upper_bound: float, overflow_bucket: bool = False
    ):
        self.sections = sections
        self.upper_bound = upper_bound
        linspace_sections = sections if overflow_bucket else sections + 1
        self.thresholds = np.linspace(
            start=0, stop=upper_bound, num=linspace_sections, dtype=np.float64
        )
        if overflow_bucket:
            self.thresholds = np.concatenate((self.thresholds, [np.inf]))

    def get_coverage_profile(self, surprise_values: np.ndarray) -> np.ndarray:
        """Maps the provided adequacy values to a coverage profile"""
        res = np.zeros(shape=(surprise_values.shape[0], self.sections), dtype=bool)
        for i in range(self.sections):
            res[..., i] = np.logical_and(
                self.thresholds[i] <= surprise_values,
                surprise_values < self.thresholds[i + 1],
            )
        return res


class SA(abc.ABC):
    """Abstract Superclass implemented by all DSA, LSA and MDSA"""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(
        self, activations: Activations, predictions: Predictions, num_threads: int = 1
    ) -> np.ndarray:
        """Calculates the Surprise Adequacy of the provided activations and predictions"""
        pass


class MultiModalSA(SA):
    """Discriminates samples to be used in different SA instances."""

    def __init__(self, discriminator: Discriminator, modal_sa: Dict[int, SA]):
        super().__init__()
        self.discriminator: Discriminator = discriminator
        self.modal_sa: Dict[int, SA] = modal_sa

    @staticmethod
    def build_by_class(
        activations: Activations,
        predictions: Predictions,
        sa_constructor: Callable[[Activations, Predictions], SA],
    ) -> "MultiModalSA":
        """A multi-modal SA that discriminates by the predicted class"""
        return MultiModalSA.build(
            activations=activations,
            predictions=predictions,
            discriminator=_by_class_discriminator,
            sa_constructor=sa_constructor,
        )

    @staticmethod
    def build_with_kmeans(
        activations: Activations,
        predictions: Optional[Predictions],
        sa_constructor: Callable[[Activations, Predictions], SA],
        potential_k: Iterable[int],
        n_init: int = 10,
        max_iter: int = 300,
        subsampling: Union[int, float] = 1.0,
        subsampling_seed: int = 0,
    ) -> "MultiModalSA":
        """A multi-modal SA that discriminates by k-means clustering (e.g., MMDSA)."""
        discriminator = _KmeansDiscriminator(
            training_data=activations,
            potential_k=potential_k,
            n_init=n_init,
            max_iter=max_iter,
            subsampling=subsampling,
            subsampling_seed=subsampling_seed,
        )
        return MultiModalSA.build(
            activations=activations,
            predictions=predictions,
            discriminator=discriminator,
            sa_constructor=sa_constructor,
        )

    @staticmethod
    def build(
        activations: Activations,
        predictions: Optional[Predictions],
        discriminator: Discriminator,
        sa_constructor: Callable[[Activations, Predictions], SA],
    ) -> "MultiModalSA":
        """Utility method to create a new instance for given training activations"""
        activations = _flatten_layers(activations)
        predictions = _flatten_predictions(predictions)

        modal_indexes: np.ndarray = discriminator(activations, predictions)
        sa_s: Dict[int, SA] = dict()
        for modal_id in np.unique(modal_indexes):
            modal_activations = activations[modal_indexes == modal_id]
            if predictions is not None:
                modal_predictions = predictions[modal_indexes == modal_id]
            else:
                modal_predictions = None
            sa_s[modal_id] = sa_constructor(modal_activations, modal_predictions)

        return MultiModalSA(discriminator=discriminator, modal_sa=sa_s)

    @staticmethod
    def _calculate_surprise_for_modal(
        sa: SA,
        activations: Activations,
        predictions: Optional[Predictions],
        num_threads: int,
    ) -> np.ndarray:
        """Returns the SA instance for the provided modal id"""
        return sa(activations, predictions, num_threads=num_threads)

    def _get_sa_for_modal_id(self, modal_id: int) -> SA:
        """Gets a SA instance with given model id, with reasonable error if not existing."""
        try:
            return self.modal_sa[modal_id]
        except KeyError:
            raise ValueError(
                f"No modal found for modal id {modal_id}. Check your discriminator"
            )

    def __call__(
        self,
        activations: Activations,
        predictions: Optional[Predictions],
        num_threads: int = 1,
    ) -> np.ndarray:
        discriminator_idxs = self.discriminator(activations, predictions)

        activations = _flatten_layers(activations)
        predictions = _flatten_predictions(predictions)

        # Check that the returned modal indexes are valid
        assert len(discriminator_idxs) == activations.shape[0], (
            f"The discriminator returned an invalid number "
            f"({len(discriminator_idxs)}) of modal indexes."
            f"Expected: {activations.shape[0]} indexes."
        )

        if len(discriminator_idxs) == 0:
            return np.ndarray(shape=(0,))

        modals_in_this_set = np.unique(discriminator_idxs)
        futures = []
        # Note; the way we distribute workers amonst sub-sets is not optimal,
        #  (workers freed up by a multi-threaded child thread are not passed on to other tasks)
        #  but it is it is an easy way given the hierachical structure of our SA instances.
        workers = min(num_threads, len(modals_in_this_set))
        remaining_workers = num_threads - workers
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for modal_id in modals_in_this_set:
                sa = self._get_sa_for_modal_id(modal_id)
                a = activations[discriminator_idxs == modal_id]
                p = (
                    None
                    if predictions is None
                    else predictions[discriminator_idxs == modal_id]
                )
                workers = int(np.ceil(remaining_workers / (len(modals_in_this_set))))
                remaining_workers -= workers
                f = executor.submit(
                    self._calculate_surprise_for_modal, sa, a, p, workers
                )
                futures.append(f)

        per_modal_values = [f.result() for f in futures]
        res = np.full(
            fill_value=-np.inf,
            shape=discriminator_idxs.shape,
            dtype=per_modal_values[0].dtype,
        )

        for i, adequacies in enumerate(per_modal_values):
            res[discriminator_idxs == modals_in_this_set[i]] = adequacies

        return res


class MDSA(SA):
    """Mahalanobis-based Surprise Adequacy"""

    def __init__(self, activations: Activations):
        super().__init__()
        activations = _flatten_layers(activations)
        self.covariance_matrix: EmpiricalCovariance = (
            sklearn.covariance.EmpiricalCovariance()
        )
        self.covariance_matrix.fit(activations)

    def __call__(
        self,
        activations: Activations,
        predictions: Predictions = None,
        num_threads: int = None,
    ) -> np.ndarray:
        activations = _flatten_layers(activations)
        distances = self.covariance_matrix.mahalanobis(activations)
        return distances


class LSA(SA):
    """Likelihood-based Surprise Adequacy"""

    def __init__(
        self,
        activations: Activations,
        # Threshold and max_features used for performance and to increase chance that ats matrix
        #  is not singular, which would dissalow KDE
        var_threshold: Optional[float] = None,
        max_features: Optional[Union[int, float]] = 300,
    ):
        super().__init__()

        activations = _flatten_layers(activations)

        assert var_threshold is None or max_features is None, (
            "Both var_threshold and max_features cannot be specified at the same time."
            "We recommend using the max_features arg to dynamically keep the features"
            "with the highest variance."
        )

        self.removed_neurons: List[int]
        if var_threshold is not None and var_threshold > 0:
            activations = _flatten_layers(activations)
            self.removed_neurons = np.where(
                np.var(activations, axis=0) < var_threshold
            )[0]

        # Figure out which neurons to remove, as they have low variance
        if max_features is not None:
            if max_features < 1:
                num_features = min(
                    max_features * activations.shape[1], activations.shape[1]
                )
            else:
                num_features = min(max_features, activations.shape[1])

            dropped_columns = np.argsort(np.var(activations, axis=0))[:-num_features]
            self.removed_neurons = list(int(x) for x in dropped_columns)

        # Fit the density model
        self.kde = self._create_gaussian_kde(activations)
        print(f"Done creating KDE")

    def _create_gaussian_kde(self, activations: np.ndarray) -> gaussian_kde:
        cleaned_activations = self._remove_unused_columns(activations)
        if cleaned_activations.shape[1] == 0:
            warnings.warn(
                (
                    f"The passed min_var threshold {self.min_var_threshold} and/or"
                    f"the automatic removal of numerically unstable features"
                    f"led to the removal of all ATs. This instance of LSA will"
                    f"thus always return density 0"
                ),
                UserWarning,
            )
            self.kde = None
        else:
            try:
                return StableGaussianKDE(cleaned_activations.transpose())
            except (np.linalg.LinAlgError, ValueError) as e:
                if "-th leading minor of the array is not positive definite" in str(
                    e
                ) or "numerical imprecision in covariance matrix" in str(e):
                    problematic_row = int(re.findall("\\d*", str(e))[0]) - 1
                    original_indexes = np.delete(
                        np.arange(activations.shape[1]), self.removed_neurons
                    )
                    problematic_index = original_indexes[problematic_row]

                    warnings.warn(
                        f"Dropping AT {problematic_index}, as leading to numerical error.",
                        UserWarning,
                        1,
                    )

                    self.removed_neurons.append(problematic_index)
                    return self._create_gaussian_kde(activations)
                else:
                    warnings.warn(f"Problem regarding KDE fitting", UserWarning)
                    raise e

    def _remove_unused_columns(self, tr_activations):
        if self.removed_neurons is not None and len(self.removed_neurons) > 0:
            return np.delete(tr_activations, self.removed_neurons, axis=1)
        return tr_activations

    def __call__(
        self,
        activations: Activations,
        predictions: Predictions = None,  # ignored in LSA
        num_threads: int = 0,  # Ignored in LSA
    ) -> np.ndarray:
        activations = _flatten_layers(activations)
        activations = self._remove_unused_columns(activations)
        if self.kde is None:
            return np.zeros(shape=(activations.shape[0],))

        density = self.kde.evaluate(activations.transpose())
        return -np.log(density)


class MLSA(SA):
    """Multimodal Likelihood-Based Surprise Adequacy"""

    def __init__(self, activations: Activations, num_components: int = 2):
        super().__init__()

        activations = _flatten_layers(activations)
        logging.info(
            f"Fitting Gaussian Mixture with {num_components} components for MLSA"
        )
        self.gmm = GaussianMixture(n_components=num_components)
        self.gmm.fit(activations)

    def __call__(
        self,
        activations: Activations,
        predictions: Predictions = None,  # ignored in LSA
        num_threads: int = 0,  # Ignored in LSA
    ) -> np.ndarray:
        activations = _flatten_layers(activations)
        log_likelihood = self.gmm.score_samples(activations)
        # Like LSA, MLSA is defined as negative log likelihood
        return -log_likelihood


class DSA(SA):
    """Distance-based Surprise Adequacy

    Our implementation of DSA is based on the paper:
    `Weiss et. al., A Review and Refinement of Surprise Adequacy, ICSE-W 2021`
    """

    def __init__(
        self,
        activations: Activations,
        predictions: Predictions,
        badge_size: int = 10,
        subsampling: Union[int, float] = 1.0,
        subsampling_seed: int = 0,
    ):
        super().__init__()
        self.train_activations: np.ndarray = _flatten_layers(activations)
        self.train_predictions: np.ndarray = _class_predictions(predictions)

        self.train_activations, self.train_predictions = _subsample_arrays(
            subsampling,
            (self.train_activations, self.train_predictions),
            subsampling_seed,
        )
        self.num_classes = np.max(self.train_predictions) + 1
        self.class_matrix = self._class_matrix()
        self.badge_size = badge_size

    def _class_matrix(self) -> List[List[int]]:
        class_matrix = []
        for label in range(self.num_classes):
            indexes = np.argwhere(self.train_predictions == label)
            class_matrix.append(indexes.flatten())
        return class_matrix

    def __call__(
        self,
        activations: Activations,
        predictions: Predictions,
        num_threads: int = None,
    ) -> np.ndarray:

        target_pred = _class_predictions(predictions)
        target_ats = _flatten_layers(activations)

        del activations, predictions

        self._check_memory_warning(
            test_samples=target_ats.shape[0],
            size_per_element=target_ats.dtype.itemsize,
            num_threads=num_threads,
        )

        dsa = np.empty(shape=target_pred.shape[0])

        # Split work into badges (each badge consisting of indexes with a common label)
        task_inputs = []
        for label in range(self.num_classes):
            l_indexes = np.argwhere(target_pred == label).flatten()
            if l_indexes.size == 0:
                continue
            num_splits = math.ceil((l_indexes.shape[0] / self.badge_size))
            l_index_badges = np.array_split(l_indexes, num_splits)
            for batch in l_index_badges:
                task_inputs.append((batch, label))

        # A task to compute the dsa for a badge
        #  Will be executed in parallel
        def task(idxs, t_label):
            """A single, atomic task to be run in a separate thread."""
            t_samples = target_ats[idxs]
            a_min_dist, b_min_dist = self._dsa_distances(t_samples, t_label)
            t_task_dsa = a_min_dist / b_min_dist
            return idxs, t_task_dsa

        # Execute the tasks
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Start the load operations and mark each future with its URL
            future_to_url = {
                executor.submit(task, params[0], params[1]): params
                for params in task_inputs
            }
            for future in tqdm.tqdm(
                as_completed(future_to_url),
                total=len(future_to_url),
                desc="Calculating DSA",
            ):
                f_idxs, f_task_dsa = future.result()
                dsa[f_idxs] = f_task_dsa

        return dsa

    def _dsa_distances(
        self, sample_ats: np.ndarray, label: int
    ) -> Tuple[np.ndarray, np.ndarray]:

        train_matches_same_class = self.train_activations[self.class_matrix[label]]
        ats_a, dist_a = self._get_closest_ats(sample_ats, train_matches_same_class)

        other_classes_indexes = np.ones(
            shape=self.train_activations.shape[0], dtype=bool
        )
        other_classes_indexes[self.class_matrix[label]] = 0
        train_matches_other_classes = self.train_activations[other_classes_indexes]
        dist_b = self._get_closest_ats(
            ats_a, train_matches_other_classes, calc_ats=False
        )

        return dist_a, dist_b

    @staticmethod
    def _get_closest_ats(from_ats, to_ats, calc_ats=True):
        # Expand the from_ats array to allow comparison of each element
        #   with each element of to_ats, and calculate the differences
        #   for each activation
        at_distances = from_ats[:, None] - to_ats
        # Calculate the norms of the differences
        distance_norms = np.linalg.norm(at_distances, axis=2)
        # Release immediately, as huge
        del at_distances
        gc.collect()
        # Find the minimum distance for each element of from_ats
        distances = np.min(distance_norms, axis=1)
        if calc_ats:
            closest_position = np.argmin(distance_norms, axis=1)
            closest_ats = to_ats[closest_position]
            return closest_ats, distances
        else:
            return distances

    def _check_memory_warning(
        self, test_samples: int, size_per_element, num_threads: int = None
    ):

        # Figure out how many threads will be used
        if num_threads is None:
            cpu_count = os.cpu_count()
            if cpu_count is None:
                warnings.warn(
                    "Could not determine the number of CPUs. ,"
                    "Hence, cannot estimate DSA memory use",
                    RuntimeWarning,
                )
                return
            if version.Version(
                f"{sys.version_info.major}.{sys.version_info.minor}"
            ) >= version.Version("3.8"):
                num_threads = min(32, cpu_count + 4)
            else:
                num_threads = cpu_count * 5

        peak_concurrent_tasks = min(
            num_threads, math.ceil(test_samples / self.badge_size)
        )

        # Figure out how much memory will be used in peak loads
        required_memory = (
            # The float precision of the used data structure
            size_per_element
            # The number of concurrently working threads
            * peak_concurrent_tasks
            # The number of test samples handled per thread
            * self.badge_size
            # The number of training samples to compare against
            * self.train_activations.shape[0]
            # The size of a simple training sample
            * self.train_activations[0].size
        )
        _memory_warning(required_memory)


def _memory_warning(required_memory: int):
    available_memory = psutil.virtual_memory().available
    if available_memory is not None and required_memory > available_memory / 2:
        human_readable = bytes2human(required_memory)
        warnings.warn(
            f"Expected peak memory use for DSA is higher than {human_readable}, "
            f"which exceeds 50% available memory.",
            UserWarning,
            2,
        )
