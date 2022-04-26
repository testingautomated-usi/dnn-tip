import numpy as np
import pytest

from dnn_tip.surprise import (
    DSA,
    LSA,
    MDSA,
    MLSA,
    MultiModalSA,
    SurpriseCoverageMapper,
    _by_class_discriminator,
    _class_predictions,
    _flatten_predictions,
    _KmeansDiscriminator,
)


@pytest.mark.parametrize(
    "activations, predictions",
    [
        ([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [0, 1]),
        ([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.4, 0.5, 0.6]], [0, 1, 1]),
        ([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.4, 0.5, 0.6]], [0, 1, 1]),
    ],
)
def test__by_class_discriminator(activations, predictions):
    activations, predictions = np.array(activations), np.array(predictions)
    modal_ids = _by_class_discriminator(activations, predictions)
    assert modal_ids.shape == predictions.shape
    assert np.all(modal_ids == np.array(predictions))


@pytest.mark.parametrize(
    "predictions, num_classes, message",
    [
        ([0.5, 0.5], 2, "Predictions must be integers"),
        ([-1, 5, 7], 2, "Class predictions must be >= 0"),
        ([0, 2, 6], 6, "must be < num_classes"),
        ([[0, 0, 0, 1]], 2, "must be one-dimensional"),
    ],
)
def test__by_class_predictions_assertions(predictions, num_classes, message):
    with pytest.raises(AssertionError) as e:
        _class_predictions(predictions, num_classes=num_classes)
    assert message in str(e.value)


@pytest.mark.parametrize(
    "method_input, expected",
    [
        (np.array([0, 2, 3, 5, 0.1, -5]), np.array([0, 2, 3, 5, 0.1, -5])),
        ([0, 2, 3, 5, 0.1, -5], np.array([0, 2, 3, 5, 0.1, -5])),
    ],
)
def test__flatten_predictions(method_input, expected: np.ndarray):
    assert np.all(expected == _flatten_predictions(method_input))


@pytest.mark.parametrize(
    "buckets, limit, overflow, sa, expected",
    [
        (
            3,
            1,
            False,
            np.array([0.1, 0.2, 0.8]),
            np.array(
                [[True, False, False], [True, False, False], [False, False, True]]
            ),
        ),
        (
            3,
            1,
            True,
            np.array([0.1, 0.2, 0.8]),
            np.array(
                [[True, False, False], [True, False, False], [False, True, False]]
            ),
        ),
        (
            3,
            1,
            True,
            np.array([0.1, 0.2, 1.1]),
            np.array(
                [[True, False, False], [True, False, False], [False, False, True]]
            ),
        ),
    ],
)
def test_surprise_coverage_mapper(buckets, limit, overflow, sa, expected):
    profile = SurpriseCoverageMapper(buckets, limit, overflow).get_coverage_profile(sa)
    assert profile.shape == expected.shape
    assert np.all(profile == expected)


def test_multi_modal_sa():
    rng = np.random.RandomState(42)
    activations = rng.random((10000, 10))
    labels = rng.randint(0, 3, size=10000)
    sa = MultiModalSA.build_by_class(activations, labels, lambda x, y: LSA(x))
    assert sa.modal_sa.keys() == {0, 1, 2}
    assert sa.modal_sa[0].__class__ == LSA

    test_activations = rng.random((1000, 10))
    test_labels = rng.randint(0, 3, size=1000)
    test_surprises = sa(test_activations, test_labels)
    # The default initialization of the returned array is -inf
    #   Hence remaining -inf would show that the result array
    #   is not correctly assembled
    assert test_surprises.shape == (1000,)
    assert np.sum(test_surprises == -np.inf) == 0
    for label in range(3):
        class_surp = test_surprises[test_labels == label]
        this_label_lsa = sa.modal_sa[label]
        label_surprises = this_label_lsa(
            test_activations[test_labels == label], test_labels[test_labels == label]
        )
        assert np.all(class_surp == label_surprises)


def test_mdsa_covariance():
    rng = np.random.RandomState(42)
    activations = rng.random((100000, 10))
    cov = np.cov(np.copy(activations).T)
    mdsa = MDSA(activations)

    # Make sure the estimate is at most 10% of the true covariance
    #   This makes sure we use the scipy function correctly
    np.testing.assert_allclose(mdsa.covariance_matrix.covariance_, cov, 0.1)


@pytest.mark.parametrize(
    "class_creator, strictly_positive",
    [
        pytest.param(lambda x, y: MDSA(x), True, id="MDSA"),
        pytest.param(lambda x, y: LSA(x), False, id="LSA"),
        pytest.param(lambda x, y: DSA(x, y), False, id="DSA"),
    ],
)
def test_sa_plausibility(class_creator, strictly_positive):
    rng = np.random.RandomState(42)
    activations = rng.random((100, 10))
    labels = rng.randint(0, 3, size=100)
    sa = class_creator(activations, labels)

    # The actual MD calculation is just a scipy call.
    # We check for regression using a simple metamorphic test:
    #   We know that on in-distribution samples, the LSA is lower than
    #   on out-of-distribution samples.
    id_sa = sa(activations[:10], labels[:10])
    ood_sa = sa(activations[:10] + 10, labels[:10])

    # ID should have lower surprise than OOD
    assert np.all(ood_sa > id_sa)

    if strictly_positive:
        assert np.all(id_sa >= 0)
        assert np.all(ood_sa >= 0)

    # Check shape
    assert id_sa.shape == ood_sa.shape == (10,)

    # Outcome should be deterministic (testing on badge)
    badge = np.concatenate([activations for _ in range(2)])
    badge_labels = np.concatenate([labels for _ in range(2)])
    badge_sa = sa(badge, badge_labels).reshape((2, -1))
    np.testing.assert_array_almost_equal(badge_sa[1], badge_sa[0])
    # Outcome should be deterministic (testing two calls)
    same_badge_sa = sa(badge, badge_labels).reshape((2, -1))
    np.testing.assert_array_almost_equal(same_badge_sa, badge_sa)


def test_mlsa_plausability():
    rng = np.random.RandomState(42)
    activations_1 = rng.random((10000, 10))
    activations_2 = rng.random((10000, 10)) + 0.4
    activations_3 = rng.random((10000, 10)) + 0.9
    activations = np.concatenate([activations_1, activations_2, activations_3])
    mlsa = MLSA(activations, num_components=3)
    test_activations = np.array(
        [
            [0.5] * 10,
            [0.9] * 10,
            [1.4] * 10,
        ]
    )

    # we do not know the cluster numbers, but we know all three points
    #   are at the centers of the three distinct clusters
    id_clusters = mlsa.gmm.predict(test_activations)
    assert len(set(id_clusters)) == 3

    # We know that points at the centers of the distributions should have lower
    #   surprise than points which are a bit away from the centers
    ood_data = test_activations + 2
    id_surprises = mlsa(test_activations)
    ood_surprises = mlsa(ood_data)
    assert np.all(ood_surprises > id_surprises)


def test_k_means_clusterer_and_mmdsa():
    rng = np.random.RandomState(42)
    activations_1 = rng.random((100, 10))
    activations_3 = rng.random((100, 10)) + 0.9
    activations = np.concatenate([activations_1, activations_3])
    test_activations = np.array(
        [
            [0.5] * 10,
            [1.4] * 10,
        ]
    )

    # we do not know the cluster numbers, but we know all three points
    #   are at the centers of the three distinct clusters
    discriminator = _KmeansDiscriminator(activations, [2, 3, 4])
    assert discriminator.best_k == 2
    id_clusters = discriminator(test_activations, None)
    assert len(set(id_clusters)) == 2

    # We know that points at the centers of the distributions should have lower
    #   surprise than points which are a bit away from the centers
    ood_data = test_activations + 2
    mmdsa = MultiModalSA.build_with_kmeans(
        activations, None, lambda x, _: MDSA(x), potential_k=[2, 3, 4]
    )
    id_surprises = mmdsa(test_activations, None)
    ood_surprises = mmdsa(ood_data, None)
    assert np.all(ood_surprises > id_surprises)
