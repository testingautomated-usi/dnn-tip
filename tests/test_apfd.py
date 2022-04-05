import numpy as np
import pytest

from dnn_tip.apfd import apfd_from_order


@pytest.mark.parametrize(
    "order, fault, expected",
    [
        ([0, 1, 2], np.array([True, True, True]), (1 - 6 / 9 + 1 / 6)),
        ([0, 1, 2], np.array([True, False, False]), (1 - 1 / 3 + 1 / 6)),
        ([0, 1, 2], np.array([False, False, True]), (1 - 3 / 3 + 1 / 6)),
        ([2, 1, 0], np.array([False, False, True]), (1 - 1 / 3 + 1 / 6)),
        ([2, 1, 0], np.array([True, False, False]), (1 - 3 / 3 + 1 / 6)),
    ],
)
def test_apfd_sanity(order, fault, expected):
    assert apfd_from_order(fault, order) == expected
