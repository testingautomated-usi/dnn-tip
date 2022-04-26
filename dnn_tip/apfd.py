"""An implementation of APFD, as described by Feng. et. al. (DeepGini)"""

from typing import List, Union

import numpy as np


def apfd_from_order(is_fault, index_order: Union[List[int], np.ndarray]) -> float:
    """
    Compute APFD from the index order of the misclassified samples.
    """
    assert is_fault.ndim == 1, "at the moment, only unique faults are supported"
    ordered_faults = is_fault[index_order]
    fault_indexes = np.where(ordered_faults == 1)[0]
    k = np.count_nonzero(is_fault)
    n = is_fault.shape[0]
    # The +1 comes from the fact that the first sample has index 0 but order 1
    sum_of_fault_orders = np.sum(fault_indexes + 1)
    return 1 - (sum_of_fault_orders / (k * n)) + (1 / (2 * n))
