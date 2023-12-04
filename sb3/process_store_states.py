import numpy as np
from typing import Union

def safe_std(arr: Union[np.ndarray, list]) -> np.ndarray:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr: Numpy array or list of values
    :return:
    """
    return np.nan if len(arr) == 0 else np.std(arr)


def safe_sum(arr: Union[np.ndarray, list]) -> np.ndarray:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr: Numpy array or list of values
    :return:
    """
    return np.nan if len(arr) == 0 else np.sum(arr)

