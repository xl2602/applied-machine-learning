import numpy as np


def task3(dataset, axis=0):
    """Computes mean and standard deviation of a numpy array .  

    This function that takes two inputs: 
    a dataset represented as a numpy-array, and an “axis” argument.
     The function computes mean and standard deviation of a dataset along the specified “axis”, which can be 0, 1, or None
    Mean and standard deviation are returned.

    Parameters
    ----------
    dataset : numpy array
        A numpy array.

    axis : 0 or 1 or None, optional
        Default value is 0 which is to compute the mean and standard deviation of each column.
        If value=1, compute the mean and standard deviation of each row.
        If axis=None, compute results of the flattened array.

    Returns
    -------
    tuple :
        Return a tuple of two numpy objects. First is mean of dataset, and second is standard deviation of dataset.
    """

    if axis == 0:
        return np.mean(dataset, axis=0), np.std(dataset, axis=0)
    elif axis == 1:
        return np.mean(dataset, axis=1), np.std(dataset, axis=1)
    elif axis is None:
        return np.mean(dataset), np.std(dataset)
    else:
        raise ValueError('invalid axis value')











