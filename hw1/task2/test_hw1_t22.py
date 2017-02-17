# Task 2.2

import numpy as np


def test_built_in():
    assert float(2)/float(8) == 0.25


def test_np_arr():
    assert np.divide(np.array([2], dtype='f'), np.array([8], dtype='f')) == 0.25


