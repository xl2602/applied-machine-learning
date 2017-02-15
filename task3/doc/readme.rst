README for task3
===================

Description
----------------

This function that takes two inputs:  a dataset represented as a numpy-array, and an “axis” argument. The function computes mean and standard deviation of a dataset along the specified “axis”, which can be 0, 1, or None Mean and standard deviation are returned.


Examples
----------------

.. math::

    A = \begin{bmatrix}
    1 & 2 \\
    3 & 4 \\
    \end{bmatrix}

* Mean and Standard Deviation of Matrix Columns

.. code-block:: python


    M, S = task3(A, axis=0)


.. math::

	M = \begin{bmatrix}
    2 & 3 \\
    \end{bmatrix}


   	S = \begin{bmatrix}
    1 & 1 \\
    \end{bmatrix}

* Mean and Standard Deviation of Matrix Rows

.. code-block:: python

    M, S = task3(A, axis=1)


.. math::

	M = \begin{bmatrix}
    1.5 & 3.5 \\
    \end{bmatrix}


   	S = \begin{bmatrix}
    0.5 & 0.5 \\
    \end{bmatrix}


* Mean and Standard Deviation of Flattened Matrix

.. code-block:: python

    M, S = task3(A, axis=None)

.. math::

	M = 2.5


   	S = 1.11803398875

