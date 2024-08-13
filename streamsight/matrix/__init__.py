"""
Matrix
----------
This module provides the functionality to create interaction matrix from the stream data.
The interaction matrix can be used to build recommendation systems.

.. currentmodule:: streamsight.matrix

Interaction Matrix
-------------------
The InteractionMatrix class is used to create an interaction matrix from the stream data.
The interaction matrix can be used to build recommendation systems.

.. autosummary::
    :toctree: generated/

    InteractionMatrix

Utils
-----
Below are some of the utility classes and functions that are used in the matrix module.
These provides functionality for the matrix module and for use cases in other modules.

.. autosummary::
    :toctree: generated/

    to_csr_matrix
    ItemUserBasedEnum
    TimestampAttributeMissingError
"""
from streamsight.matrix.interaction_matrix import InteractionMatrix, ItemUserBasedEnum
from streamsight.matrix.util import to_csr_matrix
from streamsight.matrix.exception import TimestampAttributeMissingError