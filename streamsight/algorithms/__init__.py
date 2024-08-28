"""
The algorithms module in streamsight contains a wide array of state-of-the-art
collaborative filtering algorithms.
Also included are some baseline algorithms, as well as several reusable building blocks
such as commonly used loss functions and sampling methods.

.. currentmodule:: streamsight.algorithms

Algorithm
---------
Base class for all algorithms. Programmer should inherit from this class when implementing
a new algorithm.

.. autosummary::
    :toctree: generated/

    Algorithm
    
Baseline Algorithms
-------------------

.. autosummary::
    :toctree: generated/

    Random
    Popularity

Item Similarity Algorithms
----------------------------

Item similarity algorithms exploit relationships between items to make recommendations.
At prediction time, the user is represented by the items
they have interacted with.

.. autosummary::
    :toctree: generated/

    ItemKNN
    ItemKNNIncremental
    ItemKNNRolling
    ItemKNNStatic
"""
from streamsight.algorithms.base import Algorithm
from streamsight.algorithms.random import Random
from streamsight.algorithms.popularity import Popularity
from streamsight.algorithms.itemknn import ItemKNN
from streamsight.algorithms.itemknn_incremental import ItemKNNIncremental
from streamsight.algorithms.itemknn_rolling import ItemKNNRolling
from streamsight.algorithms.itemknn_static import ItemKNNStatic