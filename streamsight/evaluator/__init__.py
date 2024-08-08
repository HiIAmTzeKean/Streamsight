"""
The algorithms module in streamsight contains a wide array of state-of-the-art
collaborative filtering algorithms.
Also included are some baseline algorithms, as well as several reusable building blocks
such as commonly used loss functions and sampling methods.

.. currentmodule:: streamsight.algorithms

Item Similarity Algorithms
----------------------------

Item similarity algorithms exploit relationships between items to make recommendations.
At prediction time, the user is represented by the items
they have interacted with.

.. autosummary::
    :toctree: generated/

    EvaluatorBuilder
"""
from streamsight.evaluator.evaluator_builder import EvaluatorBuilder
from streamsight.evaluator.evaluator import Evaluator
from streamsight.evaluator.util import MetricAccumulator, MetricLevelEnum