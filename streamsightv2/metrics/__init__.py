"""
.. currentmodule:: streamsight.metrics

Metrics
----------

This module provides the functionality to evaluate the performance of the recommendation system.

.. autosummary::
    :toctree: generated/

    Metric
    PrecisionK
    RecallK
"""

from streamsight2.metrics.base import Metric
from streamsight2.metrics.precision import PrecisionK
from streamsight2.metrics.recall import RecallK
from streamsight2.metrics.dcg import DCGK
from streamsight2.metrics.ndcg import NDCGK
from streamsight2.metrics.hit import HitK
