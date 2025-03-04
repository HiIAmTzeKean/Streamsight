"""
.. currentmodule:: streamsightv2.metrics

Metrics
----------

This module provides the functionality to evaluate the performance of the recommendation system.

.. autosummary::
    :toctree: generated/

    Metric
    PrecisionK
    RecallK
    DCGK
    NDCGK
    HitK
"""

from streamsightv2.metrics.base import Metric
from streamsightv2.metrics.precision import PrecisionK
from streamsightv2.metrics.recall import RecallK
from streamsightv2.metrics.dcg import DCGK
from streamsightv2.metrics.ndcg import NDCGK
from streamsightv2.metrics.hit import HitK
