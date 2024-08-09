"""
Evaluator Builder
----------------------------

.. currentmodule:: streamsight.evaluator.evaluator_builder

The evaluator module in streamsight contains the EvaluatorBuilder class which is
used to build an evaluator object. :class:`EvaluatorBuilder` allows the programmer
to add algorithms, metrics and settings to the evaluator object before the
:class:`Evaluator` object is built.

It is recommended to initialise the :class:`Evaluator` object with the :class:`EvaluatorBuilder`
as it provides the API for the needed configurations. Beyond the API for the configurations,
the builder checks for the validity of the configurations and raises exceptions if the
configurations are invalid. The programmer can choose to build the :class:`Evaluator` object
without the :class:`EvaluatorBuilder` as described below but might face exceptions if the
configurations are invalid.

.. autosummary::
    :toctree: generated/

    EvaluatorBuilder
    
Evaluator
----------------------------

.. currentmodule:: streamsight.evaluator.evaluator

The evaluator module in streamsight contains the Evaluator class which is
used to evaluate the performance of the algorithms on the data.

.. autosummary::
    :toctree: generated/

    Evaluator

Accumulator
----------------------------

.. currentmodule:: streamsight.evaluator.accumulator

The evaluator module in streamsight contains the Accumulator class which is
used to accumulate the metrics.

.. autosummary::
    :toctree: generated/

    MetricAccumulator
    MacroMetricAccumulator
    MicroMetricAccumulator
"""
from streamsight.evaluator.evaluator_builder import EvaluatorBuilder
from streamsight.evaluator.evaluator import Evaluator
from streamsight.evaluator.util import MetricLevelEnum
from streamsight.evaluator.accumulator import MetricAccumulator, MacroMetricAccumulator, MicroMetricAccumulator