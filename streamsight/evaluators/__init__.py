"""
.. currentmodule:: streamsight.evaluators

Evaluator Builder
----------------------------

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

The adding of new algorithms through :meth:`add_algorithm` and metrics through :meth:`add_metric`
are made such that it can be done through the class type via importing the class or 
thorough specifying the class name as a string.

.. autosummary::
    :toctree: generated/

    EvaluatorBuilder
    EvaluatorStreamerBuilder

Example
~~~~~~~~~

Below is a typical example of how to use the :class:`EvaluatorBuilder` to build an
:class:`Evaluator` object. This example also follows from the example in the
python notebook.

.. code-block:: python

    from streamsight.evaluator import EvaluatorBuilder
    
    builder = EvaluatorBuilder(item_user_based="item",
                        ignore_unknown_user=True,
                        ignore_unknown_item=True)
    builder.add_setting(setting)
    builder.add_algorithm("ItemKNNIncremental", {"K": 10})
    builder.add_metric("PrecisionK")
    evaluator = builder.build()

    evaluator.run()

Evaluator
----------------------------

The evaluator module in streamsight contains the evaluators which allows the
programmer to evaluate the algorithms on a fixed setting. The evaluator also
aids the programmer in masking the shape of the dataset and then subsequently
computing the metric for each prediction against the ground truth. There are
2 concrete implementation of :class:`Evaluator`.

.. autosummary::
    :toctree: generated/

    EvaluatorBase
    EvaluatorPipeline
    EvaluatorStreamer


Accumulator
----------------------------

The evaluator module in streamsight contains the Accumulator class which is
used to accumulate the metrics.

.. autosummary::
    :toctree: generated/

    MetricAccumulator

    
Utility
----------------------------

The evaluator module in streamsight contains the utility classes which abstract
some of the common functionalities used in the evaluator module.

.. autosummary::
    :toctree: generated/

    MetricLevelEnum
    UserItemBaseStatus
    AlgorithmStatusWarning
"""

from streamsight.evaluators.base import EvaluatorBase
from streamsight.evaluators.evaluator_builder import EvaluatorBuilder, EvaluatorStreamerBuilder
from streamsight.evaluators.evaluator_pipeline import EvaluatorPipeline
from streamsight.evaluators.evaluator_stream import EvaluatorStreamer
from streamsight.evaluators.util import MetricLevelEnum, UserItemBaseStatus, AlgorithmStatusWarning
from streamsight.evaluators.accumulator import (
    MetricAccumulator
)
