Welcome to Streamsight's documentation!
=======================================

.. image:: /_static/logo_no_text.png
    :align: center

Streamsight is an open-source python toolkit developed that provides a framework
which observes the context of time to accurately model offline setting to actual
real-world scenarios. We aim to provide API for the programmer to build and
evaluate recommendation systems. **Read our report:**
`Streamsight: a toolkit for offline evaluation of recommender systems <https://hdl.handle.net/10356/181114>`_

The overall architecture of the package is shown in the figure below. We split
the toolkit into three main components: data handling, recommendation system,
and evaluation. The data handling component is responsible for loading and
preprocessing the data, the RecSys on implementing the recommendation algorithms
and the Evaluation for evaluating the recommendation algorithms.

.. image:: /_static/architecture.png
    :align: center

The demo notebooks can be found in the `examples` directory
`here <https://github.com/HiIAmTzeKean/Streamsight/tree/master/examples>`_.
The notebooks demonstrate how to use the toolkit to build a recommendation
system and evaluate them.

Why Choose Streamsight?
=======================

- Real-World Applicability: Designed with a focus on real-world temporal contexts
  to enhance recommendation accuracy.

- Comprehensive Components: Offers a seamless integration of data handling,
  algorithm implementation, and evaluation.

- User-Friendly API: Simplifies the process of developing and testing
  recommendation systems, making it accessible for both researchers and practitioners.

Contents
========

.. toctree::
    :maxdepth: 1
    :caption: Data handling

    streamsightv2.matrix
    streamsightv2.preprocessing
    streamsightv2.datasets
    streamsightv2.settings

.. toctree::
    :maxdepth: 1
    :caption: RecSys

    streamsightv2.algorithms

.. toctree::
    :maxdepth: 1
    :caption: Evaluation

    streamsightv2.evaluators
    streamsightv2.metrics

.. toctree::
    :maxdepth: 1
    :caption: Other supporting modules

    streamsightv2.registries
    streamsightv2.utils

.. toctree::
    :maxdepth: 2
    :caption: Additional Resources

    poster


Indices and References
==================

* :ref:`genindex`
* :ref:`ref-sheet`