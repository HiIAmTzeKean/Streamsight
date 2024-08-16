"""
.. currentmodule:: streamsight.datasets

Dataset
-------------

The dataset module allows users to easily use to publicly available datasets
in their experiments. The dataset class are built on top of the :class:`Dataset`
allowing for easy extension and customization. In this module, we provide
the Amazon datasets and the Yelp dataset. The programmer is free to add more
datasets as they see fit by defining the abstract methods that must be implemented.

Other than the 2 publicly available datasets, we also provide a test dataset
that can be used for testing purposes.

.. autosummary::
    :toctree: generated/

    Dataset
    TestDataset
    AmazonBookDataset
    AmazonComputerDataset
    AmazonMovieDataset
    AmazonMusicDataset
    YelpDataset
    
Example
~~~~~~~~~

Loading a dataset only takes a couple of lines.
If the file specified does not exist, the dataset is downloaded and written into this file.
Subsequent loading of the dataset then happens from this file.

.. code-block:: python

    from streamsight.datasets import AmazonMusicDataset

    dataset = AmazonMusicDataset()
    data = dataset.load()


Each dataset has its own default preprocessing steps, documented in the classes respectively.
To use custom preprocessing a couple more lines should be added to the example.

.. code-block:: python

    from streamsight.datasets import AmazonMusicDataset

    dataset = AmazonMusicDataset()
    data = dataset.load()

For an overview of available filters see :mod:`streamsight.preprocessing`
"""

from streamsight.datasets.base import Dataset
from streamsight.datasets.test import TestDataset
from streamsight.datasets.amazon import (AmazonBookDataset,
                                         AmazonComputerDataset,
                                         AmazonMovieDataset,
                                         AmazonMusicDataset)
from streamsight.datasets.yelp import YelpDataset