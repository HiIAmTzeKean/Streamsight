"""The dataset module allows users to easily use to publicly available datasets in their experiments.

.. currentmodule:: streamsight.datasets


.. autosummary::
    :toctree: generated/

    Dataset
    TestDataset
    AmazonBookDataset
    AmazonComputerDataset
    AmazonMovieDataset
    AmazonMusicDataset
    
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