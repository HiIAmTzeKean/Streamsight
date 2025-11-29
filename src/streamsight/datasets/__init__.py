"""Dataset module for public datasets in streaming experiments.

The dataset module allows users to easily use to publicly available datasets
in their experiments. The dataset class are built on top of the :class:`Dataset`
allowing for easy extension and customization. In this module, we provide
the a few dataset that is available from public sources. The programmer is free to add more
datasets as they see fit by defining the abstract methods that must be implemented.

Other than the publicly available datasets, we also provide a test dataset
that can be used for testing purposes. The test dataset is a simple dataset
that can be used to test the functionality of the algorithms.

While the MovieLens100K dataset is available in the module, we recommend that
the programmer use the other publicly available datasets as the data are not
chunked into "blocks". The setting of a global timeline to split the data
could potentially cause a chuck of data to be lost.

Example
-------

If the file specified does not exist, the dataset is downloaded and written into this file.
Subsequent loading of the dataset will not require downloading the dataset again,
and will be obtained from the file in the directory.

```python
from streamsight.datasets import AmazonMusicDataset

dataset = AmazonMusicDataset()
data = dataset.load()
```

Each dataset can be loaded with default filters that are applied to the dataset.
To use the default filters, set `use_default_filters` parameter to True.
The dataset can also be loaded without filters and preprocessing of ID by
calling the :meth:`load` method with the parameter `apply_filters` set to False.
The recommended loading is with filters applied to ensure that the user and item
ids are incrementing in the order of time.

```python
from streamsight.datasets import AmazonMusicDataset

dataset = AmazonMusicDataset(use_default_filters=True)
data = dataset.load(apply_filters=False)
```

Available Datasets
------------------

The module provides the following datasets:

- :class:`AmazonBookDataset`: Amazon Books reviews
- :class:`AmazonMovieDataset`: Amazon Movies reviews
- :class:`AmazonMusicDataset`: Amazon Music reviews
- :class:`AmazonSubscriptionBoxesDataset`: Amazon Subscription Boxes reviews
- :class:`LastFMDataset`: Last.FM music listening history
- :class:`MovieLens100K`: MovieLens 100K rating dataset
- :class:`YelpDataset`: Yelp business reviews
- :class:`TestDataset`: Lightweight dataset for testing algorithms

Extending the Framework
-----------------------

To add custom datasets, inherit from :class:`Dataset` and implement all
abstract methods. Refer to the base class documentation for implementation details.

See Also
--------

:mod:`streamsight.preprocessing` : Data preprocessing and filtering utilities
"""

from streamsight.datasets.amazon import (
    AmazonBookDataset,
    AmazonMovieDataset,
    AmazonMusicDataset,
    AmazonSubscriptionBoxesDataset,
)
from streamsight.datasets.base import Dataset
from streamsight.datasets.lastfm import LastFMDataset
from streamsight.datasets.movielens import MovieLens100K
from streamsight.datasets.test import TestDataset
from streamsight.datasets.yelp import YelpDataset
