"""
.. currentmodule:: streamsight.datasets.config

Exports
~~~~~~~
The following configuration classes are provided by this module and are
re-exported from `streamsight.datasets.config`:

.. autosummary::
    :toctree: generated/

    MovieLensDatasetConfig
    MovieLens100KDatasetConfig
    AmazonDatasetConfig
    AmazonMusicDatasetConfig
    AmazonMovieDatasetConfig
    AmazonBookDatasetConfig
    AmazonSubscriptionBoxesDatasetConfig
    LastFMDatasetConfig
    YelpDatasetConfig
    DatasetConfig

Usage
~~~~~
A typical usage pattern is to import a dataset config, optionally override
fields, and pass it to dataset-loading utilities or your own convenience
wrappers:

.. code-block:: python

    from streamsight.datasets.config import AmazonMusicDatasetConfig

    # Create config instance using defaults
    cfg = AmazonMusicDatasetConfig()

    # Inspect config values (example fields, actual attributes depend on class)
    print(cfg.name)
    print(cfg.local_path)
    print(cfg.source_url)

    # Optionally override defaults at runtime
    custom_cfg = AmazonMusicDatasetConfig(min_user_interactions=5, min_item_interactions=10)

"""

from .amazon import (
    AmazonBookDatasetConfig,
    AmazonDatasetConfig,
    AmazonMovieDatasetConfig,
    AmazonMusicDatasetConfig,
    AmazonSubscriptionBoxesDatasetConfig,
)
from .base import DatasetConfig
from .lastfm import LastFMDatasetConfig
from .movielens import (
    MovieLens100KDatasetConfig,
    MovieLensDatasetConfig,
)
from .yelp import YelpDatasetConfig


__all__ = [
    "MovieLensDatasetConfig",
    "MovieLens100KDatasetConfig",
    "AmazonDatasetConfig",
    "AmazonMusicDatasetConfig",
    "AmazonMovieDatasetConfig",
    "AmazonBookDatasetConfig",
    "AmazonSubscriptionBoxesDatasetConfig",
    "LastFMDatasetConfig",
    "YelpDatasetConfig", "DatasetConfig",
]
