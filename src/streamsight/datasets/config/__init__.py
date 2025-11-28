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
    AmazonBooksItemMetadataConfig,
    AmazonDatasetConfig,
    AmazonDigitalMusicItemMetadataConfig,
    AmazonItemMetadataConfig,
    AmazonMovieDatasetConfig,
    AmazonMoviesAndTVItemMetadataConfig,
    AmazonMusicDatasetConfig,
    AmazonSubscriptionBoxesDatasetConfig,
    AmazonSubscriptionBoxesItemMetadataConfig,
)
from .base import DatasetConfig, MetadataConfig
from .lastfm import (
    LastFMDatasetConfig,
    LastFMItemMetadataConfig,
    LastFMTagMetadataConfig,
    LastFMUserMetadataConfig,
)
from .movielens import (
    MovieLens100KDatasetConfig,
    MovieLens100kItemMetadataConfig,
    MovieLens100kUserMetadataConfig,
    MovieLensDatasetConfig,
)
from .yelp import YelpDatasetConfig


__all__ = [
    "AmazonDatasetConfig",
    "AmazonMusicDatasetConfig",
    "AmazonMovieDatasetConfig",
    "AmazonBookDatasetConfig",
    "AmazonSubscriptionBoxesDatasetConfig",
    "LastFMDatasetConfig",
    "YelpDatasetConfig",
    "DatasetConfig",
    "MetadataConfig",
    "MovieLensDatasetConfig",
    "MovieLens100KDatasetConfig",
    "MovieLens100kItemMetadataConfig",
    "MovieLens100kUserMetadataConfig",
    "AmazonBooksItemMetadataConfig",
    "AmazonDigitalMusicItemMetadataConfig",
    "AmazonItemMetadataConfig",
    "AmazonMoviesAndTVItemMetadataConfig",
    "AmazonSubscriptionBoxesItemMetadataConfig",
    "LastFMItemMetadataConfig",
    "LastFMTagMetadataConfig",
    "LastFMUserMetadataConfig",
]
