"""
.. currentmodule:: streamsight.datasets.config

Exports
~~~~~~~
The following configuration classes are provided by this module and are
re-exported from `streamsight.datasets.config`:

.. autosummary::
    :toctree: generated/

    MovieLensConfig
    MovieLens100KConfig
    AmazonConfig
    AmazonMusicConfig
    AmazonMovieConfig
    AmazonBookConfig
    AmazonSubscriptionBoxesConfig
    LastFMConfig
    YelpConfig
    DatasetConfig

Usage
~~~~~
A typical usage pattern is to import a dataset config, optionally override
fields, and pass it to dataset-loading utilities or your own convenience
wrappers:

.. code-block:: python

    from streamsight.datasets.config import AmazonMusicConfig

    # Create config instance using defaults
    cfg = AmazonMusicConfig()

    # Inspect config values (example fields, actual attributes depend on class)
    print(cfg.name)
    print(cfg.local_path)
    print(cfg.source_url)

    # Optionally override defaults at runtime
    custom_cfg = AmazonMusicConfig(min_user_interactions=5, min_item_interactions=10)

"""

from .amazon import (
    AmazonBookConfig,
    AmazonConfig,
    AmazonMovieConfig,
    AmazonMusicConfig,
    AmazonSubscriptionBoxesConfig,
)
from .base import DatasetConfig
from .lastfm import LastFMConfig
from .movielens import (
    MovieLens100KConfig,
    MovieLensConfig,
)
from .yelp import YelpConfig


__all__ = [
    "MovieLensConfig",
    "MovieLens100KConfig",
    "AmazonConfig",
    "AmazonMusicConfig",
    "AmazonMovieConfig",
    "AmazonBookConfig",
    "AmazonSubscriptionBoxesConfig",
    "LastFMConfig",
    "YelpConfig", "DatasetConfig",
]
