"""Dataset configurations."""

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
