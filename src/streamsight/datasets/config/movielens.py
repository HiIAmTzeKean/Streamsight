from dataclasses import dataclass

from .base import DatasetConfig


@dataclass
class MovieLensConfig(DatasetConfig):
    """MovieLens base configuration."""

    user_ix: str = "userId"
    item_ix: str = "movieId"
    timestamp_ix: str = "timestamp"
    rating_ix: str = "rating"
    """Name of the column in the DataFrame that contains the rating a user gave to the item."""
    dataset_url: str = "https://files.grouplens.org/datasets/movielens"
    remote_zipname: str = "ml-100k"
    """Name of the zip-file on the MovieLens server."""
    remote_filename: str = "ratings.csv"
    """Name of the file containing user ratings on the MovieLens server."""
    default_base_path: str = DatasetConfig.default_base_path + "/movielens"


@dataclass
class MovieLens100KConfig(MovieLensConfig):
    """MovieLens 100K specific configuration."""

    remote_filename: str = "u.data"

