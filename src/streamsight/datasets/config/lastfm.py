from dataclasses import dataclass

from .base import DatasetConfig


@dataclass
class LastFMConfig(DatasetConfig):
    """LastFM dataset configuration."""

    user_ix: str = "userID"
    item_ix: str = "artistID"
    timestamp_ix: str = "timestamp"
    tag_ix: str = "tagID"
    """Name of the column in the DataFrame that contains the tag a user gave to the item."""
    dataset_url: str = "https://files.grouplens.org/datasets/hetrec2011"
    remote_zipname: str = "hetrec2011-lastfm-2k"
    remote_filename: str = "user_taggedartists-timestamps.dat"
    default_base_path: str = DatasetConfig.default_base_path + "/lastfm"
