from dataclasses import dataclass

from .base import DatasetConfig


@dataclass
class AmazonDatasetConfig(DatasetConfig):
    """Amazon dataset base configuration."""

    user_ix: str = "user_id"
    item_ix: str = "parent_asin"
    timestamp_ix: str = "timestamp"
    rating_ix: str = "rating"
    helpful_vote_ix: str = "helpful_vote"
    dataset_url: str = ""  # Set per subclass
    remote_filename: str = ""  # Set per subclass
    default_base_path: str = DatasetConfig.default_base_path + "/amazon"

    @property
    def default_filename(self) -> str:
        """Return just the filename for Amazon datasets (no zipname prefix)."""
        return self.remote_filename


@dataclass
class AmazonMusicDatasetConfig(AmazonDatasetConfig):
    """Amazon Music dataset configuration."""

    remote_filename: str = "Digital_Music.jsonl.gz"
    dataset_url: str = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Digital_Music.jsonl.gz"


@dataclass
class AmazonMovieDatasetConfig(AmazonDatasetConfig):
    """Amazon Movie dataset configuration."""

    remote_filename: str = "Movies_and_TV.jsonl.gz"
    dataset_url: str = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Movies_and_TV.jsonl.gz"


@dataclass
class AmazonSubscriptionBoxesDatasetConfig(AmazonDatasetConfig):
    """Amazon Subscription Boxes dataset configuration."""

    remote_filename: str = "Subscription_Boxes.jsonl.gz"
    dataset_url: str = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Subscription_Boxes.jsonl.gz"


@dataclass
class AmazonBookDatasetConfig(AmazonDatasetConfig):
    """Amazon Books dataset configuration."""

    remote_filename: str = "Books.jsonl.gz"
    dataset_url: str = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz"
