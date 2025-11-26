# streamsight/datasets/config.py
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Base configuration for datasets."""

    user_ix: str = "user_id"
    item_ix: str = "item_id"
    timestamp_ix: str = "timestamp"
    dataset_url: str = "http://example.com"
    remote_zipname: str = ""
    remote_filename: str = ""

    @property
    def default_filename(self) -> str:
        """Derived filename from remote components."""
        if not self.remote_zipname or not self.remote_filename:
            return "dataset.csv"
        return f"{self.remote_zipname}_{self.remote_filename}"
