import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from streamsight.datasets.base import Dataset
from streamsight.metadata.amazon import AmazonBookItemMetadata, AmazonMovieItemMetadata, AmazonMusicItemMetadata, AmazonSubscriptionBoxesItemMetadata

logger = logging.getLogger(__name__)
tqdm.pandas()

class AmazonDataset(Dataset):
    ITEM_IX = "parent_asin"
    """Name of the column in the DataFrame that contains item identifiers."""
    USER_IX = "user_id"
    """Name of the column in the DataFrame that contains user identifiers."""
    TIMESTAMP_IX = "timestamp"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""
    RATING_IX = "rating"
    """Name of the column in the DataFrame that contains the rating a user gave to the item."""
    HELPFUL_VOTE_IX = "helpful_vote"
    """Name of the column in the DataFrame that contains the helpful vote of the item."""
    DATASET_URL=None
    """URL to fetch the dataset from."""
    
    ITEM_METADATA = None

    def _download_dataset(self):
        """Downloads the dataset.

        Downloads the csv file from the dataset URL and saves it to the file path.
        """
        if not self.DATASET_URL:
            raise ValueError(f"{self.name} does not have URL specified.")
        
        self._fetch_remote(self.DATASET_URL,
                           os.path.join(self.base_path, f"{self.DEFAULT_FILENAME}"))

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.
        
        Transform the dataset downloaded to have integer user and item ids. This
        will be needed for representation in the interaction matrix.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """
        self.fetch_dataset()

        df = pd.read_json(
            self.file_path,  # Ensure file_path contains the JSONL file path
            dtype={
                self.ITEM_IX: str,
                self.USER_IX: str,
                self.TIMESTAMP_IX: np.int64,
                self.RATING_IX: np.float32,
                self.HELPFUL_VOTE_IX: np.int64
            },
            lines=True,  # Required for JSONL format
        )

        # Select only the required columns
        df = df[[self.ITEM_IX, self.USER_IX, self.TIMESTAMP_IX, self.RATING_IX, self.HELPFUL_VOTE_IX]]
        df[self.TIMESTAMP_IX] = df[self.TIMESTAMP_IX] // 1_000_000_000  # Convert to seconds

        return df
    

class AmazonMusicDataset(AmazonDataset):
    """Handles Amazon Music dataset."""
    DEFAULT_FILENAME = "Digital_Music.jsonl.gz"
    """Default filename that will be used if it is not specified by the user."""

    DATASET_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Digital_Music.jsonl.gz"
    """URL to fetch the dataset from."""

    def _fetch_dataset_metadata(self, user_id_mapping: pd.DataFrame, item_id_mapping: pd.DataFrame):
        self.ITEM_METADATA = AmazonMusicItemMetadata(item_id_mapping=item_id_mapping).load()

class AmazonMovieDataset(AmazonDataset):
    """Handles Amazon Movie dataset."""
    DEFAULT_FILENAME = "Movies_and_TV.jsonl.gz"
    """Default filename that will be used if it is not specified by the user."""

    DATASET_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Movies_and_TV.jsonl.gz"
    """URL to fetch the dataset from."""

    def _fetch_dataset_metadata(self, user_id_mapping: pd.DataFrame, item_id_mapping: pd.DataFrame):
        self.ITEM_METADATA = AmazonMovieItemMetadata(item_id_mapping=item_id_mapping).load()
    
class AmazonSubscriptionBoxesDataset(AmazonDataset):
    """Handles Amazon Computer dataset."""
    DEFAULT_FILENAME = "Subscription_Boxes.jsonl.gz"
    """Default filename that will be used if it is not specified by the user."""

    DATASET_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Subscription_Boxes.jsonl.gz"
    """URL to fetch the dataset from."""

    def _fetch_dataset_metadata(self, user_id_mapping: pd.DataFrame, item_id_mapping: pd.DataFrame):
        self.ITEM_METADATA = AmazonSubscriptionBoxesItemMetadata(item_id_mapping=item_id_mapping).load()
    
class AmazonBookDataset(AmazonDataset):
    """Handles Amazon Book dataset."""
    DEFAULT_FILENAME = "Books.jsonl.gz"
    """Default filename that will be used if it is not specified by the user."""

    DATASET_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz"
    """URL to fetch the dataset from."""

    def _fetch_dataset_metadata(self, user_id_mapping: pd.DataFrame, item_id_mapping: pd.DataFrame):
        self.ITEM_METADATA = AmazonBookItemMetadata(item_id_mapping=item_id_mapping).load()

        