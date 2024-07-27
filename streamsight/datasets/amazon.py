import logging
import os
import pandas as pd
import numpy as np
from streamsight.datasets.base import Dataset

logger = logging.getLogger(__name__)

class AmazonDataset(Dataset):
    USER_IX = "userId"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "songId"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "timestamp"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""
    RATING_IX = "rating"
    """Name of the column in the DataFrame that contains the rating a user gave to the item."""
    DATASET_URL=None
    """URL to fetch the dataset from."""
    
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
        df = pd.read_csv(
            self.file_path,
            dtype={
                self.ITEM_IX: str,
                self.USER_IX: str,
                self.RATING_IX: np.float32,
                self.TIMESTAMP_IX: np.int64,
            },
            names=[self.ITEM_IX, self.USER_IX, self.RATING_IX, self.TIMESTAMP_IX],
        )
        # sort by timestamp to incrementally assign user and item ids by timestamp
        df.sort_values(by=[self.TIMESTAMP_IX], inplace=True, ignore_index=True)
        
        
        df[self.USER_IX] = pd.CategoricalIndex(df[self.USER_IX],categories=df[self.USER_IX].unique()).codes
        df[self.ITEM_IX] = pd.CategoricalIndex(df[self.ITEM_IX],categories=df[self.ITEM_IX].unique()).codes
        return df

class AmazonMusicDataset(AmazonDataset):
    """Handles Amazon Music dataset."""
    DEFAULT_FILENAME = "amazon_digitalmusic_dataset.csv"
    """Default filename that will be used if it is not specified by the user."""

    DATASET_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Digital_Music.csv"
    """URL to fetch the dataset from."""

class AmazonMovieDataset(AmazonDataset):
    """Handles Amazon Movie dataset."""
    DEFAULT_FILENAME = "amazon_movie_dataset.csv"
    """Default filename that will be used if it is not specified by the user."""

    DATASET_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Movies_and_TV.csv"
    """URL to fetch the dataset from."""