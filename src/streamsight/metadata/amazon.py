import logging
import os

import numpy as np
import pandas as pd

from streamsight.metadata.base import Metadata

logger = logging.getLogger(__name__)


class AmazonItemMetadata(Metadata):
  # Column names
  MAIN_CATEGORY_IX = "main_category"
  """Name of the column in the DataFrame that contains the main category of the item."""
  TITLE_IX = "title"
  """Name of the column in the DataFrame that contains the title of the item."""
  AVERAGE_RATING_IX = "average_rating"
  """Name of the column in the DataFrame that contains the average rating of the item."""
  RATING_NUMBER_IX = "rating_number"
  """Name of the column in the DataFrame that contains the number of ratings of the item."""
  FEATURES_IX = "features"
  """Name of the column in the DataFrame that contains the features of the item."""
  DESCRIPTION_IX = "description"
  """Name of the column in the DataFrame that contains the description of the item."""
  PRICE_IX = "price"
  """Name of the column in the DataFrame that contains the price of the item."""
  IMAGES_IX = "images"
  """Name of the column in the DataFrame that contains the images of the item."""
  VIDEOS_IX = "videos"
  """Name of the column in the DataFrame that contains the videos of the item."""
  STORE_IX = "store"
  """Name of the column in the DataFrame that contains the store of the item."""
  CATEGORIES_IX = "categories"
  """Name of the column in the DataFrame that contains the categories of the item."""
  DETAILS_IX = "details"
  """Name of the column in the DataFrame that contains the details of the item."""
  ITEM_IX = "parent_asin"
  """Name of the column in the DataFrame that contains item identifiers."""
  BOUGHT_TOGETHER_IX = "bought_together"
  """Name of the column in the DataFrame that contains the items bought together with the item."""

  def __init__(self, item_id_mapping: pd.DataFrame):
    super().__init__()
    self.item_id_mapping = item_id_mapping

  def _load_metadata_dataframe(self) -> pd.DataFrame:
    self.fetch_metadata()
    df = pd.read_json(
        self.file_path,  # Ensure file_path contains the JSONL file path
        dtype={
            self.MAIN_CATEGORY_IX: str,
            self.TITLE_IX: str,
            self.AVERAGE_RATING_IX: np.float32,
            self.RATING_NUMBER_IX: np.int64,
            self.FEATURES_IX: list,
            self.DESCRIPTION_IX: list,
            self.PRICE_IX: np.float32,
            self.IMAGES_IX: list,
            self.VIDEOS_IX: list,
            self.STORE_IX: str,
            self.CATEGORIES_IX: list,
            self.DETAILS_IX: dict,
            self.ITEM_IX: str,
            self.BOUGHT_TOGETHER_IX: list,
        },
        lines=True,  # Required for JSONL format
    )

    item_id_to_iid = dict(zip(self.item_id_mapping[self.ITEM_IX], self.item_id_mapping['iid']))

    # Map ITEM_IX in metadata_df using the optimized function
    df[self.ITEM_IX] = df[self.ITEM_IX].map(lambda x: item_id_to_iid.get(x, x))

    return df
  

class AmazonMusicItemMetadata(AmazonItemMetadata):
  REMOTE_FILENAME = "meta_Digital_Music.jsonl.gz"
  METADATA_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Digital_Music.jsonl.gz"

  @property
  def DEFAULT_FILENAME(self) -> str:
    """Default filename that will be used if it is not specified by the user."""
    return self.REMOTE_FILENAME

  def _download_metadata(self):
    """Downloads the metadata for the dataset.
    
    Downloads the zipfile, and extracts the ratings file to `self.file_path`
    """
    if not self.METADATA_URL:
        raise ValueError(f"{self.name} does not have URL specified.")
    
    self._fetch_remote(self.METADATA_URL,
                        os.path.join(self.base_path, f"{self.DEFAULT_FILENAME}"))


class AmazonMovieItemMetadata(AmazonItemMetadata):
  REMOTE_FILENAME = "meta_Movies_and_TV.jsonl.gz"
  METADATA_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Movies_and_TV.jsonl.gz"

  @property
  def DEFAULT_FILENAME(self) -> str:
    """Default filename that will be used if it is not specified by the user."""
    return self.REMOTE_FILENAME

  def _download_metadata(self):
    """Downloads the metadata for the dataset.
    
    Downloads the zipfile, and extracts the ratings file to `self.file_path`
    """
    if not self.METADATA_URL:
        raise ValueError(f"{self.name} does not have URL specified.")
    
    self._fetch_remote(self.METADATA_URL,
                        os.path.join(self.base_path, f"{self.DEFAULT_FILENAME}"))

class AmazonSubscriptionBoxesItemMetadata(AmazonItemMetadata):
  REMOTE_FILENAME = "meta_Subscription Boxes.jsonl.gz"
  METADATA_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Subscription_Boxes.jsonl.gz"

  @property
  def DEFAULT_FILENAME(self) -> str:
    """Default filename that will be used if it is not specified by the user."""
    return self.REMOTE_FILENAME

  def _download_metadata(self):
    """Downloads the metadata for the dataset.
    
    Downloads the zipfile, and extracts the ratings file to `self.file_path`
    """
    if not self.METADATA_URL:
        raise ValueError(f"{self.name} does not have URL specified.")
    
    self._fetch_remote(self.METADATA_URL,
                        os.path.join(self.base_path, f"{self.DEFAULT_FILENAME}"))
    
class AmazonBookItemMetadata(AmazonItemMetadata):
  REMOTE_FILENAME = "meta_Books.jsonl.gz"
  METADATA_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz"

  @property
  def DEFAULT_FILENAME(self) -> str:
    """Default filename that will be used if it is not specified by the user."""
    return self.REMOTE_FILENAME

  def _download_metadata(self):
    """Downloads the metadata for the dataset.
    
    Downloads the zipfile, and extracts the ratings file to `self.file_path`
    """
    if not self.METADATA_URL:
        raise ValueError(f"{self.name} does not have URL specified.")
    
    self._fetch_remote(self.METADATA_URL,
                        os.path.join(self.base_path, f"{self.DEFAULT_FILENAME}"))
    
