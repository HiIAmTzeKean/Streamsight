import os
import zipfile

import numpy as np
import pandas as pd

from streamsight.metadata.base import Metadata

class MovieLens100kUserMetadata(Metadata):
  REMOTE_FILENAME = "u.user"
  REMOTE_ZIPNAME = "ml-100k"

  METADATA_URL = "http://files.grouplens.org/datasets/movielens"

  # Column names
  USER_IX = "userId"
  AGE_IX = "age"
  GENDER_IX = "gender"
  OCCUPATION_IX = "occupation"
  ZIPCODE_IX = "zipcode"

  def __init__(self, user_id_mapping: pd.DataFrame):
    super().__init__()
    self.user_id_mapping = user_id_mapping

  @property
  def DEFAULT_FILENAME(self) -> str:
    """Default filename that will be used if it is not specified by the user."""
    return f"{self.REMOTE_ZIPNAME}_{self.REMOTE_FILENAME}"
  
  def _download_metadata(self):
    """Downloads the metadata for the dataset.
    
    Downloads the zipfile, and extracts the ratings file to `self.file_path`
    """
    # Download the zip into the data directory
    self._fetch_remote(
      f"{self.METADATA_URL}/{self.REMOTE_ZIPNAME}.zip",
      os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"),
    )

    # Extract the ratings file which we will use
    with zipfile.ZipFile(
      os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"), "r"
    ) as zip_ref:
      zip_ref.extract(
        f"{self.REMOTE_ZIPNAME}/{self.REMOTE_FILENAME}", self.base_path
      )

    # Rename the ratings file to the specified filename
    os.rename(
      os.path.join(
        self.base_path, f"{self.REMOTE_ZIPNAME}/{self.REMOTE_FILENAME}"
      ),
      self.file_path,
    )

  def _load_metadata_dataframe(self) -> pd.DataFrame:
    self.fetch_metadata()
    df = pd.read_table(
        self.file_path,
        dtype={
            self.AGE_IX: np.int64,
            self.GENDER_IX: str,
            self.OCCUPATION_IX: str,
            self.ZIPCODE_IX: str,
        },
        sep="|",
        names=[
            self.USER_IX,
            self.AGE_IX,
            self.GENDER_IX,
            self.OCCUPATION_IX,
            self.ZIPCODE_IX,
        ],
        converters={self.USER_IX: self._map_user_id},
    )
    return df
  
  def _map_user_id(self, user_id):
    user_id_to_uid = dict(zip(self.user_id_mapping[self.USER_IX], self.user_id_mapping['uid']))
    return user_id_to_uid.get(int(user_id), user_id)


class MovieLens100kItemMetadata(Metadata):
  REMOTE_FILENAME = "u.item"
  REMOTE_ZIPNAME = "ml-100k"

  METADATA_URL = "http://files.grouplens.org/datasets/movielens"

  # Column names
  ITEM_IX = "movieId"
  TITLE_IX = "title"
  RELEASE_DATE_IX = "releaseDate"
  VIDEO_RELEASE_DATE_IX = "videoReleaseDate"
  IMDB_URL_IX = "imdbUrl"
  UNKNOWN_IX = "unknown"
  ACTION_IX = "action"
  ADVENTURE_IX = "adventure"
  ANIMATION_IX = "animation"
  CHILDREN_IX = "children"
  COMEDY_IX = "comedy"
  CRIME_IX = "crime"
  DOCUMENTARY_IX = "documentary"
  DRAMA_IX = "drama"
  FANTASY_IX = "fantasy"
  FILM_NOIR_IX = "filmNoir"
  HORROR_IX = "horror"
  MUSICAL_IX = "musical"
  MYSTERY_IX = "mystery"
  ROMANCE_IX = "romance"
  SCI_FI_IX = "sciFi"
  THRILLER_IX = "thriller"
  WAR_IX = "war"
  WESTERN_IX = "western"

  def __init__(self, item_id_mapping: pd.DataFrame):
    super().__init__()
    self.item_id_mapping = item_id_mapping

  @property
  def DEFAULT_FILENAME(self) -> str:
    """Default filename that will be used if it is not specified by the user."""
    return f"{self.REMOTE_ZIPNAME}_{self.REMOTE_FILENAME}"
  
  def _download_metadata(self):
    """Downloads the metadata for the dataset.
    
    Downloads the zipfile, and extracts the ratings file to `self.file_path`
    """
    # Download the zip into the data directory
    self._fetch_remote(
      f"{self.METADATA_URL}/{self.REMOTE_ZIPNAME}.zip",
      os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"),
    )

    # Extract the ratings file which we will use
    with zipfile.ZipFile(
      os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"), "r"
    ) as zip_ref:
      zip_ref.extract(
        f"{self.REMOTE_ZIPNAME}/{self.REMOTE_FILENAME}", self.base_path
      )

    # Rename the ratings file to the specified filename
    os.rename(
      os.path.join(
        self.base_path, f"{self.REMOTE_ZIPNAME}/{self.REMOTE_FILENAME}"
      ),
      self.file_path,
    )

  def _load_metadata_dataframe(self) -> pd.DataFrame:
    self.fetch_metadata()
    df = pd.read_table(
        self.file_path,
        dtype={
            self.TITLE_IX: str,
            self.RELEASE_DATE_IX: str,
            self.VIDEO_RELEASE_DATE_IX: str,
            self.IMDB_URL_IX: str,
            self.UNKNOWN_IX: np.int64,
            self.ACTION_IX: np.int64,
            self.ADVENTURE_IX: np.int64,
            self.ANIMATION_IX: np.int64,
            self.CHILDREN_IX: np.int64,
            self.COMEDY_IX: np.int64,
            self.CRIME_IX: np.int64,
            self.DOCUMENTARY_IX: np.int64,
            self.DRAMA_IX: np.int64,
            self.FANTASY_IX: np.int64,
            self.FILM_NOIR_IX: np.int64,
            self.HORROR_IX: np.int64,
            self.MUSICAL_IX: np.int64,
            self.MYSTERY_IX: np.int64,
            self.ROMANCE_IX: np.int64,
            self.SCI_FI_IX: np.int64,
            self.THRILLER_IX: np.int64,
            self.WAR_IX: np.int64,
            self.WESTERN_IX: np.int64,
        },
        sep="|",
        names=[
            self.ITEM_IX,
            self.TITLE_IX,
            self.RELEASE_DATE_IX,
            self.VIDEO_RELEASE_DATE_IX,
            self.IMDB_URL_IX,
            self.UNKNOWN_IX,
            self.ACTION_IX,
            self.ADVENTURE_IX,
            self.ANIMATION_IX,
            self.CHILDREN_IX,
            self.COMEDY_IX,
            self.CRIME_IX,
            self.DOCUMENTARY_IX,
            self.DRAMA_IX,
            self.FANTASY_IX,
            self.FILM_NOIR_IX,
            self.HORROR_IX,
            self.MUSICAL_IX,
            self.MYSTERY_IX,
            self.ROMANCE_IX,
            self.SCI_FI_IX,
            self.THRILLER_IX,
            self.WAR_IX,
            self.WESTERN_IX,
        ],
        converters={
          self.ITEM_IX: self._map_item_id,
        },
        encoding="ISO-8859-1",
    )
    return df
  
  def _map_item_id(self, item_id):
    item_id_to_iid = dict(zip(self.item_id_mapping[self.ITEM_IX], self.item_id_mapping['iid']))
    return item_id_to_iid.get(int(item_id), item_id)
  
