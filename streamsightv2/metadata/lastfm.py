import logging
import os
import zipfile

import numpy as np
import pandas as pd

from streamsight2.metadata.base import Metadata

logger = logging.getLogger(__name__)


class LastFMUserMetadata(Metadata):
  REMOTE_FILENAME = "user_friends.dat"
  REMOTE_ZIPNAME = "hetrec2011-lastfm-2k"

  METADATA_URL = "https://files.grouplens.org/datasets/hetrec2011"

  # Column names
  USER_IX = "userID"
  FRIEND_IX = "friendID"

  def __init__(self, user_id_mapping: pd.DataFrame):
    super().__init__()
    self.user_id_mapping = user_id_mapping

  @property
  def DEFAULT_FILENAME(self) -> str:
    """Default filename that will be used if it is not specified by the user."""
    return self.REMOTE_FILENAME
  
  def fetch_metadata(self, force=False) -> None:
    """Check if dataset is present, if not download

    :param force: If True, dataset will be downloaded,
            even if the file already exists.
            Defaults to False.
    :type force: bool, optional
    """
    path = os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip")
    file = self.REMOTE_FILENAME
    if not os.path.exists(path) or force:
        logger.debug(f"{self.name} dataset zipfile not found in {path}.")
        self._download_metadata()
    elif not os.path.exists(self.file_path) or force:
        logger.debug(
            f"{self.name} dataset file not found, but the zipfile has already been downloaded. Extracting file from zipfile."
        )
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extract(file, self.base_path)

    logger.debug(f"Data zipfile is in memory and in dir specified.")

  def _download_metadata(self):
    """Downloads the metadata for the dataset.
    
    Downloads the zipfile, and extracts the ratings file to `self.file_path`
    """
    # Download the zip into the data directory
    self._fetch_remote(
        f"{self.METADATA_URL}/{self.REMOTE_ZIPNAME}.zip",
        os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"),
    )

    # Extract the interaction file which we will use
    with zipfile.ZipFile(
        os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"), "r"
    ) as zip_ref:
        zip_ref.extract(f"{self.REMOTE_FILENAME}", self.base_path)

  def _load_metadata_dataframe(self) -> pd.DataFrame:
    self.fetch_metadata()
    df = pd.read_csv(
        self.file_path,
        sep="\t",
        names=[
            self.USER_IX,
            self.FRIEND_IX,
        ],
        converters={
          self.USER_IX: self._map_user_id, 
          self.FRIEND_IX: self._map_user_id
        },
        header=0,
    )
    return df
  
  def _map_user_id(self, user_id):
    user_id_to_uid = dict(zip(self.user_id_mapping[self.USER_IX], self.user_id_mapping['uid']))
    return user_id_to_uid.get(int(user_id), user_id)


class LastFMItemMetadata(Metadata):
  REMOTE_FILENAME = "artists.dat"
  REMOTE_ZIPNAME = "hetrec2011-lastfm-2k"

  METADATA_URL = "https://files.grouplens.org/datasets/hetrec2011"

  # Column names
  ITEM_IX = "id"
  NAME_IX = "name"
  URL_IX = "url"
  PICTURE_URL_IX = "pictureURL"

  def __init__(self, item_id_mapping: pd.DataFrame):
    super().__init__()
    self.item_id_mapping = item_id_mapping

  @property
  def DEFAULT_FILENAME(self) -> str:
    """Default filename that will be used if it is not specified by the user."""
    return self.REMOTE_FILENAME

  def fetch_metadata(self, force=False) -> None:
    """Check if dataset is present, if not download

    :param force: If True, dataset will be downloaded,
            even if the file already exists.
            Defaults to False.
    :type force: bool, optional
    """
    path = os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip")
    file = self.REMOTE_FILENAME
    if not os.path.exists(path) or force:
        logger.debug(f"{self.name} dataset zipfile not found in {path}.")
        self._download_metadata()
    elif not os.path.exists(self.file_path) or force:
        logger.debug(
            f"{self.name} dataset file not found, but the zipfile has already been downloaded. Extracting file from zipfile."
        )
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extract(file, self.base_path)

    logger.debug(f"Data zipfile is in memory and in dir specified.")

  def _download_metadata(self):
    """Downloads the metadata for the dataset.
    
    Downloads the zipfile, and extracts the ratings file to `self.file_path`
    """
    # Download the zip into the data directory
    self._fetch_remote(
        f"{self.METADATA_URL}/{self.REMOTE_ZIPNAME}.zip",
        os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"),
    )

    # Extract the interaction file which we will use
    with zipfile.ZipFile(
        os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"), "r"
    ) as zip_ref:
        zip_ref.extract(f"{self.REMOTE_FILENAME}", self.base_path)

  def _load_metadata_dataframe(self) -> pd.DataFrame:
    self.fetch_metadata()
    df = pd.read_csv(
        self.file_path,
        dtype={
            self.NAME_IX: str,
            self.URL_IX: str,
            self.PICTURE_URL_IX: str,
        },
        sep="\t",
        names=[
            self.ITEM_IX,
            self.NAME_IX,
            self.URL_IX,
            self.PICTURE_URL_IX,
        ],
        converters={
          self.ITEM_IX: self._map_item_id,
        },
        header=0,
    )
    return df
  
  def _map_item_id(self, item_id):
    item_id_to_iid = dict(zip(self.item_id_mapping["artistID"], self.item_id_mapping['iid']))
    return item_id_to_iid.get(int(item_id), item_id)


class LastFMTagMetadata(Metadata):
  REMOTE_FILENAME = "tags.dat"
  REMOTE_ZIPNAME = "hetrec2011-lastfm-2k"

  METADATA_URL = "https://files.grouplens.org/datasets/hetrec2011"

  # Column names
  TAG_IX = "tagID"
  TAG_VALUE_IX = "tagValue"

  def __init__(self):
    super().__init__()

  @property
  def DEFAULT_FILENAME(self) -> str:
    """Default filename that will be used if it is not specified by the user."""
    return self.REMOTE_FILENAME
  
  def fetch_metadata(self, force=False) -> None:
    """Check if dataset is present, if not download

    :param force: If True, dataset will be downloaded,
            even if the file already exists.
            Defaults to False.
    :type force: bool, optional
    """
    path = os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip")
    file = self.REMOTE_FILENAME
    if not os.path.exists(path) or force:
        logger.debug(f"{self.name} dataset zipfile not found in {path}.")
        self._download_metadata()
    elif not os.path.exists(self.file_path) or force:
        logger.debug(
            f"{self.name} dataset file not found, but the zipfile has already been downloaded. Extracting file from zipfile."
        )
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extract(file, self.base_path)

    logger.debug(f"Data zipfile is in memory and in dir specified.")

  def _download_metadata(self):
    """Downloads the metadata for the dataset.
    
    Downloads the zipfile, and extracts the ratings file to `self.file_path`
    """
    # Download the zip into the data directory
    self._fetch_remote(
        f"{self.METADATA_URL}/{self.REMOTE_ZIPNAME}.zip",
        os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"),
    )

    # Extract the interaction file which we will use
    with zipfile.ZipFile(
        os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"), "r"
    ) as zip_ref:
        zip_ref.extract(f"{self.REMOTE_FILENAME}", self.base_path)

  def _load_metadata_dataframe(self) -> pd.DataFrame:
    self.fetch_metadata()
    df = pd.read_csv(
        self.file_path,
        dtype={
            self.TAG_IX: np.int64,
            self.TAG_VALUE_IX: str,
        },
        sep="\t",
        names=[
            self.TAG_IX,
            self.TAG_VALUE_IX,
        ],
        encoding="ISO-8859-1",
        header=0,
    )
    return df
  
