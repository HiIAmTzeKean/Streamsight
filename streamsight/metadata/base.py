import logging
import os
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from typing import Optional
from abc import ABC, abstractmethod

from streamsight.utils.util import ProgressBar


logger = logging.getLogger(__name__)

class Metadata(ABC):
  DEFAULT_FILENAME = None
  """Default filename that will be used if it is not specified by the user."""
  
  DEFAULT_BASE_PATH = "metadata"
  """Default base path where the dataset will be stored."""
  
  def __init__(self, filename: Optional[str] = None, base_path: Optional[str] = None):
    self.base_path = base_path if base_path else self.DEFAULT_BASE_PATH
    logger.debug(
        f"{self.name} being initialized with '{self.base_path}' as the base path."
    )

    self.filename = filename if filename else self.DEFAULT_FILENAME
    if not self.filename:
        raise ValueError("No filename specified, and no default known.")

    self._check_safe()
    logger.debug(f"{self.name} is initialized.")


  @property
  def name(self):
      """Name of the object's class."""
      return self.__class__.__name__
  
  @property
  def file_path(self) -> str:
    """File path of the metadata."""
    return os.path.join(self.base_path, self.filename) # type: ignore
  
  def _check_safe(self):
    """Check if the directory is safe. If directory does not exit, create it."""
    p = Path(self.base_path)
    p.mkdir(exist_ok=True)
    
  def fetch_metadata(self, force=False) -> None:
    """Check if metadata is present, if not download

    :param force: If True, metadata will be downloaded,
            even if the file already exists.
            Defaults to False.
    :type force: bool, optional
    """
    if not os.path.exists(self.file_path) or force:
        logger.debug(f"{self.name} metadata not found in {self.file_path}.")
        self._download_metadata()
    logger.debug(f"Data file is in memory and in dir specified.")
  
  def load(self) -> pd.DataFrame:
    """Load the metadata from file and return it as a DataFrame.

    :return: Dataframe containing the metadata
    :rtype: pd.DataFrame
    """
    return self._load_metadata_dataframe()

  def _fetch_remote(self, url: str, filename: str) -> str:
    """Fetch metadata from remote url and save locally

    :param url: url to fetch metadata from
    :type url: str
    :param filename: Path to save file to
    :type filename: str
    :return: The filename where metadata was saved
    :rtype: str
    """
    logger.debug(f"{self.name} will fetch metadata from remote url at {url}.")
    urlretrieve(url, filename, ProgressBar())
    return filename

  @abstractmethod
  def _load_metadata_dataframe(self) -> pd.DataFrame:
    """Load the raw metadata from file, and return it as a pandas DataFrame.

    .. warning::
        This does not apply any preprocessing, and returns the raw dataset.

    :return: Dataframe containing the raw metadata
    :rtype: pd.DataFrame
    """
    raise NotImplementedError("Needs to be implemented")
  
  @abstractmethod
  def _download_metadata(self):
    """Downloads the metadata.

    Downloads the csv file from the metadata URL and saves it to the file path.
    """
    raise NotImplementedError("Needs to be implemented")