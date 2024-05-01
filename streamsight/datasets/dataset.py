import logging
import os
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from streamsight.matrix.interation_matrix import InteractionMatrix

"""
The purpose of dataset is to provide meta data and to contain the data of the
dataset that we are interested in. It will provide the specific details such as
url to dataset and the configurations to load the dataset.

To support the incremental training of the model, the class will contain 2 attr,
`train_set` and `test_set`. These sets are provided to the recommender to be
trained and tested on.
"""

logger = logging.getLogger(__name__)

class Dataset():
    """Represents a collaborative filtering dataset. Dataset must minimmally contain
    user, item and timestamp columns.

    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
        If the dataset does not have a default filename, a ValueError will be raised.
    :param base_path: The base_path to the data directory.
        Defaults to `data`
    :type filename: str, optional
    :type base_path: str, optional
    """
    
    USER_IX = None
    """Name of the column in the DataFrame with user identifiers"""
    ITEM_IX = None
    """Name of the column in the DataFrame with item identifiers"""
    TIMESTAMP_IX = None
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""

    DEFAULT_FILENAME = None
    """Default filename that will be used if it is not specified by the user."""
    
    DEFAULT_BASE_PATH = "data"
    """Default base path where the dataset will be stored."""
    
    def __init__(self, filename: str = None, base_path: str = None):
        self.base_path = base_path if base_path else self.DEFAULT_BASE_PATH
        logger.debug(f"Dataset {__class__.__name__} initialized with base_path {self.base_path}")
        
        self.filename = filename if filename else self.DEFAULT_FILENAME
        if not self.filename:
            raise ValueError("No filename specified, and no default known.")
        
        self._check_safe()
        logger.debug("Dataset class initialized.")
        
    @property
    def file_path(self) -> str:
        """File path of the dataset."""
        return os.path.join(self.base_path, self.filename)
    
    def _check_safe(self):
        """Check if the directory is safe. If directory does not exit, create it."""
        p = Path(self.base_path)
        p.mkdir(exist_ok=True)
        
    def fetch_dataset(self, force=False) -> None:
        """Check if dataset is present, if not download

        :param force: If True, dataset will be downloaded,
                even if the file already exists.
                Defaults to False.
        :type force: bool, optional
        """
        if not os.path.exists(self.file_path) or force:
            self._download_dataset()
            
    def load(self) -> InteractionMatrix:
        """Loads data into an InteractionMatrix object.

        Data is loaded into a DataFrame using the `_load_dataframe` function.
        Resulting DataFrame is parsed into an `InteractionMatrix` object.

        :return: The resulting InteractionMatrix
        :rtype: InteractionMatrix
        """
        logger.info(f"{self.__class__.__name__} loading dataset.")
        df = self._load_dataframe()
        im = self._dataframe_to_matrix(df)
        logger.info(f"{self.__class__.__name__} dataset loaded.")
        return im
    
    def _dataframe_to_matrix(self, df: pd.DataFrame) -> InteractionMatrix:
        """Converts a DataFrame to an InteractionMatrix.

        :param df: DataFrame to convert
        :type df: pd.DataFrame
        :return: InteractionMatrix object
        :rtype: InteractionMatrix
        """
        if not self.USER_IX or not self.ITEM_IX or not self.TIMESTAMP_IX:
            raise ValueError("USER_IX, ITEM_IX or TIMESTAMP_IX not set.")
        return InteractionMatrix(
            df,
            user_ix=self.USER_IX,
            item_ix=self.ITEM_IX,
            timestamp_ix=self.TIMESTAMP_IX,
        )
        
    def _fetch_remote(self, url: str, filename: str) -> str:
        """Fetch data from remote url and save locally

        :param url: url to fetch data from
        :type url: str
        :param filename: Path to save file to
        :type filename: str
        :return: The filename where data was saved
        :rtype: str
        """
        urlretrieve(url, filename)
        return filename

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::
            This does not apply any preprocessing, and returns the raw dataset.

        :return: Interation with minimal columns of {user, item, timestamp}.
        :rtype: pd.DataFrame
        """
        raise NotImplementedError("Needs to be implemented")
    
    def _download_dataset(self):
        raise NotImplementedError("Needs to be implemented")