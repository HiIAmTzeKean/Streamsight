import logging
import os
from abc import ABC, abstractmethod
from typing import ClassVar

import httpx
import pandas as pd

from .config import DatasetConfig


logger = logging.getLogger(__name__)


class DataFetcher(ABC):
    """Represents a abstract class to be used by Dataset or Metadata subclass.
    """

    config: ClassVar[DatasetConfig] = DatasetConfig()
    """Configuration for the dataset."""

    @property
    def name(self) -> str:
        """Name of the object's class."""
        return self.__class__.__name__

    @property
    def file_path(self) -> str:
        """File path of the dataset."""
        return os.path.join(self.config.default_base_path, self.config.default_filename)

    @property
    def processed_cache_path(self) -> str:
        """Path for cached processed data."""
        return os.path.join(
            self.config.default_base_path, f"{self.config.default_filename}.processed.parquet"
        )

    def fetch_dataset(self) -> None:
        """Check if dataset is present, if not download"""
        if os.path.exists(self.file_path):
            logger.debug("Data file is in memory and in dir specified.")
            return
        logger.debug(f"{self.name} dataset not found in {self.file_path}.")
        self._download_dataset()

    def fetch_dataset_force(self) -> None:
        """Force re-download of the dataset."""
        logger.debug(f"{self.name} force re-download of dataset.")
        self._download_dataset()

    def _fetch_remote(self, url: str, filename: str) -> str:
        """Fetch data from remote url and save locally (synchronous fallback).

        This function keeps the previous synchronous behaviour but uses
        `httpx.Client` to stream the response and write to disk. If you
        want async behaviour, use :meth:`_fetch_remote_async` instead.

        :param url: url to fetch data from
        :param filename: Path to save file to
        :return: The filename where data was saved
        """
        logger.debug(f"{self.name} will fetch dataset from remote url at {url}.")

        with httpx.Client(timeout=httpx.Timeout(60.0)) as client, client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(filename, "wb") as fd:
                for chunk in resp.iter_bytes():
                    if chunk:
                        fd.write(chunk)

        return filename

    async def _fetch_remote_async(self, url: str, filename: str) -> str:
        """Asynchronously fetch data from a remote URL and save locally.

        Uses `httpx.AsyncClient` and streams the response to disk. Callers
        running inside an event loop should use this coroutine instead of
        the synchronous `_fetch_remote`.

        :param url: url to fetch data from
        :param filename: Path to save file to
        :return: The filename where data was saved
        """
        logger.debug(f"{self.name} will asynchronously fetch dataset from {url}.")

        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client,
            client.stream("GET", url) as resp,
        ):
            resp.raise_for_status()
            # Write bytes as they arrive
            with open(filename, "wb") as fd:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        fd.write(chunk)

        return filename

    def _cache_processed_dataframe(self, df: pd.DataFrame) -> None:
        """Cache the processed DataFrame to disk.

        :param df: DataFrame to cache
        :type df: pd.DataFrame
        """
        logger.debug(f"Caching processed DataFrame to {self.processed_cache_path}")
        df.to_parquet(self.processed_cache_path)
        logger.debug("Processed DataFrame cached successfully.")

    @abstractmethod
    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::
            This does not apply any preprocessing, and returns the raw dataset.

        :return: Interation with minimal columns of {user, item, timestamp}.
        :rtype: pd.DataFrame
        """
        raise NotImplementedError("Needs to be implemented")

    @abstractmethod
    def _download_dataset(self) -> None:
        """Downloads the dataset.

        Downloads the csv file from the dataset URL and saves it to the file path.
        """
        raise NotImplementedError("Needs to be implemented")
