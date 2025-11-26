import logging
import os
import zipfile
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
from tqdm import tqdm

from streamsight.datasets.base import Dataset
from streamsight.datasets.config import DatasetConfig
from streamsight.metadata.movielens import MovieLens100kItemMetadata, MovieLens100kUserMetadata


logger = logging.getLogger(__name__)


@dataclass
class MovieLensConfig(DatasetConfig):
    """MovieLens base configuration."""

    user_ix: str = "userId"
    item_ix: str = "movieId"
    timestamp_ix: str = "timestamp"
    rating_ix: str = "rating"
    dataset_url: str = "https://files.grouplens.org/datasets/movielens"
    remote_zipname: str = "ml-100k"
    remote_filename: str = "ratings.csv"


@dataclass
class MovieLens100KConfig(MovieLensConfig):
    """MovieLens 100K specific configuration."""

    remote_filename: str = "u.data"


class MovieLensDataset(Dataset):
    """Base class for Movielens datasets.

    Other Movielens datasets should inherit from this class.

    This code is adapted from RecPack :cite:`recpack`
    """

    USER_IX: ClassVar[str] = "userId"
    ITEM_IX: ClassVar[str] = "movieId"
    TIMESTAMP_IX: ClassVar[str] = "timestamp"
    RATING_IX: ClassVar[str] = "rating"
    """Name of the column in the DataFrame that contains the rating a user gave to the item."""
    DATASET_URL: ClassVar[str] = "https://files.grouplens.org/datasets/movielens"
    REMOTE_ZIPNAME: ClassVar[str] = "ml-100k"
    """Name of the zip-file on the MovieLens server."""
    REMOTE_FILENAME: ClassVar[str] = "ratings.csv"
    """Name of the file containing user ratings on the MovieLens server."""

    @classmethod
    def DEFAULT_FILENAME(cls) -> str:
        return f"{cls.REMOTE_ZIPNAME}_{cls.REMOTE_FILENAME}"

    def _download_dataset(self) -> None:
        # Download the zip into the data directory
        self._fetch_remote(
            url=f"{self.__class__.DATASET_URL}/{self.__class__.REMOTE_ZIPNAME}.zip",
            filename=os.path.join(self.__class__.DEFAULT_BASE_PATH, f"{self.__class__.REMOTE_ZIPNAME}.zip"),
        )

        # Extract the ratings file which we will use
        with zipfile.ZipFile(
            os.path.join(self.__class__.DEFAULT_BASE_PATH, f"{self.__class__.REMOTE_ZIPNAME}.zip"), "r"
        ) as zip_ref:
            zip_ref.extract(f"{self.__class__.REMOTE_ZIPNAME}/{self.__class__.REMOTE_FILENAME}", self.__class__.DEFAULT_BASE_PATH)

        # Rename the ratings file to the specified filename
        os.rename(
            os.path.join(self.__class__.DEFAULT_BASE_PATH, f"{self.__class__.REMOTE_ZIPNAME}/{self.__class__.REMOTE_FILENAME}"),
            self.file_path,
        )


class MovieLens100K(MovieLensDataset):
    """MovieLens 100K dataset."""

    REMOTE_ZIPNAME = "ml-100k"
    REMOTE_FILENAME = "u.data"
    ITEM_METADATA = None
    USER_METADATA = None

    def _load_dataframe(self) -> pd.DataFrame:
        self.fetch_dataset()
        chunks = pd.read_table(
            self.file_path,
            dtype={
                self.__class__.USER_IX: np.int64,
                self.__class__.ITEM_IX: np.int64,
                self.__class__.RATING_IX: np.float64,
                self.__class__.TIMESTAMP_IX: np.int64,
            },
            sep="\t",
            names=[
                self.__class__.USER_IX,
                self.__class__.ITEM_IX,
                self.__class__.RATING_IX,
                self.__class__.TIMESTAMP_IX,
            ],
            chunksize=100_000,
        )
        df = pd.concat(
            [chunk for chunk in tqdm(chunks, desc="Reading table", unit="chunk")], ignore_index=True
        )
        return df

    def _fetch_dataset_metadata(
        self, user_id_mapping: pd.DataFrame, item_id_mapping: pd.DataFrame
    ) -> None:
        self.USER_METADATA = MovieLens100kUserMetadata(user_id_mapping=user_id_mapping).load()
        self.ITEM_METADATA = MovieLens100kItemMetadata(item_id_mapping=item_id_mapping).load()
