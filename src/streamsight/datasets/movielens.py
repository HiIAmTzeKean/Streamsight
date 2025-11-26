import logging
import os
import zipfile
from typing import ClassVar

import numpy as np
import pandas as pd
from tqdm import tqdm

from streamsight.datasets.base import Dataset
from streamsight.metadata.movielens import MovieLens100kItemMetadata, MovieLens100kUserMetadata


logger = logging.getLogger(__name__)


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

    @property
    def DEFAULT_FILENAME(self) -> str:
        return f"{self.REMOTE_ZIPNAME}_{self.REMOTE_FILENAME}"

    def _download_dataset(self) -> None:
        # Download the zip into the data directory
        self._fetch_remote(
            url=f"{self.DATASET_URL}/{self.REMOTE_ZIPNAME}.zip",
            filename=os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"),
        )

        # Extract the ratings file which we will use
        with zipfile.ZipFile(
            os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"), "r"
        ) as zip_ref:
            zip_ref.extract(f"{self.REMOTE_ZIPNAME}/{self.REMOTE_FILENAME}", self.base_path)

        # Rename the ratings file to the specified filename
        os.rename(
            os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}/{self.REMOTE_FILENAME}"),
            self.file_path,
        )


class MovieLens100K(MovieLensDataset):
    """MovieLens 100K dataset."""

    REMOTE_FILENAME = "u.data"
    REMOTE_ZIPNAME = "ml-100k"
    ITEM_METADATA = None
    USER_METADATA = None

    def _load_dataframe(self) -> pd.DataFrame:
        self.fetch_dataset()
        chunks = pd.read_table(
            self.file_path,
            dtype={
                self.USER_IX: np.int64,
                self.ITEM_IX: np.int64,
                self.RATING_IX: np.float64,
                self.TIMESTAMP_IX: np.int64,
            },
            sep="\t",
            names=[
                self.USER_IX,
                self.ITEM_IX,
                self.RATING_IX,
                self.TIMESTAMP_IX,
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
