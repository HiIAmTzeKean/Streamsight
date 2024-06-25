from streamsight.datasets.base import Dataset


class MovieLensDataset(Dataset):
    """Base class for Movielens datasets.
    
    Other Movielens datasets should inherit from this class.
    """
    def __init__(self):

        USER_IX = "userId"
        """Name of the column in the DataFrame that contains user identifiers."""
        ITEM_IX = "movieId"
        """Name of the column in the DataFrame that contains item identifiers."""
        TIMESTAMP_IX = "timestamp"
        """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""
        RATING_IX = "rating"
        """Name of the column in the DataFrame that contains the rating a user gave to the item."""

        DATASETURL = "http://files.grouplens.org/datasets/movielens"

        REMOTE_ZIPNAME = ""
        """Name of the zip-file on the MovieLens server."""

        REMOTE_FILENAME = "ratings.csv"
        """Name of the file containing user ratings on the MovieLens server."""
        
        self.url = DATASETURL + "/ml-latest-small.zip"
        logger.debug("MovieLensDataset initialized.")
        
    def download(self):
        """Download dataset into directory defined
        """
        pass
    def sort(self):
        """Sorts the dataset
        """
        pass

class MovieLens100K(MovieLensDataset):
    """Handles Movielens 100K dataset.

    All information on the dataset can be found at https://grouplens.org/datasets/movielens/100k/.
    Uses the `u.data` file to generate an interaction matrix.

    Default processing  is done as in "Variational autoencoders for collaborative filtering." Liang, Dawen, et al.:

    - Ratings above or equal to 4 are interpreted as implicit feedback
    - Each remaining item has been interacted with by at least 5 users

    To use another value as minimal rating to mark interaction as positive,
    you have to manually set the preprocessing filters.::

        from recpack.preprocessing.filters import MinRating, MinItemsPerUser, MinUsersPerItem
        from recpack.datasets import MovieLens100K
        d = MovieLens100K(path='path/to/', use_default_filters=False)
        d.add_filter(MinRating(3, d.RATING_IX, 3))
        d.add_filter(MinItemsPerUser(3, d.ITEM_IX, d.USER_IX))
        d.add_filter(MinUsersPerItem(5, d.ITEM_IX, d.USER_IX))

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised? Defaults to True
    :type use_default_filters: bool, optional

    """

    REMOTE_FILENAME = "u.data"
    REMOTE_ZIPNAME = "ml-100k"

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """

        self.fetch_dataset()
        df = pd.read_table(
            self.file_path,
            dtype={
                self.USER_IX: np.int64,
                self.ITEM_IX: np.int64,
                self.RATING_IX: np.float64,
                self.TIMESTAMP_IX: np.int64,
            },
            sep="\t",
            names=[self.USER_IX, self.ITEM_IX, self.RATING_IX, self.TIMESTAMP_IX],
        )

        return df