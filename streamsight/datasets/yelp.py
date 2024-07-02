from streamsight.datasets.base import Dataset


class YelpDataset(Dataset):
    USER_IX = "userId"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "songId"
    """Name of the column in the DataFrame that contains item identifiers."""
    COUNT_IX = "playCount"
    """Name of the column in the DataFrame that contains how often an item was played."""
    
    DEFAULT_FILENAME = "yelp.csv"
    """Default filename that will be used if it is not specified by the user."""

    DATASET_URL = ""
    """URL to fetch the dataset from."""

