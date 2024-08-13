from abc import ABC, abstractmethod

import pandas as pd


class Filter(ABC):
    """Abstract baseclass for filter implementations

    A filter needs to implement an ``apply`` method,
    which takes as input a pandas DataFrame, and returns a processed pandas DataFrame.
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Filter to the DataFrame passed.

        :param df: DataFrame to filter
        :type df: pd.DataFrame
        """
        raise NotImplementedError

    def __str__(self):
        attrs = self.__dict__
        return f"{self.__class__.__name__}({', '.join((f'{k}={v}' for k, v in attrs.items()))})"


class MinItemsPerUser(Filter):
    """Require that a user has interacted with a minimum number of items.

    :param min_items_per_user: Minimum number of items required.
    :type min_items_per_user: int
    :param item_ix: Name of the column in which item identifiers are listed.
    :type item_ix: str
    :param user_ix: Name of the column in which user identifiers are listed.
    :type user_ix: str
    :param count_duplicates: Count multiple interactions with the same item, defaults to False
    :type count_duplicates: bool
    """

    def __init__(
        self,
        min_items_per_user: int,
        item_ix: str,
        user_ix: str,
        count_duplicates: bool = False,
    ):
        self.min_iu = min_items_per_user
        self.count_duplicates = count_duplicates

        self.item_ix = item_ix
        self.user_ix = user_ix

    def apply(self, df):
        uids = (
            df[self.user_ix]
            if self.count_duplicates
            else df.drop_duplicates([self.user_ix, self.item_ix])[self.user_ix]
        )
        cnt_items_per_user = uids.value_counts()
        users_of_interest = list(
            cnt_items_per_user[cnt_items_per_user >= self.min_iu].index
        )

        return df[df[self.user_ix].isin(users_of_interest)].copy()
