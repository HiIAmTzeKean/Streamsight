import logging

import pandas as pd
from streamsight.matrix.interation_matrix import InteractionMatrix
from streamsight.preprocessing.filter import Filter
from tqdm.auto import tqdm


tqdm.pandas()

logger = logging.getLogger(__name__)

class DataFramePreprocessor:
    def __init__(self, item_ix:str, user_ix:str, timestamp_ix:str):
        self._item_id_mapping = dict()
        self._user_id_mapping = dict()

        self.item_ix = item_ix
        self.user_ix = user_ix
        self.timestamp_ix = timestamp_ix

        self.filters = []

    def add_filter(self, _filter: Filter):
        """Add a preprocessing filter to be applied
        before transforming to a InteractionMatrix object.

        Filters are applied in order, different orderings can lead to different results!

        :param _filter: The filter to be applied
        :type _filter: Filter
        """
        self.filters.append(_filter)


    def process(self, df: pd.DataFrame) -> InteractionMatrix:

        logger.debug(f"\tinteractions before preprocess: {len(df.index)}")
        logger.debug(f"\titems before preprocess: {df[self.item_ix].nunique()}")
        logger.debug(f"\tusers before preprocess: {df[self.user_ix].nunique()}")

        for filter in self.filters:
            logger.debug(f"applying filter: {filter}")
            df = filter.apply(df)
            logger.debug(f"\tinteractions after filter: {len(df.index)}")
            logger.debug(f"\titems after filter: {df[self.item_ix].nunique()}")
            logger.debug(f"\tusers after filter: {df[self.user_ix].nunique()}")


        self._update_id_mappings(df)

        # Convert input data into internal data objects
        interaction_m = InteractionMatrix(
            df,
            self.item_ix,
            self.user_ix,
            self.timestamp_ix
        )
        
        return interaction_m

    def _update_id_mappings(self, df: pd.DataFrame):
        """
        Update the id mapping so we can combine multiple files
        """
        # sort by timestamp to incrementally assign user and item ids by timestamp
        df.sort_values(by=[self.timestamp_ix], inplace=True, ignore_index=True)
        user_index = pd.CategoricalIndex(df[self.user_ix],categories=df[self.user_ix].unique())
        self._user_id_mapping = dict(enumerate(user_index.drop_duplicates()))
        df[self.user_ix] = user_index.codes
        
        item_index = pd.CategoricalIndex(df[self.item_ix],categories=df[self.item_ix].unique())
        self._item_id_mapping = dict(enumerate(item_index.drop_duplicates()))
        df[self.item_ix] = item_index.codes

    @property
    def item_id_mapping(self) -> pd.DataFrame:
        """Pandas DataFrame containing mapping from original item IDs to internal (consecutive) item IDs as columns."""
        return pd.DataFrame.from_records(
            list(self._item_id_mapping.items()), columns=[self.item_ix, InteractionMatrix.ITEM_IX]
        )

    @property
    def user_id_mapping(self) -> pd.DataFrame:
        """Pandas DataFrame containing mapping from original user IDs to internal (consecutive) user IDs as columns."""
        return pd.DataFrame.from_records(
            list(self._user_id_mapping.items()), columns=[self.user_ix, InteractionMatrix.USER_IX]
        )