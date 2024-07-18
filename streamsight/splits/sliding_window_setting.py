import logging
from typing import List, Optional
from warnings import warn

import numpy as np

from streamsight.matrix import InteractionMatrix
from streamsight.splits import Setting
from streamsight.splits.splitters import TimestampSplitter

logger = logging.getLogger(__name__)

# TODO consider allowing unlabeled data and ground truth data to have different window size
class SlidingWindowSetting(Setting):
    """Sliding window setting
    
    background_data
    unlabeled_data
    ground_truth_data
    """

    def __init__(
        self,
        t: int,
        delta_out: int = np.iinfo(np.int32).max,
        delta_in: int = np.iinfo(np.int32).max,
        window_size: int = np.iinfo(np.int32).max,  # in seconds
        seed: int | None = None,
    ):
        super().__init__(seed=seed)
        self.t = t
        self.delta_out = delta_out
        """Interval size to be used for out-sample data."""
        self.delta_in = delta_in
        """Interval size to be used for in-sample data."""
        self.window_size = window_size
        """Window size in seconds for spliiter to slide over the data."""

        logger.info(
            f"Splitting data till time {t} with delta_in interval {delta_in} , delta_out interval {delta_out}, interval size of {window_size} seconds.")
        self.splitter = TimestampSplitter(t, delta_out, delta_in)

    def _split(self, data: InteractionMatrix):
        """Splits your dataset into a training, validation and test dataset
            based on the timestamp of the interaction.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """
        
        min_timestamp = data.min_timestamp

        
        if not data.has_timestamps:
            raise ValueError(
                "SingleTimePointSetting requires timestamp information in the InteractionMatrix.")
        if min_timestamp > self.t:
            warn(
                f"Splitting at time {self.t} is before the first timestamp in the data. No data will be in the training set.")

        self._background_data,remainder_data = self.splitter.split(data)
        self._ground_truth_data_frame, self._unlabeled_data_frame = [], []
        self._data_timestamp_limit = []

        # sub_time is the subjugate time point that the splitter will slide over the data
        sub_time = self.t
        self._data_timestamp_limit.append(sub_time)
        # we slide over the next point such that we start one window after
        # the background data in the unlabeled data
        sub_time += self.window_size
        
        #? use for loop instead?
        while sub_time < remainder_data.max_timestamp:
            self._data_timestamp_limit.append(sub_time)
            logger.debug(
                f"Sliding split t={sub_time},delta_in={self.delta_in},delta_out={self.delta_out}")
            
            self.splitter.update_split_point(
                sub_time, self.delta_out, self.delta_in)
            
            data_in, data_out = self.splitter.split(remainder_data)
            self._unlabeled_data_frame.append(data_in)
            self._ground_truth_data_frame.append(data_out)
            sub_time += self.window_size

        # update the number of folds
        self.num_split_set = len(self._unlabeled_data_frame)
        logger.info(
            f"Finished split with window size {self.window_size} seconds.\n"
                f"Number of splits: {self.num_split_set}")
