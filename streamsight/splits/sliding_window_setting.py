import logging
from typing import List, Optional
from warnings import warn

import numpy as np

from streamsight.matrix import InteractionMatrix
from streamsight.splits import Setting
from streamsight.splits.splitters import NPastInteractionTimestampSplitter, TimestampSplitter

logger = logging.getLogger(__name__)

# TODO consider allowing unlabeled data and ground truth data to have different window size


class SlidingWindowSetting(Setting):
    """Sliding window setting

    background_data ``[0, t)``
    unlabeled_data (n_prev_interaction, n_last_masked)
    ground_truth_data (n_last_masked)
    """

    def __init__(
        self,
        background_t: int,
        delta_out: int = np.iinfo(np.int32).max,
        delta_in: int = np.iinfo(np.int32).max,
        window_size: int = np.iinfo(np.int32).max,  # in seconds
        n_seq_data: int = 1,
        seed: int | None = None,
    ):
        super().__init__(seed=seed)
        self._sliding_window_setting = True
        self.t = background_t
        self.delta_out = delta_out
        """Interval size to be used for out-sample data."""
        self.delta_in = delta_in
        """Interval size to be used for in-sample data."""
        self.window_size = window_size
        """Window size in seconds for spliiter to slide over the data."""
        self.n_seq_data = n_seq_data
        """Number of last interactions to provide as unlabeled data for model to make prediction."""
        self._background_splitter = TimestampSplitter(
            background_t, delta_out, delta_in)
        self._window_splitter = NPastInteractionTimestampSplitter(
            background_t, window_size, n_seq_data)

    def _split(self, data: InteractionMatrix):
        """Splits dataset into a background, unlabeled and ground truth data.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """
        if not data.has_timestamps:
            raise ValueError(
                "SingleTimePointSetting requires timestamp information in the InteractionMatrix.")
        if data.min_timestamp > self.t:
            warn(
                f"Splitting at time {self.t} is before the first timestamp in the data. No data will be in the background(training) set.")
            
        self._background_data, remainder_data = self._background_splitter.split(data)
        self._ground_truth_data_frame, self._unlabeled_data_frame = [], []
        self._data_timestamp_limit = []

        # sub_time is the subjugate time point that the splitter will slide over the data
        sub_time = self.t
        max_timestamp = data.max_timestamp
        
        while sub_time < max_timestamp:
            self._data_timestamp_limit.append(sub_time)
            sub_time += self.window_size
            # the set used for eval will always have a timestamp greater than
            # data released such that it is unknown to the model
            logger.debug(
                f"Sliding split t={sub_time},delta_in={self.delta_in},delta_out={self.delta_out}")
            # TODO right now there is data leakage where the global_user and item base is leaked 
            self._window_splitter.update_split_point(sub_time)
            past_interaction, future_interaction = self._window_splitter.split(remainder_data)
            self._unlabeled_data_frame.append(past_interaction)
            self._ground_truth_data_frame.append(future_interaction)
            
        self._num_split_set = len(self._unlabeled_data_frame)
        logger.info(
            f"Finished split with window size {self.window_size} seconds. "
            f"Number of splits: {self._num_split_set}")
