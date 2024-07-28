import logging
from typing import List, Optional
from warnings import warn

import numpy as np

from streamsight.matrix import InteractionMatrix
from streamsight.setting import Setting
from streamsight.setting.splitters import NPastInteractionTimestampSplitter, TimestampSplitter

logger = logging.getLogger(__name__)

# TODO consider allowing unlabeled data and ground truth data to have different window size


class SlidingWindowSetting(Setting):
    """Sliding window setting for splinting data into background and evaluation set.

    The data is split into a background set and evaluation set. The evaluation set is defined by a sliding window
    that moves over the data. The window size is defined by the ``window_size`` parameter. The evaluation set comprises of the
    unlabeled data and ground truth data stored in a list. The unlabeled data is the last ``n_seq_data`` interactions of the users/item before the
    split point. The ground truth data is the interactions after the split point and spans ``window_size`` seconds.


    :param background_t: Time point to split the data into background and evaluation data. Split will be from ``[0, t)``
    :type background_t: int
    :param window_size: Size of the window in seconds to slide over the data.
    :type window_size: int, optional
    :param n_seq_data: Number of last interactions to provide as unlabeled data for model to make prediction.
    :type n_seq_data: int, optional
    :param seed: Seed for random number generator.
    :type seed: int, optional
    """

    def __init__(
        self,
        background_t: int,
        window_size: int = np.iinfo(np.int32).max,  # in seconds
        n_seq_data: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._sliding_window_setting = True
        self.t = background_t
        # ? is there a value in putting this here rather then in the loader
        # self.min_timestamp_threshold = min_timestamp_threshold
        # """Minimum timestamp value for the overall data to be used."""
        # self.max_timestamp_threshold = max_timestamp_threshold
        # """Maximum timestamp value for the overall data to be used."""
        self.window_size = window_size
        """Window size in seconds for spliiter to slide over the data."""
        self.n_seq_data = n_seq_data
        """Number of last interactions to provide as unlabeled data for model to make prediction."""

        self._background_splitter = TimestampSplitter(
            background_t, None, None)
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

        self._background_data, _ = self._background_splitter.split(data)
        self._ground_truth_data_frame, self._unlabeled_data_frame = [], []
        self._data_timestamp_limit = []

        # sub_time is the subjugate time point that the splitter will slide over the data
        sub_time = self.t
        max_timestamp = data.max_timestamp

        while sub_time < max_timestamp:
            self._data_timestamp_limit.append(sub_time)
            # the set used for eval will always have a timestamp greater than
            # data released such that it is unknown to the model
            # TODO right now there is data leakage where the global_user and item base is leaked
            self._window_splitter.update_split_point(sub_time)
            past_interaction, future_interaction = self._window_splitter.split(
                data)
            self._unlabeled_data_frame.append(past_interaction)
            self._ground_truth_data_frame.append(future_interaction)
            sub_time += self.window_size

        self._num_split_set = len(self._unlabeled_data_frame)
        logger.info(
            f"Finished split with window size {self.window_size} seconds. "
            f"Number of splits: {self._num_split_set}")
