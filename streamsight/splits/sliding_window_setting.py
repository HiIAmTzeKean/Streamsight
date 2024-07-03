import logging
from typing import Optional

import numpy as np

from streamsight.matrix import InteractionMatrix
from streamsight.splits import Setting
from streamsight.splits.splitters import TimestampSplitter

logger = logging.getLogger(__name__)


class SlidingWindowSetting(Setting):

    def __init__(
        self,
        t: int,
        t_validation_delta: Optional[int] = None,
        delta_out: int = np.iinfo(np.int32).max,
        delta_in: int = np.iinfo(np.int32).max,
        validation: bool = False,
        window_size: int = np.iinfo(np.int32).max,  # in seconds
        seed: int | None = None,
    ):
        super().__init__(validation=validation, seed=seed)
        self.t = t
        self.delta_out = delta_out
        """Interval size to be used for out-sample data."""
        self.delta_in = delta_in
        """Interval size to be used for in-sample data."""
        self.t_validation_delta = t_validation_delta
        """Timestamp where the validation data is split from the training data."""
        self.window_size = window_size
        """Window size in seconds for spliiter to slide over the data."""

        if self.validation and not self.t_validation_delta:
            raise Exception(
                "t_validation_delta should be provided when requesting a validation dataset.")
        if self.t_validation_delta and self.t_validation_delta > self.window_size:
            logger.warning(
                "t_validation_delta should be smaller than window_size. Else validation data set will be the same as the training data set.")

        logger.info(
            f"Splitting data till time {t} with delta_in interval {delta_in} , delta_out interval {delta_out}, interval size of {window_size} seconds.")
        self.splitter = TimestampSplitter(window_size, delta_out, delta_in)
        if self.validation:
            # Override the validation splitter to a timed splitter.
            # set the validation time splitter to 0 as it will be updated in the split method
            self.validation_splitter = TimestampSplitter(
                0, delta_out, delta_in)

    def _split(self, data: InteractionMatrix):
        """Splits your dataset into a training, validation and test dataset
            based on the timestamp of the interaction.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """
        logger.debug(
            f"Staring split with window size {self.window_size} seconds")
        self._full_train_X_frame, self._test_data_out_frame, self._test_data_in_frame = [], [], []
        if self.validation:
            self._validation_train_X_frame, self._validation_data_out_frame, self._validation_data_in_frame = [], [], []

        for sub_time in range(self.window_size, self.t, self.window_size):
            logger.info(
                f"Splitting data at time {sub_time} with delta_in interval {self.delta_in} and delta_out interval {self.delta_out}")
            self.splitter.update_split_point(
                sub_time, self.delta_out, self.delta_in)
            data_in, data_out = self.splitter.split(data)
            self._full_train_X_frame.append(data_in)
            self._test_data_out_frame.append(data_out)
            self._test_data_in_frame.append(data_in.copy())

            if self.validation:
                self.validation_splitter.update_split_point(
                    sub_time-self.t_validation_delta, self.delta_out, self.delta_in)
                data_in, data_out = self.validation_splitter.split(data_in)
                self._validation_train_X_frame.append(data_in)
                self._validation_data_out_frame.append(data_out)
                self._validation_data_in_frame.append(data_in.copy())

        # update the number of folds
        self.num_split_set = len(self._full_train_X_frame)
        logger.debug(
            f"Finished split with window size {self.window_size} seconds. Number of splits: {self.num_split_set}")
