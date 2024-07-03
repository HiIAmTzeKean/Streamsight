import logging
from typing import Optional

import numpy as np

from streamsight.matrix import InteractionMatrix
from streamsight.splits import Setting
from streamsight.splits.splitters import TimestampSplitter

logger = logging.getLogger(__name__)


class SingleTimePointSetting(Setting):
    """Predict users' future interactions, given information about historical interactions.

    - :attr:`full_training_data` is constructed by using
      all interactions whose timestamps
      are in the interval ``[t - delta_in, t)``
    - :attr:`test_data_in` are events with timestamps in  ``[t - delta_in, t)``.
    - :attr:`test_data_out` are events with timestamps in ``[t, t + delta_out]``.
    - :attr:`validation_training_data` are all interactions
      with timestamps in ``[t_validation - delta_in, t_validation)``.
    - :attr:`validation_data_in` are interactions with timestamps in
      ``[t_validation - delta_in, t_validation)``
    - :attr:`validation_data_out` are interactions with timestamps in
      ``[t_validation, min(t, t_validation + delta_out)]``.

    .. warning::

        The scenario can only be used when the dataset has timestamp information.

    **Example**

    As an example, we split this data with ``t = 4``, ``t_validation = 2``
    ``delta_in = None (infinity)``, ``delta_out = 2``, and ``validation = True``::

        time    0   1   2   3   4   5   6
        Alice   X   X               X
        Bob         X   X   X   X
        Carol   X   X       X       X   X

    would yield full_training_data::

        time    0   1   2   3   4   5   6
        Alice   X   X
        Bob         X   X   X
        Carol   X   X       X

    validation_training_data::

        time    0   1   2   3   4   5   6
        Alice   X   X
        Bob         X
        Carol   X   X

    validation_data_in::

        time    0   1   2   3   4   5   6
        Bob         X
        Carol   X   X

    validation_data_out::

        time    0   1   2   3   4   5   6
        Bob             X   X
        Carol           X

    test_data_in::

        time    0   1   2   3   4   5   6
        Alice   X   X
        Carol   X   X       X

    test_data_out::

        time    0   1   2   3   4   5   6
        Alice                       X
        Carol                       X   X

    :param t: Timestamp to split target dataset :attr:`test_data_out`
        from the remainder of the data.
    :type t: int
    :param t_validation: Timestamp to split :attr:`validation_data_out`
        from :attr:`validation_training_data`. Required if validation is True.
    :type t_validation: int, optional
    :param delta_out: Size of interval in seconds for
        both :attr:`validation_data_out` and :attr:`test_data_out`.
        Both sets will contain interactions that occurred within ``delta_out`` seconds
        after the splitting timestamp.
        Defaults to maximal integer value (acting as infinity).
    :type delta_out: int, optional
    :param delta_in: Size of interval in seconds for
        :attr:`full_training_data`, :attr:`validation_training_data`,
        :attr:`validation_data_in` and :attr:`test_data_in`.
        All sets will contain interactions that occurred within ``delta_out`` seconds
        before the splitting timestamp.
        Defaults to maximal integer value (acting as infinity).
    :type delta_in: int, optional
    :param validation: Assign a portion of the full training dataset to validation data
        if True, else split without validation data
        into only a training and test dataset.
    :type validation: boolean, optional
    :param seed: Seed for randomisation parts of the scenario.
        Timed scenario is deterministic, so changing seed should not matter.
        Defaults to None, so random seed will be generated.
    :type seed: int, optional

    """

    def __init__(
        self,
        t: int,
        t_validation: Optional[int] = None,
        delta_out: int = np.iinfo(np.int32).max,
        delta_in: int = np.iinfo(np.int32).max,
        validation: bool = False,
        seed: int | None = None,
    ):
        super().__init__(validation=validation, seed=seed)
        self.t = t
        self.delta_out = delta_out
        """Interval size to be used for out-sample data."""
        self.delta_in = delta_in
        """Interval size to be used for in-sample data."""
        self.t_validation = t_validation
        """Timestamp where the validation data is split from the training data."""

        if self.validation and not self.t_validation:
            raise Exception(
                "t_validation should be provided when requesting a validation dataset.")
        if self.t_validation and not (self.t_validation < self.t):
            logger.warning(
                "t_validation should be smaller than t. Else there will be data leakage.")

        logger.info(
            f"Splitting data at time {t} with delta_in interval {delta_in} and delta_out interval {delta_out}")
        self.splitter = TimestampSplitter(t, delta_out, delta_in)

        if self.validation:
            # Override the validation splitter to a timed splitter.
            logger.info(
                f"Splitting data at time {t_validation} with delta_in interval {delta_in} and delta_out interval {delta_out}")
            self.validation_splitter = TimestampSplitter(
                t_validation, delta_out, delta_in)

    def _split(self, data: InteractionMatrix):
        """Splits your dataset into a training, validation and test dataset
            based on the timestamp of the interaction.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """
        self._full_train_X, self._test_data_out = self.splitter.split(data)
        self._test_data_in = self._full_train_X.copy()

        if self.validation:
            self._validation_train_X, self._validation_data_out = self.validation_splitter.split(
                self._full_train_X)
            self._validation_data_in = self._validation_train_X.copy()
