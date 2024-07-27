from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import Generator, List, Optional, Tuple, Union
from warnings import warn

from streamsight.matrix import InteractionMatrix
from streamsight.splits.util import FrameExpectedError, SeriesExpectedError

logger = logging.getLogger(__name__)


def check_split_complete(func):
    def check_split_for_func(self):
        if not self.is_ready:
            raise KeyError(
                f"Split before trying to access {func.__name__} property.")
        return func(self)
    return check_split_for_func


class Setting(ABC):
    """Base class for defining an evaluation setting.

    A setting is a set of steps that splits data into training,
    validation and test datasets.
    The test dataset is made up of two components:
    a fold-in set of interactions that is used to predict another held-out
    set of interactions.
    The creation of the validation dataset, from the full training dataset,
    should follow the same splitting strategy as
    the one used to create training and test datasets from the full dataset.

    :param seed: Seed for randomisation parts of the setting.
        Defaults to None, so random seed will be generated.
    :type seed: int, optional
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            # Set seed if it was not set before.
            seed = np.random.get_state()[1][0]
        self.seed = seed
        self._num_split_set = 1
        self._sliding_window_setting = False
        """Number of splits created from sliding window. Defaults to 1 (no splits on training set)."""
        self._num_full_interactions: int
        self._split_complete = False
        self._unlabeled_data_series: InteractionMatrix
        self._unlabeled_data_frame: List[InteractionMatrix]
        self._ground_truth_data_series: InteractionMatrix
        self._ground_truth_data_frame: List[InteractionMatrix]
        self._background_data: InteractionMatrix
        self._data_timestamp_limit: Union[int, List[int]]
        """This is the limit of the data in the corresponding split."""

    @abstractmethod
    def _split(self, data_m: InteractionMatrix) -> None:
        """Abstract method to be implemented by the scenarios.

        Splits the data and assigns to :attr:``background_data``,
        :attr:``ground_truth_data``, :attr:``unlabeled_data``

        :param data_m: Interaction matrix to be split.
        :type data_m: InteractionMatrix
        """

    def split(self, data_m: InteractionMatrix) -> None:
        """Splits ``data_m`` according to the scenario.

        After splitting properties :attr:``training_data``,
        :attr:``validation_data`` and :attr:``test_data`` can be used
        to retrieve the splitted data.

        :param data_m: Interaction matrix that should be split.
        :type data: InteractionMatrix
        """
        self._num_full_interactions = data_m.num_interactions
        self._split(data_m)

        logger.debug("Checking split attribute and sizes.")
        self._check_split()

        self._split_complete = True

    @property
    @check_split_complete
    def background_data(self) -> InteractionMatrix:
        """Background data that can be provided for the model for the initial training.

        :return: Interaction Matrix of training interactions.
        :rtype: InteractionMatrix
        """
        return self._background_data

    @property
    @check_split_complete
    def data_timestamp_limit(self) -> Union[int, List[int]]:
        """The timestamp limit of the data in the corresponding split.

        :return: Timestamp limit of the data in the corresponding split.
        :rtype: Union[int, List[int]]
        """
        return self._data_timestamp_limit

    @property
    def num_split(self) -> int:
        """Number of splits created from sliding window. Defaults to 1 (no splits on training set)."""
        return self._num_split_set

    @property
    def is_ready(self):
        """Flag on setting if it is ready to be used for evaluation."""
        return self._split_complete

    @property
    @check_split_complete
    def unlabeled_data(self) -> Union[InteractionMatrix, List[InteractionMatrix]]:
        if not self._sliding_window_setting:
            return self._unlabeled_data_series
        return self._unlabeled_data_frame

    @property
    @check_split_complete
    def ground_truth_data(self) -> Union[InteractionMatrix, List[InteractionMatrix]]:
        if not self._sliding_window_setting:
            return self._ground_truth_data_series
        return self._ground_truth_data_frame

    @property
    def _evaluation_data_frame(self) -> List[Tuple[InteractionMatrix, InteractionMatrix]]:
        if not self._sliding_window_setting:
            raise SeriesExpectedError()

        datasets = []
        for dataset_idx in range(self._num_split_set):
            datasets.append(
                (self._unlabeled_data_frame[dataset_idx], self._ground_truth_data_frame[dataset_idx]))
        return datasets

    @property
    def _evaluation_data_series(self) -> Tuple[InteractionMatrix, InteractionMatrix]:
        return (
            self._unlabeled_data_series,
            self._ground_truth_data_series
        )

    @property
    @check_split_complete
    def evaluation_data(self) -> Union[Tuple[InteractionMatrix, InteractionMatrix], List[Tuple[InteractionMatrix, InteractionMatrix]]]:
        """The evaluation dataset. Consist of the unlabled data and ground truth set of interactions.

        :return: Test data matrices as InteractionMatrix in, InteractionMatrix out.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        # make sure users match.
        if not self._sliding_window_setting:
            return self._evaluation_data_series
        else:
            return self._evaluation_data_frame

    def _check_split(self):
        """Checks that the splits have been done properly.

        Makes sure all expected attributes are set.
        """
        logger.debug("Checking split attributes.")
        assert (hasattr(self, "_background_data")
                and self._background_data is not None)

        assert (hasattr(self, "_unlabeled_data_series") and self._unlabeled_data_series is not None) \
            or (hasattr(self, "_unlabeled_data_frame") and self._unlabeled_data_frame is not None)

        assert (hasattr(self, "_ground_truth_data_series") and self._ground_truth_data_series is not None) \
            or (hasattr(self, "_ground_truth_data_frame") and self._ground_truth_data_frame is not None)
        logger.debug("Split attributes are set.")

        self._check_size()

    def _check_size(self):
        """
        Warns user if any of the sets is unusually small or empty
        """
        logger.debug("Checking size of split sets.")

        def check_ratio(name, count, total, threshold):
            if (count + 1e-9) / (total + 1e-9) < threshold:
                warn(
                    f"{name} resulting from {type(self).__name__} is unusually small.")

        def check_empty(name, count):
            if count == 0:
                warn(
                    f"{name} resulting from {type(self).__name__} is empty (no interactions).")

        n_background = self._background_data.num_interactions
        check_empty("Background set", n_background)
        check_ratio("Background set", n_background,
                    self._num_full_interactions, 0.05)

        # TODO check if len of ground truth and unlabeled is the same
        if not self._sliding_window_setting:
            n_unlabel = self._unlabeled_data_series.num_interactions
            n_ground_truth = self._ground_truth_data_series.num_interactions

            check_empty("Unlabeled set", n_unlabel)
            check_empty("Ground truth set", n_ground_truth)
            check_ratio("Ground truth set", n_ground_truth, n_unlabel, 0.05)

        else:
            for dataset_idx in range(self._num_split_set):
                n_unlabel = self._unlabeled_data_frame[dataset_idx].num_interactions
                n_ground_truth = self._ground_truth_data_frame[dataset_idx].num_interactions

                check_empty(f"Unlabeled set[{dataset_idx}]", n_unlabel)
                check_empty(f"Ground truth set[{dataset_idx}]", n_ground_truth)
        logger.debug("Size of split sets are checked.")

    def _unlabeled_data_generator(self):
        self.unlabeled_data_iter: Generator[InteractionMatrix] = self._create_generator(
            "_unlabeled_data_series", "_unlabeled_data_frame")

    def _create_generator(self, series: str, frame: str):
        """Creates generator for provided series or frame attribute name

        :param series: _description_
        :type series: str
        :param frame: _description_
        :type frame: str
        :yield: _description_
        :rtype: _type_
        """
        if not self._sliding_window_setting:
            yield getattr(self, series)
        else:
            for data in getattr(self, frame):
                yield data

    def next_unlabeled_data(self, reset=False) -> Optional[InteractionMatrix]:
        # Create generator if it does not exist or reset is True
        if reset or not hasattr(self, "unlabeled_data_iter"):
            self._unlabeled_data_generator()

        try:
            return next(self.unlabeled_data_iter)
        except StopIteration:
            logger.debug(
                "End of unlabeled data reached. To reset, set reset=True")
            return None

    def _ground_truth_data_generator(self):
        self.ground_truth_data_iter: Generator[InteractionMatrix] = self._create_generator(
            "_ground_truth_data_series", "_ground_truth_data_frame")

    def next_ground_truth_data(self, reset=False) -> Optional[InteractionMatrix]:
        # Create generator if it does not exist or reset is True
        if reset or not hasattr(self, "ground_truth_data_iter"):
            self._ground_truth_data_generator()

        try:
            return next(self.ground_truth_data_iter)
        except StopIteration:
            logger.debug(
                "End of ground truth data reached. To reset, set reset=True")
            return None

    # TODO consider better naming
    def _next_data_timestamp_limit_generator(self):
        self.data_timestamp_limit_iter: Generator[int] = self._create_generator(
            "_data_timestamp_limit", "_data_timestamp_limit")

    def next_data_timestamp_limit(self, reset=False):
        if reset or not hasattr(self, "data_timestamp_limit_iter"):
            self._next_data_timestamp_limit_generator()

        try:
            return next(self.data_timestamp_limit_iter)
        except StopIteration:
            logger.debug("End of data timestamp_limit reached.")
            warn("Reset the generators by calling reset_data_generators()")
            return None

    def reset_data_generators(self):
        logger.info("Resetting data generators.")
        self._unlabeled_data_generator()
        self._ground_truth_data_generator()
        self._next_data_timestamp_limit_generator()
        logger.info("Data generators are reset.")
