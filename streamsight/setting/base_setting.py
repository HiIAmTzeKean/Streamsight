import logging
from abc import ABC, abstractmethod
from typing import Generator, List, Literal, Optional, Tuple, Union
from warnings import warn

import numpy as np

from streamsight.matrix import InteractionMatrix, ItemUserBasedEnum
from streamsight.setting.processor import PredictionDataProcessor

logger = logging.getLogger(__name__)

class EOWSetting(Exception):
    """End of Window Setting Exception."""
    def __init__(self, message="End of window setting reached. Call reset_data_generators() to start again."):
        self.message = message
        super().__init__(self.message)

def check_split_complete(func):
    def check_split_for_func(self):
        if not self.is_ready:
            raise KeyError(
                f"Split before trying to access {func.__name__} property.")
        return func(self)
    return check_split_for_func


class Setting(ABC):
    """Base class for defining an evaluation setting.

    :param seed: Seed for randomisation parts of the setting.
        Defaults to None, so random seed will be generated.
    :type seed: int, optional
    :param item_user_based: Item or User based setting.
        Defaults to "user".
    :type item_user_based: Literal['user', 'item'], optional
    :raises ValueError: Invalid value for item_user_based
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        item_user_based: Literal['user', 'item'] = "user"
    ):
        if seed is None:
            # Set seed if it was not set before.
            seed = np.random.get_state()[1][0]
        self.seed = seed
        
        # self._check_valid_item_user_based(item_user_based)
        if not ItemUserBasedEnum.has_value(item_user_based):
            raise ValueError(f"Invalid value for item_user_based: {item_user_based}")
        self._item_user_based = ItemUserBasedEnum(item_user_based)
        self.prediction_data_processor = PredictionDataProcessor(self._item_user_based)
        
        self._num_split_set = 1
        
        self._sliding_window_setting = False
        self._split_complete = False
        """Number of splits created from sliding window. Defaults to 1 (no splits on training set)."""
        self._num_full_interactions: int
        self._unlabeled_data_series: InteractionMatrix
        self._unlabeled_data_frame: List[InteractionMatrix]
        self._ground_truth_data_series: InteractionMatrix
        self._ground_truth_data_frame: List[InteractionMatrix]
        self._incremental_data_frame: List[InteractionMatrix]
        """Data that is used to incrementally update the model. Unique to sliding window setting."""
        self._background_data: InteractionMatrix
        self._data_timestamp_limit: Union[int, List[int]]
        """This is the limit of the data in the corresponding split."""
        self.n_seq_data: int
        """Number of last sequential interactions to provide as unlabeled data for model to make prediction."""
        self.top_K: int
        """Number of interaction per user that should be selected for evaluation purposes."""
        
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
        logger.info("Splitting data...")
        self._num_full_interactions = data_m.num_interactions
        self._split(data_m)

        logger.debug("Checking split attribute and sizes.")
        self._check_split()

        self._split_complete = True
        
    @property
    def is_item_user_based(self) -> str:
        """Item or User based setting.

        :return: Item or User based setting.
        :rtype: ItemUserBasedEnum
        """
        return self._item_user_based.value
    
    @property
    def is_sliding_window_setting(self) -> bool:
        """Flag to indicate if the setting is a sliding window setting."""
        return self._sliding_window_setting
    
    @property
    @check_split_complete
    def background_data(self) -> InteractionMatrix:
        """Background data provided for the model for the initial training.
        
        This data is used as the initial set of interactions to train the model.

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
        """Unlabeled data for the model to make predictions on.
        
        Contains the user/item ID for prediction along with previous sequential
        interactions of user-item on items if it exists. This data is used to
        make predictions on the ground truth data.

        :return: Either a single InteractionMatrix or a list of InteractionMatrix
            if the setting is a sliding window setting.
        :rtype: Union[InteractionMatrix, List[InteractionMatrix]]
        """
        if not self._sliding_window_setting:
            return self._unlabeled_data_series
        return self._unlabeled_data_frame

    @property
    @check_split_complete
    def ground_truth_data(self) -> Union[InteractionMatrix, List[InteractionMatrix]]:
        """Ground truth data to evaluate the model's predictions on.
        
        Contains the actual interactions of the user-item interaction that the
        model is supposed to predict.

        :return: _description_
        :rtype: Union[InteractionMatrix, List[InteractionMatrix]]
        """
        if not self._sliding_window_setting:
            return self._ground_truth_data_series
        return self._ground_truth_data_frame

    @property
    @check_split_complete
    def incremental_data(self) -> List[InteractionMatrix]:
        """Data that is used to incrementally update the model.

        Unique to sliding window setting.

        :return: _description_
        :rtype: List[InteractionMatrix]
        """
        if not self._sliding_window_setting:
            raise AttributeError(
                "Incremental data is only available for sliding window setting.")
        return self._incremental_data_frame

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
        """Creates generator for data
        
        ==
        Note
        ===
        A private method is specifically created to abstract the creation of
        the generator and to allow for easy resetting when needed.
        """
        self.unlabeled_data_iter: Generator[InteractionMatrix] = self._create_generator(
            "_unlabeled_data_series", "_unlabeled_data_frame")
    def _incremental_data_generator(self):
        """Creates generator for data
        
        ==
        Note
        ===
        A private method is specifically created to abstract the creation of
        the generator and to allow for easy resetting when needed.
        """
        self.incremental_data_iter: Generator[InteractionMatrix] = self._create_generator(
            "_incremental_data_frame", "_incremental_data_frame")
    def _ground_truth_data_generator(self):
        """Creates generator for data
        
        ==
        Note
        ===
        A private method is specifically created to abstract the creation of
        the generator and to allow for easy resetting when needed.
        """
        self.ground_truth_data_iter: Generator[InteractionMatrix] = self._create_generator(
            "_ground_truth_data_series", "_ground_truth_data_frame")
    # TODO consider better naming
    def _next_data_timestamp_limit_generator(self):
        self.data_timestamp_limit_iter: Generator[int] = self._create_generator(
            "_data_timestamp_limit", "_data_timestamp_limit")

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

    def next_unlabeled_data(self, reset=False) -> InteractionMatrix:
        # Create generator if it does not exist or reset is True
        if reset or not hasattr(self, "unlabeled_data_iter"):
            self._unlabeled_data_generator()

        try:
            return next(self.unlabeled_data_iter)
        except StopIteration:
            raise EOWSetting()

    def next_ground_truth_data(self, reset=False) -> InteractionMatrix:
        # Create generator if it does not exist or reset is True
        if reset or not hasattr(self, "ground_truth_data_iter"):
            self._ground_truth_data_generator()

        try:
            return next(self.ground_truth_data_iter)
        except StopIteration:
            raise EOWSetting()

    def next_incremental_data(self, reset=False) -> InteractionMatrix:
        # Create generator if it does not exist or reset is True
        if reset or not hasattr(self, "incremental_data_iter"):
            self._incremental_data_generator()

        try:
            return next(self.incremental_data_iter)
        except StopIteration:
            raise EOWSetting()

    def next_data_timestamp_limit(self, reset=False):
        if reset or not hasattr(self, "data_timestamp_limit_iter"):
            self._next_data_timestamp_limit_generator()

        try:
            return next(self.data_timestamp_limit_iter)
        except StopIteration:
            raise EOWSetting()

    def reset_data_generators(self):
        logger.info("Resetting data generators.")
        self._unlabeled_data_generator()
        self._ground_truth_data_generator()
        self._next_data_timestamp_limit_generator()
        self._incremental_data_generator()
        logger.info("Data generators are reset.")
