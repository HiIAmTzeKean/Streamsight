import logging
from abc import ABC, abstractmethod
from typing import Generator, List, Optional, Union
from warnings import warn

import numpy as np

from streamsight.matrix import InteractionMatrix
from streamsight.settings.processor import PredictionDataProcessor

logger = logging.getLogger(__name__)

class EOWSetting(Exception):
    """End of Window Setting Exception."""
    def __init__(self, message="End of window setting reached. Call reset_data_generators() to start again."):
        self.message = message
        super().__init__(self.message)

def check_split_complete(func):
    """Check split complete decorator.
    
    Decorator to check if the split is complete before accessing the property.
    To be used on :class:`Setting` class methods.
    """
    def check_split_for_func(self):
        if not self.is_ready:
            raise KeyError(
                f"Split before trying to access {func.__name__} property.")
        return func(self)
    return check_split_for_func


class Setting(ABC):
    """Base class for defining an evaluation setting.
    
    Core attribute
    ====================
    - :attr:`background_data`: Data used for training the model. Interval is `[0, background_t)`.
    - :attr:`unlabeled_data`: List of unlabeled data. Each element is a :class:`InteractionMatrix` object of interval `[0, t)`.
    - :attr:`ground_truth_data`: List of ground truth data. Each element is a :class:`InteractionMatrix` object of interval `[t, t + window_size)`.
    - :attr:`incremental_data`: List of data that is used to incrementally update the model. Each element is a :class:`InteractionMatrix` object of interval `[t, t + window_size)`. Unique to :class:`SlidingWindowSetting`.
    - :attr:`data_timestamp_limit`: List of timestamps that the splitter will slide over the data.

    :param seed: Seed for randomisation parts of the setting.
        Defaults to None, so random seed will be generated.
    :type seed: int, optional
    """

    def __init__(
        self,
        seed: Optional[int] = None,
    ):
        if seed is None:
            # Set seed if it was not set before.
            seed = np.random.get_state()[1][0]
        self.seed = seed

        self.prediction_data_processor = PredictionDataProcessor()

        self._num_split_set = 1

        self._sliding_window_setting = False
        self._split_complete = False
        """Number of splits created from sliding window. Defaults to 1 (no splits on training set)."""
        self._num_full_interactions: int
        self._unlabeled_data: Union[InteractionMatrix, List[InteractionMatrix]]
        self._ground_truth_data: Union[InteractionMatrix, List[InteractionMatrix]]
        """Data containing the ground truth interactions to :attr:`_unlabeled_data`. If :class:`SlidingWindowSetting`, then it will be a list of :class:`InteractionMatrix`."""
        self._incremental_data: List[InteractionMatrix]
        """Data that is used to incrementally update the model. Unique to :class:`SlidingWindowSetting`."""
        self._background_data: InteractionMatrix
        """Data used as the initial set of interactions to train the model."""
        self._data_timestamp_limit: Union[int, List[int]]
        """This is the limit of the data in the corresponding split."""
        self.n_seq_data: int
        """Number of last sequential interactions to provide in :attr:`unlabeled_data` as unlabeled data for model to make prediction."""
        self.top_K: int
        """Number of interaction per user that should be selected for evaluation purposes in :attr:`ground_truth_data`."""

    @property
    def name(self) -> str:
        """Name of the setting.
        
        :return: Name of the setting.
        :rtype: str
        """
        return self.__class__.__name__
    
    @property
    def params(self):
        """Parameters of the setting."""
        return {}
    
    def get_params(self):
        """Get the parameters of the setting."""
        return self.params
    
    @property
    def identifier(self):
        """Name of the setting."""
        # return f"{super().identifier[:-1]},K={self.K})"
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items() if v is not None))
        return self.name + "(" + paramstring + ")"
    
    @abstractmethod
    def _split(self, data_m: InteractionMatrix) -> None:
        """Abstract method to be implemented by the scenarios.

        Splits the data and assigns to :attr:`background_data`,
        :attr:`ground_truth_data`, :attr:`unlabeled_data`

        :param data_m: Interaction matrix to be split.
        :type data_m: InteractionMatrix
        """

    def split(self, data_m: InteractionMatrix) -> None:
        """Splits `data_m` according to the scenario.

        After splitting properties :attr:`training_data`,
        :attr:`validation_data` and :attr:`test_data` can be used
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

    def _check_split_complete(self):
        """Check if the setting is ready to be used for evaluation.

        :raises KeyError: If the setting is not ready to be used for evaluation.
        """
        if not self.is_ready:
            raise KeyError(f"Setting has not been split yet. Call split() method before accessing the property.")

    @property
    def num_split(self) -> int:
        """Number of splits created from dataset.
        
        This property defaults to 1 (no splits on training set) on a typical setting.
        Usually for the :class:`SlidingWindowSetting` this property will be greater than 1
        if there are multiple splits created from the sliding window on the dataset.

        :return: Number of splits created from dataset.
        :rtype: int
        """
        return self._num_split_set

    @property
    def is_ready(self) -> bool:
        """Flag on setting if it is ready to be used for evaluation.
        
        :return: If the setting is ready to be used for evaluation.
        :rtype: bool
        """
        return self._split_complete

    @property
    def is_sliding_window_setting(self) -> bool:
        """Flag to indicate if the setting is :class:`SlidingWindowSetting`.
        
        :return: If the setting is :class:`SlidingWindowSetting`.
        :rtype: bool
        """
        return self._sliding_window_setting

    @property
    def background_data(self) -> InteractionMatrix:
        """Background data provided for the model for the initial training.
        
        This data is used as the initial set of interactions to train the model.

        :return: Interaction Matrix of training interactions.
        :rtype: InteractionMatrix
        """
        self._check_split_complete()

        return self._background_data

    @property
    def data_timestamp_limit(self) -> Union[int, List[int]]:
        """The timestamp limit of the data in the corresponding split.

        :return: Timestamp limit of the data in the corresponding split.
        :rtype: Union[int, List[int]]
        """
        self._check_split_complete()

        return self._data_timestamp_limit

    @property
    def unlabeled_data(self) -> Union[InteractionMatrix, List[InteractionMatrix]]:
        """Unlabeled data for the model to make predictions on.
        
        Contains the user/item ID for prediction along with previous sequential
        interactions of user-item on items if it exists. This data is used to
        make predictions on the ground truth data.

        :return: Either a single InteractionMatrix or a list of InteractionMatrix
            if the setting is a sliding window setting.
        :rtype: Union[InteractionMatrix, List[InteractionMatrix]]
        """
        self._check_split_complete()

        if not self._sliding_window_setting:
            return self._unlabeled_data
        return self._unlabeled_data

    @property
    def ground_truth_data(self) -> Union[InteractionMatrix, List[InteractionMatrix]]:
        """Ground truth data to evaluate the model's predictions on.
        
        Contains the actual interactions of the user-item interaction that the
        model is supposed to predict.

        :return: _description_
        :rtype: Union[InteractionMatrix, List[InteractionMatrix]]
        """
        self._check_split_complete()

        if not self._sliding_window_setting:
            return self._ground_truth_data
        return self._ground_truth_data

    @property
    def incremental_data(self) -> List[InteractionMatrix]:
        """Data that is used to incrementally update the model.

        Unique to sliding window setting.

        :return: _description_
        :rtype: List[InteractionMatrix]
        """
        self._check_split_complete()

        if not self._sliding_window_setting:
            raise AttributeError(
                "Incremental data is only available for sliding window setting.")
        return self._incremental_data

    def _check_split(self):
        """Checks that the splits have been done properly.

        Makes sure all expected attributes are set.
        """
        logger.debug("Checking split attributes.")
        assert (hasattr(self, "_background_data")
                and self._background_data is not None)

        assert (hasattr(self, "_unlabeled_data") and self._unlabeled_data is not None) \
            or (hasattr(self, "_unlabeled_data") and self._unlabeled_data is not None)

        assert (hasattr(self, "_ground_truth_data") and self._ground_truth_data is not None) \
            or (hasattr(self, "_ground_truth_data") and self._ground_truth_data is not None)
        logger.debug("Split attributes are set.")

        self._check_size()

    def _check_size(self):
        """
        Warns user if any of the sets is unusually small or empty
        """
        logger.debug("Checking size of split sets.")

        def check_ratio(name, count, total, threshold) -> None:
            if check_empty(name, count):
                return
            
            if (count + 1e-9) / (total + 1e-9) < threshold:
                warn(
                    UserWarning(
                        f"{name} resulting from {self.name} is unusually small."
                    )
                )

        def check_empty(name, count) -> bool:
            if count == 0:
                warn(
                    UserWarning(
                        f"{name} resulting from {self.name} is empty (no interactions)."
                    )
                )
                return True
            return False

        n_background = self._background_data.num_interactions
        # check_empty("Background data", n_background)
        check_ratio("Background data", n_background, self._num_full_interactions, 0.05)

        if not self._sliding_window_setting:
            n_unlabel = self._unlabeled_data.num_interactions
            n_ground_truth = self._ground_truth_data.num_interactions

            check_empty("Unlabeled data", n_unlabel)
            # check_empty("Ground truth data", n_ground_truth)
            check_ratio("Ground truth data", n_ground_truth, n_unlabel, 0.05)

        else:
            for dataset_idx in range(self._num_split_set):
                n_unlabel = self._unlabeled_data[dataset_idx].num_interactions
                n_ground_truth = self._ground_truth_data[dataset_idx].num_interactions

                check_empty(f"Unlabeled data[{dataset_idx}]", n_unlabel)
                check_empty(f"Ground truth data[{dataset_idx}]", n_ground_truth)
        logger.debug("Size of split sets are checked.")

    def _unlabeled_data_generator(self):
        """Creates generator for data
        
        Note
        ===
        
        A private method is specifically created to abstract the creation of
        the generator and to allow for easy resetting when needed.
        """
        self.unlabeled_data_iter: Generator[InteractionMatrix] = self._create_generator(
            "_unlabeled_data", "_unlabeled_data")

    def _incremental_data_generator(self):
        """Creates generator for data
        
        ==
        Note
        ===
        A private method is specifically created to abstract the creation of
        the generator and to allow for easy resetting when needed.
        """
        self.incremental_data_iter: Generator[InteractionMatrix] = self._create_generator(
            "_incremental_data", "_incremental_data")

    def _ground_truth_data_generator(self):
        """Creates generator for data
        
        ==
        Note
        ===
        A private method is specifically created to abstract the creation of
        the generator and to allow for easy resetting when needed.
        """
        self.ground_truth_data_iter: Generator[InteractionMatrix] = self._create_generator(
            "_ground_truth_data", "_ground_truth_data")
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

    def reset_data_generators(self) -> None:
        """Reset data generators.
        
        Resets the data generators to the beginning of the
        data series. API allows the programmer to reset the data generators
        of the setting object to the beginning of the data series.
        """
        logger.info("Resetting data generators.")
        self._unlabeled_data_generator()
        self._ground_truth_data_generator()
        self._next_data_timestamp_limit_generator()
        self._incremental_data_generator()
        logger.info("Data generators are reset.")