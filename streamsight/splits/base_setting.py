from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import Iterator, List, Optional, Tuple, Union
from warnings import warn

from streamsight.splits.splitters import StrongGeneralizationSplitter
from streamsight.matrix import InteractionMatrix

logger = logging.getLogger(__name__)


class FrameExpectedError(Exception):
    def __init__(self, message: Optional[str] = None):
        if not message:
            message = "Setting is done with multiple splits. A frame should be used instead. Try calling the attribute with '_frame' suffix."
        self.message = message
        super().__init__(self.message)


class SeriesExpectedError(Exception):
    def __init__(self, message: Optional[str] = None):
        if not message:
            message = "Setting is done with single split. Try calling the attribute without '_frame' suffix."
        self.message = message
        super().__init__(self.message)


def check_series(func):
    def check_single_split(self):
        if self.num_split_set > 1:
            raise FrameExpectedError()
        return func(self)
    return check_single_split


class Setting(ABC):
    """Base class for defining an evaluation scenario.

    A scenario is a set of steps that splits data into training,
    validation and test datasets.
    The test dataset is made up of two components:
    a fold-in set of interactions that is used to predict another held-out
    set of interactions.
    The creation of the validation dataset, from the full training dataset,
    should follow the same splitting strategy as
    the one used to create training and test datasets from the full dataset.

    :param validation: Create validation datasets when True,
        else split into training and test datasets.
    :type validation: boolean, optional
    :param seed: Seed for randomisation parts of the scenario.
        Defaults to None, so random seed will be generated.
    :type seed: int, optional
    """

    def __init__(self, seed: int | None = None):
        if seed is None:
            # Set seed if it was not set before.
            seed = np.random.get_state()[1][0]
        self.seed = seed
        self.num_split_set = 1
        """Number of splits created from sliding window. Defaults to 1 (no splits on training set)."""
        self._num_full_interactions: int
        self._unlabeled_data: InteractionMatrix
        self._unlabeled_data_frame: List[InteractionMatrix]
        self._ground_truth_data: InteractionMatrix
        self._ground_truth_data_frame: List[InteractionMatrix]
        self._ground_truth_data: InteractionMatrix
        self._ground_truth_data_frame: List[InteractionMatrix]
        self._background_data: InteractionMatrix

    @abstractmethod
    def _split(self, data_m: InteractionMatrix) -> None:
        """Abstract method to be implemented by the scenarios.

        Splits the data and assigns to :attr:`background_data`,
        :attr:`ground_truth_data, :attr:`unlabeled_data`

        :param data_m: Interaction matrix to be split.
        :type data_m: InteractionMatrix
        """

    def split(self, data_m: InteractionMatrix) -> None:
        """Splits ``data_m`` according to the scenario.

        After splitting properties :attr:`training_data`,
        :attr:`validation_data` and :attr:`test_data` can be used
        to retrieve the splitted data.

        :param data_m: Interaction matrix that should be split.
        :type data: InteractionMatrix
        """
        self._num_full_interactions = data_m.num_interactions
        self._split(data_m)

        logger.debug("Checking split attribute and sizes.")
        self._check_split()

    @property
    def background_data(self) -> InteractionMatrix:
        """The full training dataset, which should be used for a final training
        after hyper parameter optimisation.

        :return: Interaction Matrix of training interactions.
        :rtype: Union[InteractionMatrix, List[InteractionMatrix]]
        """
        if hasattr(self, "_background_data"):
            return self._background_data
        raise KeyError(
            "Split before trying to access the background_data property.")

    @property
    @check_series
    def unlabeled_data_series(self) -> InteractionMatrix:
        """Fold-in part of the test dataset"""
        return self.evaluation_data_series[0]

    @property
    def unlabeled_data_frame(self) -> List[InteractionMatrix]:
        """Fold-in part of the test dataset"""
        return [x[0] for x in self.evaluation_data_frame]

    @property
    def unlabeled_data(self) -> Union[InteractionMatrix, List[InteractionMatrix]]:
        if self.num_split_set == 1:
            return self.unlabeled_data_series
        return self.unlabeled_data_frame

    @property
    @check_series
    def ground_truth_data_series(self) -> InteractionMatrix:
        """Held-out part of the test dataset"""
        return self.evaluation_data_series[1]

    @property
    def ground_truth_data_frame(self) -> List[InteractionMatrix]:
        """Held-out part of the test dataset"""
        return [x[1] for x in self.evaluation_data_frame]

    @property
    def ground_truth_data(self) -> Union[InteractionMatrix, List[InteractionMatrix]]:
        if self.num_split_set == 1:
            return self.ground_truth_data_series
        return self.ground_truth_data_frame

    @property
    def evaluation_data_frame(self) -> List[Tuple[InteractionMatrix, InteractionMatrix]]:
        if self.num_split_set == 1:
            raise SeriesExpectedError()

        # ? is this necessary? we can get the model to handle this
        datasets = []
        for dataset_idx in range(self.num_split_set):
            in_users = self._unlabeled_data_frame[dataset_idx].active_users
            out_users = self._ground_truth_data_frame[dataset_idx].active_users

            matching = list(in_users.intersection(out_users))
            datasets.append((self._unlabeled_data_frame[dataset_idx].users_in(
                matching), self._ground_truth_data_frame[dataset_idx].users_in(matching)))
        return datasets

    @property
    def evaluation_data_series(self) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """The test dataset. Consist of a fold-in and hold-out set of interactions.

        Data is processed such that both matrices contain the exact same users.
        Users that were present in only one of the matrices
        and not in the other are removed.

        :return: Test data matrices as InteractionMatrix in, InteractionMatrix out.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        in_users = self._unlabeled_data.active_users
        out_users = self._ground_truth_data.active_users

        matching = list(in_users.intersection(out_users))
        return (
            self._unlabeled_data.users_in(matching),
            self._ground_truth_data.users_in(matching)
        )

    @property
    def ground_truth(self) -> Union[Tuple[InteractionMatrix, InteractionMatrix], List[Tuple[InteractionMatrix, InteractionMatrix]]]:
        """The test dataset. Consist of a fold-in and hold-out set of interactions.

        Data is processed such that both matrices contain the exact same users.
        Users that were present in only one of the matrices
        and not in the other are removed.

        :return: Test data matrices as InteractionMatrix in, InteractionMatrix out.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        # make sure users match.
        if self.num_split_set == 1:
            return self.evaluation_data_series
        else:
            return self.evaluation_data_frame

    def _check_split(self):
        """Checks that the splits have been done properly.

        Makes sure all expected attributes are set.
        """

        assert (hasattr(self, "_background_data")
                and self._background_data is not None)

        assert (hasattr(self, "_unlabeled_data") and self._unlabeled_data is not None) \
            or (hasattr(self, "_unlabeled_data_frame") and self._unlabeled_data_frame is not None)

        assert (hasattr(self, "_ground_truth_data") and self._ground_truth_data is not None) \
            or (hasattr(self, "_ground_truth_data_frame") and self._ground_truth_data_frame is not None)

        self._check_size()

    def _check_size(self):
        """
        Warns user if any of the sets is unusually small or empty
        """

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
        if self.num_split_set == 1:
            n_unlabel = self._unlabeled_data.num_interactions
            n_ground_truth = self._ground_truth_data.num_interactions

            check_empty("Unlabeled set", n_unlabel)
            check_empty("Ground truth set", n_ground_truth)
            check_ratio("Ground truth set", n_ground_truth, n_unlabel, 0.05)

        else:
            for dataset_idx in range(self.num_split_set):
                n_unlabel = self._unlabeled_data_frame[dataset_idx].num_interactions
                n_ground_truth = self._ground_truth_data_frame[dataset_idx].num_interactions

                check_empty("Unlabeled set", n_unlabel)
                check_empty("Ground truth set", n_ground_truth)

    def _unlabeled_data_generator(self):
        def create_generator():
            if self.num_split_set == 1:
                yield self.unlabeled_data_series
            else:
                for data in self.unlabeled_data_frame:
                    yield data
        self.unlabeled_data_iter = create_generator()
        

    def next_unlabeled_data(self, reset=False):
        # Create generator if it does not exist or reset is True
        if reset or not hasattr(self, "unlabeled_data_iter"):
            self._unlabeled_data_generator()
        
        try:
            next(self.unlabeled_data_iter)
        except StopIteration:
            logger.debug("End of unlabeled data reached. To reset, set reset=True")
            return None

    def _ground_truth_data_generator(self):
        def create_generator():
            if self.num_split_set == 1:
                yield self.ground_truth_data_series
            else:
                for data in self.ground_truth_data_frame:
                    yield data
        self.ground_truth_data_iter = create_generator()
        
    def next_ground_truth_data(self, reset=False):
        # Create generator if it does not exist or reset is True
        if reset or not hasattr(self, "ground_truth_data_iter"):
            self._ground_truth_data_generator()
        
        try:
            return next(self.ground_truth_data_iter)
        except StopIteration:
            logger.debug("End of ground truth data reached. To reset, set reset=True")
            return None
