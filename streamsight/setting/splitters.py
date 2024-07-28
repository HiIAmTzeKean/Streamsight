from abc import ABC, abstractmethod
import logging
from typing import Optional, Set, Tuple

import numpy as np
from streamsight.matrix.interation_matrix import InteractionMatrix

logger = logging.getLogger(__name__)


class Splitter(ABC):
    """Splitter class for splitting datasets into two based on a splitting condition.
    """

    def __init__(self):
        pass

    @property
    def name(self):
        """The name of the splitter."""
        return self.__class__.__name__

    @property
    def identifier(self):
        """String identifier of the splitter object,
        contains name and parameter values."""
        paramstring = ",".join((f"{k}={v}" for k, v in self.__dict__.items()))
        return self.name + f"({paramstring})"

    @abstractmethod
    def split(self, data: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Splits dataset into two based on a splitting condition.

        :param data: Interactions to split
        :type data: InteractionMatrix
        """
        raise NotImplementedError(
            f"{self.name} must implement the _split method.")


class UserSplitter(Splitter):
    """Split data by the user identifiers of the interactions.

    Users in ``users_in`` are assigned to the first return value,
    users in ``users_out`` are assigned to the second return value.

    :param users_in: Users for whom the events are assigned to the `in_matrix`
    :type users_in: Set[int]
    :param users_out: Users for whom the events are assigned to the `out_matrix`
    :type users_out: Set[int]
    """

    def __init__(
        self,
        users_in: Set[int],
        users_out: Set[int],
    ):
        super().__init__()
        self.users_in = users_in
        self.users_out = users_out

    def split(self, data: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Splits data by the user identifiers of the interactions.

        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        :return: A 2-tuple: the first value contains
            the interactions of ``users_in``,
            the second the interactions of ``users_out``.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        data_in = data.users_in(self.users_in)
        data_out = data.users_in(self.users_out)

        return data_in, data_out


class TimestampSplitter(Splitter):
    """Split data so that the first return value contains interactions in ``[t-t_lower, t)``,
    and the second those in ``[t, t+t_upper]``.

    If ``t_lower`` or ``t_upper`` are omitted, they are assumed to have a value of infinity.
    A user can occur in both return values.

    :param t: Timestamp to split on in seconds since epoch.
    :type t: int
    :param t_lower: Seconds before t. Lower bound on the timestamp
        of interactions in the first return value. Defaults to None (infinity).
    :type t_lower: int, optional
    :param t_upper: Seconds past t. Upper bound on the timestamp
        of interactions in the second return value. Defaults to None (infinity).
    :type t_upper: int, optional
    """

    def __init__(self, t:int, t_lower: Optional[int] = None, t_upper: Optional[int] = None):
        super().__init__()
        self.t = t
        self.t_lower = t_lower
        self.t_upper = t_upper

    def split(self, data: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Splits data so that ``data_in`` contains interactions in ``[t-t_lower, t)``,
        and ``data_out`` those in ``[t, t+t_upper]``.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        :return: A 2-tuple containing the ``data_in`` and ``data_out`` matrices.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """

        if self.t_lower is None:
            # timestamp < t
            data_in = data.timestamps_lt(self.t)
        else:
            # t-t_lower =< timestamp < t
            data_in = data.timestamps_lt(self.t).timestamps_gte(self.t - self.t_lower)

        if self.t_upper is None:
            # timestamp >= t
            data_out = data.timestamps_gte(self.t)
        else:
            # t =< timestamp < t + t_upper
            data_out = data.timestamps_gte(self.t).timestamps_lt(self.t + self.t_upper)

        logger.debug(f"{self.identifier} - Split successful")

        return data_in, data_out


class NPastInteractionTimestampSplitter(TimestampSplitter):
    """Splits the data into unlabeled and ground truth data based on a timestamp.
    Historical data contains last ``n_seq_data`` interactions before the timestamp ``t``
    and the future interaction contains interactions after the timestamp ``t``.
    
    :param t: Timestamp to split on in seconds since epoch.
    :type t: int
    :param t_upper: Seconds past t. Upper bound on the timestamp
        of interactions. Defaults to None (infinity).
    :type t_upper: int, optional
    :param n_seq_data: Number of last interactions to provide as unlabeled data
        for model to make prediction.
    :type n_seq_data: int, optional
    :return: A 2-tuple containing the ``past_interaction`` and ``future_interaction`` matrices.
    :rtype: Tuple[InteractionMatrix, InteractionMatrix]
    """
    def __init__(self, t,
                 t_upper: Optional[int] = None,
                 n_seq_data: int = 1):
        super().__init__(t, None, t_upper)
        self.n_seq_data = n_seq_data
        
    def update_split_point(self, t:int):
        logger.debug(f"{self.identifier} - Updating split point to t={t}")
        self.t = t

    def split(self, data: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:
        if self.t_upper is None:
            future_interaction = data.timestamps_gte(self.t).timestamps_gte(self.t)
        else:
            future_interaction = data.timestamps_lt(self.t + self.t_upper).timestamps_gte(self.t)
        assert future_interaction is not None
        # TODO past interaction should only contain users/items that are in the ground truth
        # ? i filtered by user interaction now, how should i know if its item
        past_interaction = data.get_user_n_last_interaction(self.n_seq_data,self.t)
        return past_interaction, future_interaction


class StrongGeneralizationSplitter(Splitter):
    """Randomly splits the users into two sets so that
    interactions for a user will always occur only in one split.

    :param in_frac: The fraction of interactions that are assigned
        to the first value in the output tuple. Defaults to 0.7.
    :type in_frac: float, optional
    :param seed: Seed the random generator. Set this value
        if you require reproducible results.
    :type seed: int, optional
    :param error_margin: The allowed error between ``in_frac``
        and the actual split fraction. Defaults to 0.01.
    :type error_margin: float, optional
    """

    def __init__(self, in_frac=0.7, seed=None, error_margin=0.01):
        super().__init__()
        self.in_frac = in_frac
        self.out_frac = 1 - in_frac

        if seed is None:
            # Set seed if it was not set before.
            seed = np.random.get_state()[1][0]

        self.seed = seed
        self.error_margin = error_margin

    def split(self, data):
        """Randomly splits the users into two sets so that
            interactions for a user will always occur only in one split.

        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        :return: A 2-tuple containing the ``data_in`` and ``data_out`` matrices.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        sp_mat = data.values

        users = list(set(sp_mat.nonzero()[0]))

        nr_users = len(users)

        # Seed to make testing possible
        if self.seed is not None:
            np.random.seed(self.seed)

        # Try five times
        for i in range(0, 5):
            np.random.shuffle(users)

            in_cut = int(np.floor(nr_users * self.in_frac))

            users_in = users[:in_cut]
            users_out = users[in_cut:]

            total_interactions = sp_mat.nnz
            data_in_cnt = sp_mat[users_in, :].nnz

            real_frac = data_in_cnt / total_interactions

            within_margin = np.isclose(
                real_frac, self.in_frac, atol=self.error_margin)

            if within_margin:
                logger.debug(
                    f"{self.identifier} - Iteration {i} - Within margin")
                break
            else:
                logger.debug(
                    f"{self.identifier} - Iteration {i} - Not within margin")

        u_splitter = UserSplitter(users_in, users_out)
        ret = u_splitter.split(data)

        logger.debug(f"{self.identifier} - Split successful")
        return ret
