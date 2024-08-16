import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from streamsight.matrix import InteractionMatrix

logger = logging.getLogger(__name__)


class Splitter(ABC):
    """Splitter class for splitting datasets into two based on a splitting condition."""

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
    def split(
        self, data: InteractionMatrix
    ) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Splits dataset into two based on a splitting condition.

        :param data: Interactions to split
        :type data: InteractionMatrix
        """
        raise NotImplementedError(f"{self.name} must implement the _split method.")

class TimestampSplitter(Splitter):
    """Splits data by timestamp.
    
    Split data so that the first return value contains interactions in
    `[t-t_lower, t)`, and the second those in `[t, t+t_upper]`.

    If `t_lower` or `t_upper` are omitted, they are assumed to have a value of infinity.
    A user can occur in both return values.


    Attribute definition
    ====================
    
    - :attr:`past_interaction`: List of unlabeled data. Interval is `[0, t)`.
    - :attr:`future_interaction`: Data used for training the model. Interval is `[t, t+t_upper)` or `[t,inf]`.

    :param t: Timestamp to split on in seconds since epoch.
    :type t: int
    :param t_lower: Seconds before t. Lower bound on the timestamp
        of interactions in the first return value. Defaults to None (infinity).
    :type t_lower: int, optional
    :param t_upper: Seconds past t. Upper bound on the timestamp
        of interactions in the second return value. Defaults to None (infinity).
    :type t_upper: int, optional
    """

    def __init__(
        self, t: int, t_lower: Optional[int] = None, t_upper: Optional[int] = None
    ):
        super().__init__()
        self.t = t
        self.t_lower = t_lower
        self.t_upper = t_upper

    def split(
        self, data: InteractionMatrix
    ) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Splits data so that `past_interaction` contains interactions in `[t-t_lower, t)`,
        and `future_interaction` those in `[t, t+t_upper]`.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        :return: A 2-tuple containing the `past_interaction` and `future_interaction` matrices.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """

        if self.t_lower is None:
            # timestamp < t
            past_interaction = data.timestamps_lt(self.t)
        else:
            # t-t_lower =< timestamp < t
            past_interaction = data.timestamps_lt(self.t).timestamps_gte(
                self.t - self.t_lower
            )

        if self.t_upper is None:
            # timestamp >= t
            future_interaction = data.timestamps_gte(self.t)
        else:
            # t =< timestamp < t + t_upper
            future_interaction = data.timestamps_gte(self.t).timestamps_lt(
                self.t + self.t_upper
            )

        logger.debug(f"{self.identifier} - Split successful")

        return past_interaction, future_interaction


class NPastInteractionTimestampSplitter(TimestampSplitter):
    """Splits data with n past interactions based on a timestamp.
    
    Splits the data into unlabeled and ground truth data based on a timestamp.
    Historical data contains last `n_seq_data` interactions before the timestamp `t`
    and the future interaction contains interactions after the timestamp `t`.


    Attribute definition
    ====================
    
    - :attr:`past_interaction`: List of unlabeled data. Interval is `[0, t)`.
    - :attr:`future_interaction`: Data used for training the model. Interval is `[t, t+t_upper)` or `[t,inf]`.

    :param t: Timestamp to split on in seconds since epoch.
    :type t: int
    :param t_upper: Seconds past t. Upper bound on the timestamp
        of interactions. Defaults to None (infinity).
    :type t_upper: int, optional
    :param n_seq_data: Number of last interactions to provide as unlabeled data
        for model to make prediction.
    :type n_seq_data: int, optional
    :return: A 2-tuple containing the `past_interaction` and `future_interaction` matrices.
    :rtype: Tuple[InteractionMatrix, InteractionMatrix]
    """

    def __init__(
        self,
        t,
        t_upper: Optional[int] = None,
        n_seq_data: int = 1,
    ):
        super().__init__(t, None, t_upper)
        self.n_seq_data = n_seq_data

    def update_split_point(self, t: int):
        logger.debug(f"{self.identifier} - Updating split point to t={t}")
        self.t = t

    def split(
        self, data: InteractionMatrix
    ) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Splits data such that the following definition holds:
        
        - :attr:`past_interaction` : List of unlabeled data. Interval is `[0, t)`.
        - :attr:`future_interaction` : Data used for training the model. Interval is `[t, t+t_upper)` or `[t,inf]`.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        :return: A 2-tuple containing the `past_interaction` and `future_interaction` matrices.
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        if self.t_upper is None:
            future_interaction = data.timestamps_gte(self.t)
        else:
            future_interaction = data.timestamps_lt(
                self.t + self.t_upper
            ).timestamps_gte(self.t)
        
        past_interaction = data.get_users_n_last_interaction(
                self.n_seq_data, self.t,future_interaction.user_ids
            )
        logger.debug(f"{self.identifier} - Split successful")
        return past_interaction, future_interaction
