from copy import deepcopy
from dataclasses import asdict, dataclass
import logging
import operator
from typing import Callable, Optional, Set, Tuple
from scipy.sparse import csr_matrix

import pandas as pd

import numpy as np

logger = logging.getLogger(__name__)

class InteractionMatrix:
    """An InteractionMatrix contains interactions between users and items at a certain time.

    It provides a number of properties and methods for easy manipulation of this interaction data.

    .. note::

        The InteractionMatrix does not assume binary user-item pairs.
        If a user interacts with an item more than once, there will be two entries for this user-item pair.

    :param df: Dataframe containing user-item interactions. Must contain at least
        item ids and user ids.
    :type df: pd.DataFrame
    :param item_ix: Item ids column name.
    :type item_ix: str
    :param user_ix: User ids column name.
    :type user_ix: str
    :param timestamp_ix: Interaction timestamps column name.
    :type timestamp_ix: str, optional
    :param shape: The desired shape of the matrix, i.e. the number of users and items.
        If no shape is specified, the number of users will be equal to the
        maximum user id plus one, the number of items to the maximum item
        id plus one.
    :type shape: Tuple[int, int], optional
    """

    ITEM_IX = "iid"
    USER_IX = "uid"
    TIMESTAMP_IX = "ts"
    INTERACTION_IX = "interactionid"

    @dataclass
    class InteractionMatrixProperties:
        num_users: int
        num_items: int

        def to_dict(self):
            return asdict(self)

    def __init__(
        self,
        df: pd.DataFrame,
        item_ix: str,
        user_ix: str,
        timestamp_ix: str,
        shape: Optional[Tuple[int, int]] = None,
    ):
        col_mapper = {
            item_ix: InteractionMatrix.ITEM_IX,
            user_ix: InteractionMatrix.USER_IX,
            timestamp_ix: InteractionMatrix.TIMESTAMP_IX
        }

        df = df.rename(columns=col_mapper)
        df = df[[InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX, InteractionMatrix.TIMESTAMP_IX]].copy()
        

        df = df.reset_index(drop=True).reset_index().rename(columns={"index": InteractionMatrix.INTERACTION_IX})

        self._df = df

        n_users_df = self._df[InteractionMatrix.USER_IX].unique().shape[0]
        n_items_df = self._df[InteractionMatrix.ITEM_IX].unique().shape[0]

        num_users = n_users_df if shape is None else shape[0]
        num_items = n_items_df if shape is None else shape[1]

        if n_users_df > num_users:
            raise ValueError(
                "Provided shape does not match dataframe, can't have fewer rows than maximal user identifier."
                f" {num_users} < {n_users_df}"
            )

        if n_items_df > num_items:
            raise ValueError(
                "Provided shape does not match dataframe, can't have fewer columns than maximal item identifier."
                f" {num_items} < {n_items_df}"
            )

        self.shape = (int(num_users), int(num_items))

    def copy(self) -> "InteractionMatrix":
        """Create a deep copy of this InteractionMatrix.

        :return: Deep copy of this InteractionMatrix.
        :rtype: InteractionMatrix
        """
        return deepcopy(self)

    def union(self, im: "InteractionMatrix") -> "InteractionMatrix":
        """Combine events from this InteractionMatrix with another.

        The matrices need to have the same shape and either both have timestamps or neither.

        :param im: InteractionMatrix to union with.
        :type im: InteractionMatrix
        :return: Union of interactions in this InteractionMatrix and the other.
        :rtype: InteractionMatrix
        """
        if self.shape != im.shape:
            raise ValueError(
                f"Shapes don't match. This InteractionMatrix has shape {self.shape}, the other {im.shape}"
            )

        df = pd.concat([self._df, im._df])
        return InteractionMatrix(
            df,
            InteractionMatrix.ITEM_IX,
            InteractionMatrix.USER_IX,
            InteractionMatrix.TIMESTAMP_IX,
            shape=self.shape,
        )

    @property
    def properties(self) -> InteractionMatrixProperties:
        return self.InteractionMatrixProperties(
            num_users=self.shape[0],
            num_items=self.shape[1]
        )
        
    @property
    def num_interactions(self) -> int:
        """The total number of interactions.

        :return: Total interaction count.
        :rtype: int
        """
        return len(self._df)
    
    @property
    def values(self) -> csr_matrix:
        """All user-item interactions as a sparse matrix of size ``(|users|, |items|)``.

        Each entry is the number of interactions between that user and item.
        If there are no interactions between a user and item, the entry is 0.

        :return: Interactions between users and items as a csr_matrix.
        :rtype: csr_matrix
        """
        values = np.ones(self._df.shape[0])
        indices = self._df[[InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX]].values
        indices = indices[:, 0], indices[:, 1]

        matrix = csr_matrix((values, indices), shape=self.shape, dtype=np.int32)
        return matrix
        
    def users_in(self, U: Set[int], inplace=False) -> Optional["InteractionMatrix"]:
        """Keep only interactions by one of the specified users.

        :param U: A Set or List of users to select the interactions from.
        :type U: Union[Set[int], List[int]]
        :param inplace: Apply the selection in place or not, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        logger.debug("Performing users_in comparison")

        mask = self._df[InteractionMatrix.USER_IX].isin(U)

        return self._apply_mask(mask, inplace=inplace)
    
    def _apply_mask(self, mask, inplace=False) -> Optional["InteractionMatrix"]:
        interaction_m = self if inplace else self.copy()

        c_df = interaction_m._df[mask]

        interaction_m._df = c_df
        return None if inplace else interaction_m
    
    def _timestamps_cmp(self, op: Callable, timestamp: float, inplace: bool = False) -> Optional["InteractionMatrix"]:
        """Filter interactions based on timestamp.
        Keep only interactions for which op(t, timestamp) is True.

        :param op: Comparison operator.
        :type op: Callable
        :param timestamp: Timestamp to compare against in seconds from epoch.
        :type timestamp: float
        :param inplace: Modify the data matrix in place. If False, returns a new object.
        :type inplace: bool, optional
        """
        logger.debug(f"Performing {op.__name__}(t, {timestamp})")

        mask = op(self._df[InteractionMatrix.TIMESTAMP_IX], timestamp)

        return self._apply_mask(mask, inplace=inplace)

    def timestamps_gt(self, timestamp: float, inplace: bool = False) -> Optional["InteractionMatrix"]:
        """Select interactions after a given timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        return self._timestamps_cmp(operator.gt, timestamp, inplace)

    def timestamps_lt(self, timestamp: float, inplace: bool = False) -> Optional["InteractionMatrix"]:
        """Select interactions up to a given timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        return self._timestamps_cmp(operator.lt, timestamp, inplace)

    def timestamps_gte(self, timestamp: float, inplace: bool = False) -> Optional["InteractionMatrix"]:
        """Select interactions after and including a given timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        return self._timestamps_cmp(operator.ge, timestamp, inplace)

    def timestamps_lte(self, timestamp: float, inplace: bool = False) -> Optional["InteractionMatrix"]:
        """Select interactions up to and including a given timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        return self._timestamps_cmp(operator.le, timestamp, inplace)
    
    def __add__(self, other):
        return self.union(other)
    
    