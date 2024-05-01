from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import pandas as pd


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

    def properties(self) -> InteractionMatrixProperties:
        return self.InteractionMatrixProperties(
            num_users=self.shape[0],
            num_items=self.shape[1]
        )
        
    def __add__(self, other):
        return self.union(other)
    
    