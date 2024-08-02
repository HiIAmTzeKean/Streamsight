from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd

from streamsight.matrix import InteractionMatrix, ItemUserBasedEnum


class Processor(ABC):
    """Base class for processing data.
    """
    def __init__(self, item_user_based: ItemUserBasedEnum):
        self._item_user_based = item_user_based
    @abstractmethod
    def process(self, past_interaction: InteractionMatrix, future_interaction: InteractionMatrix) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Injects the item/user ID to indicate ID for prediction.
        
        User ID to be predicted by the model will be indicated with item ID of
        "-1" as the corresponding label. The matrix with past interactions will
        contain the item/user ID to be predicted which will be derived from the set
        of item/user IDs in the future interaction matrix. Timestamp of the masked
        interactions will be preserved as the item ID are simply masked with
        "-1".
        
        :param past_interaction: Matrix of past interactions.
        :type past_interaction: InteractionMatrix
        :param future_interaction: Matrix of future interactions.
        :type future_interaction: InteractionMatrix
        :return: Tuple of past interaction with injected item/user ID to predict and
            ground truth future interactions of the actual interaction
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        pass


class PredictionDataProcessor(Processor):
    """Injects the item/user ID to indicate ID for prediction.
    
    Operates on the past and future interaction matrices to inject the item/user
    ID to be predicted by the model into the past interaction matrix. The
    resulting past interaction matrix will contain the item/user ID to be
    predicted which will be derived from the set of item/user IDs in the future
    interaction matrix. Timestamp of the masked interactions will be preserved as
    the item ID are simply masked with "-1".
    
    The corresponding ground truth future interactions of the actual interaction
    will be returned as well in a tuple when `process` is called.
    """

    def _inject_user_id(self,
                        past_interaction: InteractionMatrix,
                        future_interaction: InteractionMatrix, top_K:int = 1) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Injects the user ID to indicate ID for prediction.
        
        User ID to be predicted by the model will be indicated with item ID of
        "-1" as the corresponding label. The matrix with past interactions will
        contain the user ID to be predicted which will be derived from the set
        of user IDs in the future interaction matrix. Timestamp of the masked
        interactions will be preserved as the item ID are simply masked with
        "-1".
        
        :param past_interaction: Matrix of past interactions.
        :type past_interaction: InteractionMatrix
        :param future_interaction: Matrix of future interactions.
        :type future_interaction: InteractionMatrix
        :return: Tuple of past interaction with injected user ID to predict and
            ground truth future interactions of the actual interaction
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        users_to_predict = future_interaction.get_users_n_first_interaction(top_K)
        masked_frame = users_to_predict.copy_df()
        masked_frame[InteractionMatrix.ITEM_IX] = InteractionMatrix.MASKED_LABEL
        return past_interaction.concat(masked_frame), users_to_predict

    def _inject_item_id(self, past_interaction: InteractionMatrix, future_interaction: InteractionMatrix, top_K:int = 1) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Injects the item ID to indicate ID for prediction.
        
        User ID to be predicted by the model will be indicated with item ID of
        "-1" as the corresponding label. The matrix with past interactions will
        contain the item ID to be predicted which will be derived from the set
        of item IDs in the future interaction matrix. Timestamp of the masked
        interactions will be preserved as the item ID are simply masked with
        "-1".

        :param past_interaction: Matrix of past interactions.
        :type past_interaction: InteractionMatrix
        :param future_interaction: Matrix of future interactions.
        :type future_interaction: InteractionMatrix
        :return: Tuple of past interaction with injected user ID to predict and
            ground truth future interactions of the actual interaction
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        items_to_predict = future_interaction.get_items_n_first_interaction(top_K)
        masked_frame = items_to_predict.copy_df()
        masked_frame[InteractionMatrix.USER_IX] = InteractionMatrix.MASKED_LABEL
        return past_interaction.concat(masked_frame), items_to_predict
    
    def process(self, past_interaction: InteractionMatrix, future_interaction: InteractionMatrix, top_K:int = 1) -> Tuple[InteractionMatrix, InteractionMatrix]:
        if self._item_user_based == ItemUserBasedEnum.USER:
            return self._inject_user_id(past_interaction, future_interaction, top_K)
        else:
            return self._inject_item_id(past_interaction, future_interaction, top_K)