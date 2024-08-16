from dataclasses import dataclass
import logging
from enum import StrEnum
from typing import Tuple
from uuid import UUID

from streamsight.matrix import InteractionMatrix
from streamsight.registries.registry import AlgorithmStateEnum

logger = logging.getLogger(__name__)

class MetricLevelEnum(StrEnum):
    MICRO = "micro"
    MACRO = "macro"
    
    @classmethod
    def has_value(cls, value: str):
        """Check valid value for MetricLevelEnum

        :param value: String value input
        :type value: str
        """
        if value not in MetricLevelEnum:
            return False
        return True
    
    
@dataclass
class UserItemBaseStatus():
    """Unknown and known user/item base.
    
    This class is used to store the status of the user and item base. The class
    stores the known and unknown user and item set. The class also provides
    methods to update the known and unknown user and item set.
    """
    unknown_user = set()
    known_user = set()
    unknown_item = set()
    known_item = set()

    @property
    def known_shape(self) -> Tuple[int, int]:
        """Known shape of the user-item interaction matrix.
        
        This is the shape of the released user/item interaction matrix to the
        algorithm. This shape follows from assumption in the dataset that
        ID increment in the order of time.

        :return: Tuple of (`|user|`, `|item|`)
        :rtype: Tuple[int, int]
        """
        return (len(self.known_user), len(self.known_item))
    def _update_known_user_item_base(self, data:InteractionMatrix):
        """Updates the known user and item set with the data.

        :param data: Data to update the known user and item set with.
        :type data: InteractionMatrix
        """
        self.known_item.update(data.item_ids)
        self.known_user.update(data.user_ids)

    def _update_unknown_user_item_base(self, data:InteractionMatrix):
        self.unknown_user = data.user_ids.difference(self.known_user)
        self.unknown_item = data.item_ids.difference(self.known_item)

    def _reset_unknown_user_item_base(self):
        """Clears the unknown user and item set.
        
        This method clears the unknown user and item set. This method should be
        called after the Phase 3 when the data release is done.
        """
        self.unknown_user = set()
        self.unknown_item = set()
        
class AlgorithmStatusWarning(UserWarning):
    def __init__(self, algo_id:UUID, status:AlgorithmStateEnum, phase: str):
        self.algo_id = algo_id
        self.status = status
        if phase == "data_release":
            super().__init__(f"Algorithm:{algo_id} current status is {status}. Algorithm has already requested for data. Returning the same data again.")
        elif phase == "unlabeled":
            super().__init__(f"Algorithm:{algo_id} not ready to get unlabeled data, current status is {status}. Call get_data() first.")
        elif phase == "predict_complete":
            super().__init__(f"Algorithm:{algo_id} current status is {status}. Algorithm already submitted prediction")
        elif phase == "predict":
            super().__init__(f"Algorithm:{algo_id} current status is {status}. Algorithm should request for unlabeled data first.")
        elif phase == "complete":
            super().__init__(f"Algorithm:{algo_id} current status is {status}. Algorithm has completed stream evaluation. No more data release available.")
        super().__init__(f"Algorithm:{algo_id} current status is {status}.")