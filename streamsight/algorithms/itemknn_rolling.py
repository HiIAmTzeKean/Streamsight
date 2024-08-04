import logging
from streamsight.algorithms.itemknn import ItemKNN

logger = logging.getLogger(__name__)

class ItemKNNRolling(ItemKNN):
    def __init__(
        self,
        K=200
    ):
        super().__init__(K)