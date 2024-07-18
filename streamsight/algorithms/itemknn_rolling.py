import logging
import time
from streamsight.algorithms.base import Algorithm
from streamsight.algorithms.itemknn import ItemKNN
from streamsight.matrix.interation_matrix import InteractionMatrix
from streamsight.matrix.util import Matrix

logger = logging.getLogger(__name__)

class ItemKNNRolling(ItemKNN):
    def __init__(
        self,
        K=200,
        normalize_X: bool = False,
        normalize_sim: bool = False
    ):
        super().__init__(K, normalize_X, normalize_sim)