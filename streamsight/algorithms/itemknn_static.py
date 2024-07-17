import logging
import time
from streamsight.algorithms.base import Algorithm
from streamsight.algorithms.itemknn import ItemKNN
from streamsight.matrix.interation_matrix import InteractionMatrix
from streamsight.matrix.util import Matrix

logger = logging.getLogger(__name__)

class ItemKNNStatic(ItemKNN):
    def __init__(
        self,
        K=200,
        normalize_X: bool = False,
        normalize_sim: bool = False
    ):
        super().__init__(K, normalize_X, normalize_sim)
        self.fit_complete = False
    
    def fit(self, X: InteractionMatrix) -> "Algorithm":
        if self.fit_complete:
            return self
        start = time.time()
        X = self._transform_fit_input(X)
        self._fit(X)

        self._check_fit_complete()
        end = time.time()
        logger.info(f"Fitting {self.name} complete - Took {end - start :.3}s")
        self.fit_complete = True
        return self