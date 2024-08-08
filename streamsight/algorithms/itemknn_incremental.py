import logging
import time
from streamsight.algorithms.base import Algorithm
from streamsight.algorithms.itemknn import ItemKNN
from streamsight.matrix import InteractionMatrix

logger = logging.getLogger(__name__)

class ItemKNNIncremental(ItemKNN):
    def __init__(
        self,
        K=10
    ):
        super().__init__(K)
        self.historical_data : InteractionMatrix
    
    def fit(self, X: InteractionMatrix) -> "Algorithm":
        start = time.time()
        if not hasattr(self, "historical_data"):
            self.historical_data = X
        else:
            self.historical_data = self.historical_data + X
            
        X = self._transform_fit_input(self.historical_data)
        self._fit(X)

        self._check_fit_complete()
        end = time.time()
        logger.info(f"Fitting {self.name} complete - Took {end - start :.3}s")
        return self