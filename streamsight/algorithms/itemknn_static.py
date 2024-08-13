import logging
import time
from scipy.sparse import csr_matrix
from streamsight.algorithms.base import Algorithm
from streamsight.algorithms.itemknn import ItemKNN
from streamsight.matrix import InteractionMatrix

logger = logging.getLogger(__name__)

class ItemKNNStatic(ItemKNN):
    def __init__(
        self,
        K=10
    ):
        super().__init__(K)
        self.fit_complete = False

    def fit(self, X: InteractionMatrix) -> "Algorithm":
        if self.fit_complete:
            return self
        
        super().fit(X)
        return self
    
    def _predict(self, X: csr_matrix) -> csr_matrix:
        num_item, num_user = self.similarity_matrix_.shape
        # reduce X to only the items that are in the similarity matrix
        X = X[:num_user, :num_item]
        return super()._predict(X)
