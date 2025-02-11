import logging
import numpy as np
from scipy.sparse import csr_matrix

from streamsightv2.metrics.base import ListwiseMetricK
from streamsightv2.metrics.util import sparse_divide_nonzero

logger = logging.getLogger(__name__)


class DCGK(ListwiseMetricK):
    """Computes the sum of gains of all items in a recommendation list.

    Relevant items that are ranked higher in the Top-K recommendations have a higher gain.

    The Discounted Cumulative Gain (DCG) is computed for every user as

    .. math::

        \\text{DiscountedCumulativeGain}(u) = \\sum\\limits_{i \\in Top-K(u)} \\frac{y^{true}_{u,i}}{\\log_2 (\\text{rank}(u,i) + 1)}

    ref: RecPack
        
    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
      # log number of users and ground truth interactions
      logger.debug(f"DCGK compute started - {self.name}")
      logger.debug(f"Number of users: {y_true.shape[0]}")
      logger.debug(f"Number of ground truth interactions: {y_true.nnz}")

      denominator = y_pred_top_K.multiply(y_true)
      # Denominator: log2(rank_i + 1)
      denominator.data = np.log2(denominator.data + 1)
      # Binary relevance
      # Numerator: rel_i
      numerator = y_true

      dcg = sparse_divide_nonzero(numerator, denominator)

      self._scores = csr_matrix(dcg.sum(axis=1))

      logger.debug(f"DCGK compute complete - {self.name}")

