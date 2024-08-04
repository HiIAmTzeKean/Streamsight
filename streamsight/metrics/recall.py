import logging
from typing import Optional

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

from streamsight.metrics.base import ListwiseMetricK


logger = logging.getLogger(__name__)


class RecallK(ListwiseMetricK):
    """Computes the fraction of true interactions that made it into
    the Top-K recommendations.

    Recall per user is computed as:

    .. math::

        \\text{Recall}(u) = \\frac{\\sum\\limits_{i \\in \\text{Top-K}(u)} y^{true}_{u,i} }{\\sum\\limits_{j \\in I} y^{true}_{u,j}}

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(self, K=10, timestamp_limit: Optional[int] = None):
        super().__init__(K,timestamp_limit)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        scores = scipy.sparse.lil_matrix(y_pred_top_K.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(bool)] = 1

        scores = scores.tocsr()

        self._scores = csr_matrix(sparse_divide_nonzero(scores, csr_matrix(y_true.sum(axis=1))).sum(axis=1))

        logger.debug(f"Recall compute complete - {self.name}")
    
def sparse_divide_nonzero(a: csr_matrix, b: csr_matrix) -> csr_matrix:
    """Elementwise divide of nonzero elements of a by nonzero elements of b.

    Elements that were zero in either a or b are zero in the resulting matrix.

    :param a: Numerator.
    :type a: csr_matrix
    :param b: Denominator.
    :type b: csr_matrix
    :return: Result of the elementwise division of matrix a by matrix b.
    :rtype: csr_matrix
    """
    return a.multiply(sparse_inverse_nonzero(b))


def sparse_inverse_nonzero(a: csr_matrix) -> csr_matrix:
    """Invert nonzero elements of a `scipy.sparse.csr_matrix`.

    :param a: Matrix to invert.
    :type a: csr_matrix
    :return: Matrix with nonzero elements inverted.
    :rtype: csr_matrix
    """
    inv_a = a.copy()
    inv_a.data = 1 / inv_a.data
    return inv_a