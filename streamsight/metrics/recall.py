import logging

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

    ref: RecPack
    
    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        scores = scipy.sparse.lil_matrix(y_pred_top_K.shape)

        # obtain true positives
        scores[y_pred_top_K.multiply(y_true).astype(bool)] = 1
        scores = scores.tocsr()

        # true positive/total actual interactions
        self._scores = csr_matrix(scores.sum(axis=1) / y_true.sum(axis=1))

        logger.debug(f"Recall compute complete - {self.name}")

    def cache_values(self, y_true: csr_matrix, y_pred: csr_matrix):
        """Caches the true positive and false negative values for the metric.

        :param y_true: True user-item interaction matrix.
        :type y_true: csr_matrix
        :param y_pred: Predicted user-item interaction matrix.
        :type y_pred: csr_matrix
        :raises ValueError: If caching is disabled for the metric.
        """
        if not self.cache:
            raise ValueError("Caching is disabled for this metric.")

        y_true, y_pred_top_K = self.prepare_matrix(y_true, y_pred)

        scores = scipy.sparse.lil_matrix(y_pred_top_K.shape)

        scores[y_pred_top_K.multiply(y_true).astype(bool)] = 1
        scores = scores.tocsr()

        self._scores = csr_matrix(scores.sum(axis=1) / y_true.sum(axis=1))

        tp = scores.sum(axis=1).sum()
        fn = y_true.sum(axis=1).sum() - tp

        if not hasattr(self, "_true_positive") or not hasattr(
            self, "_false_negative"
        ):
            self._true_positive = tp
            self._false_negative = fn
        else:
            self._true_positive += tp
            self._false_negative += fn

    def calculate_cached(self):
        """Calculates the metric using the cached values.

        :raises ValueError: If caching is disabled for the metric.
        """
        if not self.cache:
            raise ValueError("Caching is disabled for this metric.")

        if not hasattr(self, "_true_positive") or not hasattr(
            self, "_false_negative"
        ):
            self._scores = None
            return

        self._scores = csr_matrix(
            self._true_positive / (self._false_negative + self._true_positive)
        )
