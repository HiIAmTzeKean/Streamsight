from typing import Tuple, Union
from scipy.sparse import csr_matrix
from streamsight.matrix.interation_matrix import InteractionMatrix
from streamsight.utils.util import to_binary

Matrix = Union[InteractionMatrix, csr_matrix]

def to_csr_matrix(
    X: Union[Matrix, Tuple[Matrix, ...]], binary: bool = False
) -> Union[csr_matrix, Tuple[csr_matrix, ...]]:
    """Convert a matrix-like object to a scipy csr_matrix.

    :param X: Matrix-like object or tuple of objects to convert.
    :type X: csr_matrix
    :param binary: If true, ensure matrix is binary by setting non-zero values to 1.
    :type binary: bool, optional
    :return: Matrices as csr_matrix.
    :rtype: Union[csr_matrix, Tuple[csr_matrix, ...]]
    """
    if isinstance(X, (tuple, list)):
        return type(X)(to_csr_matrix(x, binary=binary) for x in X)
    elif isinstance(X, csr_matrix):
        res = X
    elif isinstance(X, InteractionMatrix):
        res = X.values
    else:
        raise AttributeError("Not supported Matrix conversion")
    return to_binary(res) if binary else res
