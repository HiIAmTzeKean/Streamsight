import logging
import logging.config
import os
from typing import Union

import numpy as np
import progressbar
import yaml
from scipy.sparse import csr_matrix, hstack, vstack

from streamsight.utils.directory_tools import create_config_yaml, safe_dir

logger = logging.getLogger(__name__)


def to_tuple(el):
    """Whether single element or tuple, always returns as tuple."""
    if type(el) == tuple:
        return el
    else:
        return (el,)

def arg_to_str(arg: Union[type, str]) -> str:
    if type(arg) == type:
        arg = arg.__name__

    elif type(arg) != str:
        raise TypeError(f"Argument should be string or type, not {type(arg)}!")

    return arg

def df_to_sparse(df, item_ix, user_ix, value_ix=None, shape=None):
    if value_ix is not None and value_ix in df:
        values = df[value_ix]
    else:
        if value_ix is not None:
            # value_ix provided, but not in df
            logger.warning(f"Value column {value_ix} not found in dataframe. Using ones instead.")

        num_entries = df.shape[0]
        # Scipy sums up the entries when an index-pair occurs more than once,
        # resulting in the actual counts being stored. Neat!
        values = np.ones(num_entries)

    indices = list(zip(*df.loc[:, [user_ix, item_ix]].values))

    if indices == []:
        indices = [[], []]  # Empty zip does not evaluate right

    if shape is None:
        shape = df[user_ix].max() + 1, df[item_ix].max() + 1
    sparse_matrix = csr_matrix((values, indices), shape=shape, dtype=values.dtype)

    return sparse_matrix


def to_binary(X: csr_matrix) -> csr_matrix:
    """Converts a matrix to binary by setting all non-zero values to 1.

    :param X: Matrix to convert to binary.
    :type X: csr_matrix
    :return: Binary matrix.
    :rtype: csr_matrix
    """
    X_binary = X.astype(bool).astype(X.dtype)

    return X_binary

def invert(x: Union[np.ndarray, csr_matrix]) -> Union[np.ndarray, csr_matrix]:
    """Invert an array.

    :param x: [description]
    :type x: [type]
    :return: [description]
    :rtype: [type]
    """
    if isinstance(x, np.ndarray):
        ret = np.zeros(x.shape)
    elif isinstance(x, csr_matrix):
        ret = csr_matrix(x.shape)
    else:
        raise TypeError("Unsupported type for argument x.")
    ret[x.nonzero()] = 1 / x[x.nonzero()]
    return ret


class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def prepare_logger(path) -> dict:
    if not os.path.exists(path):
        create_config_yaml(path)

    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    dir_name = os.path.dirname(config['handlers']['file']['filename'])
    safe_dir(dir_name)

    logging.config.dictConfig(config)
    return config

def add_rows_to_csr_matrix(matrix:csr_matrix, n:int=1) -> csr_matrix:
    """Add a row of zeros to a csr_matrix.
    
    ref: https://stackoverflow.com/questions/4695337/expanding-adding-a-row-or-column-a-scipy-sparse-matrix

    :param matrix: Matrix to add a row of zeros to.
    :type matrix: csr_matrix
    :return: Matrix with a row of zeros added.
    :rtype: csr_matrix
    """
    matrix = vstack([matrix,np.zeros((n,matrix.shape[1]))])
    if type(matrix) != csr_matrix:
        # matrix could be in COO format
        return matrix.tocsr()
    return matrix

def add_columns_to_csr_matrix(matrix:csr_matrix, n:int=1) -> csr_matrix:
    """Add a column of zeros to a csr_matrix.
    
    ref: https://stackoverflow.com/questions/60907414/how-to-properly-use-numpy-hstack

    :param matrix: Matrix to add a column of zeros to.
    :type matrix: csr_matrix
    :return: Matrix with a column of zeros added.
    :rtype: csr_matrix
    """
    matrix = hstack([matrix,np.zeros((matrix.shape[0],n))])
    if type(matrix) != csr_matrix:
        # matrix could be in COO format
        return matrix.tocsr()
    return matrix