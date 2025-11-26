"""
.. currentmodule:: streamsight.utils

Utility
-------------
The utility module provides a set of utility functions that are used in the
Streamsight library. These function are general functions that are not class
specific and also contains functions that are used in the top level such as
preparing the logger and creating the configuration file.

.. autosummary::
    :toctree: generated/

    create_config_yaml
    safe_dir
    add_columns_to_csr_matrix
    add_rows_to_csr_matrix
    arg_to_str
    df_to_sparse
    prepare_logger
    to_binary
    to_tuple
    ProgressBar

Logging
-------------
The logging module provides functions to control the logging level and
suppression of warnings.

Example
~~~~~~~~~

.. code-block:: python

    import streamsight

    # Set log level to INFO and suppress warnings
    streamsight.log_level_by_name("INFO")
    streamsight.suppress_warnings(suppress=True)

    # Log some information
    logger = logging.getLogger("streamsight")
    logger.info("This is an informational message.")

    # Emit a warning (this will be suppressed)
    warnings.warn("This warning will not appear.")

.. autosummary::
    :toctree: generated/

    log_level
    log_level_by_name
    suppress_warnings
    suppress_specific_warnings
"""

from streamsight.utils.logging_tools import (
    log_level,
    log_level_by_name,
    prepare_logger,
    suppress_specific_warnings,
    suppress_warnings,
)
from streamsight.utils.path import (
    get_cache_dir,
    get_data_dir,
    get_logs_dir,
    get_repo_root,
    safe_dir,
)
from streamsight.utils.util import (
    ProgressBar,
    add_columns_to_csr_matrix,
    add_rows_to_csr_matrix,
    arg_to_str,
    df_to_sparse,
    to_binary,
    to_tuple,
)
from streamsight.utils.yaml_tool import create_config_yaml
