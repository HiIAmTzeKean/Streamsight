import logging
import warnings
from enum import Enum
from typing import Union


class LogLevel(Enum):
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET


def log_level(level: Union[int, LogLevel]) -> None:
    """
    Change the logging level for root logger.

    :param level: The new logging level (either a `LogLevel` enum member or an int like `logging.DEBUG`).
    """
    if isinstance(level, LogLevel):
        numeric_level = level.value
    elif isinstance(level, int):
        numeric_level = level
    else:
        raise TypeError("level must be an int or LogLevel enum member")

    logger = logging.getLogger()  # Root logger
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)


def log_level_by_name(level_name: Union[str, LogLevel]) -> None:
    """
    Change the logging level using a `LogLevel` enum member or a level name like 'DEBUG', 'INFO'.

    :param level_name: The name of the logging level (e.g., 'DEBUG') or a `LogLevel` member.
    """
    if isinstance(level_name, LogLevel):
        level = level_name.value
    elif isinstance(level_name, str):
        try:
            level = LogLevel[level_name.upper()].value
        except KeyError:
            raise ValueError(f"Invalid logging level name: {level_name}")
    else:
        raise TypeError("level_name must be a str or LogLevel enum member")

    log_level(level)


def suppress_warnings(suppress: bool = True) -> None:
    """
    Enable or disable the suppression of warnings.

    :param suppress: If True, suppress all warnings. If False, allow warnings.
    :type suppress: bool
    """
    if suppress:
        logging.captureWarnings(False)  # Stops capturing warnings into logs
        warnings.filterwarnings("ignore")  # Suppresses all warnings
    else:
        logging.captureWarnings(True)  # Resumes capturing warnings into logs
        warnings.resetwarnings()  # Resets the filter to default behavior


def suppress_specific_warnings(category: type[Warning]):
    """
    Suppress specific warnings by category.

    :param category: The warning category to suppress (e.g., DeprecationWarning).
    :type category: type
    """
    warnings.filterwarnings("ignore", category=category)
