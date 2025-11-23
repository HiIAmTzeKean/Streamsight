import logging
import warnings
from contextlib import contextmanager
from enum import Enum
from typing import Generator, Union


class LogLevel(Enum):
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET

    @classmethod
    def from_string(cls, level: str) -> "LogLevel":
        """Case-insensitive lookup from string."""
        return cls[level.upper()]


def log_level(level: Union[int, str, LogLevel]) -> None:
    """
    Change the logging level for root logger.

    :param level: LogLevel enum, level name (case-insensitive), or int.
    """
    if isinstance(level, str):
        level = LogLevel.from_string(level)

    numeric_level = level.value if isinstance(level, LogLevel) else level

    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)


log_level_by_name = log_level  # Alias for convenience


def suppress_warnings() -> None:
    """Suppress all warnings."""
    logging.captureWarnings(False)
    warnings.filterwarnings("ignore")


def enable_warnings() -> None:
    """Enable warnings (reset to default behavior)."""
    logging.captureWarnings(True)
    warnings.resetwarnings()


def suppress_specific_warnings(category: type[Warning]) -> None:
    """Suppress specific warning category."""
    warnings.filterwarnings("ignore", category=category)


@contextmanager
def warnings_suppressed() -> Generator:
    """Temporarily suppress warnings within a context."""
    logging.captureWarnings(False)
    warnings.filterwarnings("ignore")
    try:
        yield
    finally:
        logging.captureWarnings(True)
        warnings.resetwarnings()
