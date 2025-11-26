import logging
import logging.config
import os
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum

import pyfiglet
import yaml

from streamsight.utils.path import safe_dir
from streamsight.utils.yaml_tool import create_config_yaml


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


def log_level(level: int | str | LogLevel) -> None:
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


def prepare_logger(log_config_filename: str) -> dict:
    """Prepare the logger.

    Prepare the logger by reading the configuration file and setting up the logger.
    If the configuration file does not exist, it will be created.

    :param log_config_filename: Name of configuration file.
    :type log_config_filename: str
    :return: Configuration dictionary.
    :rtype: dict
    """
    _, yaml_file_path = create_config_yaml(log_config_filename)
    try:
        with open(yaml_file_path, "r") as stream:
            config = yaml.load(
                stream, Loader=yaml.FullLoader
            )
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {yaml_file_path}.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")

    # Get the log file path from the configuration
    log_file = config["handlers"]["file"]["filename"]

    # Ensure the log file directory exists
    dir_name = os.path.dirname(log_file)
    safe_dir(dir_name)

    # Write ASCII art to the log file
    with open(log_file, "w") as log:
        ascii_art = pyfiglet.figlet_format("streamsight")
        log.write(ascii_art)
        log.write("\n")

    logging.config.dictConfig(config)
    logging.captureWarnings(True)
    return config
