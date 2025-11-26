import logging
import logging.config
import os
from pathlib import Path
import yaml
import pyfiglet


def get_repo_root() -> Path:
    """Get the repository root directory.

    This assumes the library is installed as src/streamsight/
    and navigates up to find the repo root.

    :return: Path to repository root
    :rtype: Path
    """
    # Get the path to this module
    current_file = Path(__file__).resolve()

    # Navigate up from src/streamsight/utils/logger.py (or wherever this is)
    # Adjust the number of .parent calls based on your file structure
    # If this file is at src/streamsight/utils/logger.py:
    # - .parent -> src/streamsight/utils
    # - .parent.parent -> src/streamsight
    # - .parent.parent.parent -> src
    # - .parent.parent.parent.parent -> repo root

    # Try to find repo root by looking for marker files
    current = current_file.parent
    max_depth = 10  # Prevent infinite loops

    for _ in range(max_depth):
        # Check for common repo root markers
        if any(
            (current / marker).exists()
            for marker in [".git", "pyproject.toml", "setup.py", "setup.cfg", "README.md"]
        ):
            return current

        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent

    # Fallback: assume library is in src/streamsight/
    # Go up 3 levels from this file
    return Path(__file__).resolve().parent.parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory at repo root.

    :return: Path to data directory
    :rtype: Path
    """
    return get_repo_root() / "data"


def get_logs_dir() -> Path:
    """Get the logs directory at repo root.

    :return: Path to logs directory
    :rtype: Path
    """
    return get_repo_root() / "logs"


def safe_dir(dir_path: str | Path) -> None:
    """Ensure directory exists, create if it doesn't.

    :param dir_path: Directory path to check/create
    :type dir_path: str | Path
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)


def create_config_yaml(config_filename: str) -> None:
    """
    Create a configuration file for the logger.

    Writes a default configuration file for the logger in YAML format.
    The configuration file specifies the format of the log messages,
    the output stream, and the log level.

    :param path: The name of the file to be created.
    :type path: str
    """
    # Get the current working directory
    current_directory = os.getcwd()

    # Define the default log file path in the current directory
    log_file_path = os.path.join(current_directory, "logs/streamsight.log")
    yaml_file_path = os.path.join(current_directory, config_filename)

    default_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
            "simple": {"format": "%(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": log_file_path,
            },
        },
        "loggers": {
            "streamsight": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"],
        },
    }

    # Write the YAML content
    with open(yaml_file_path, "w") as file:
        yaml.dump(default_config, file, default_flow_style=False)
