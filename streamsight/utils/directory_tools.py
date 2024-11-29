import os

import yaml


def safe_dir(path):
    """Check if directory is safe
    
    Check if the directory exists, if not create it.

    :param path: The path to the directory.
    :type path: str
    """
    if not os.path.exists(path):
        os.makedirs(path)


def create_config_yaml(path: str):
    """
    Create a configuration file for the logger.
    
    Writes a default configuration file for the logger in YAML format.
    The configuration file specifies the format of the log messages,
    the output stream, and the log level.
    
    :param path: The name of the file to be created.
    :type path: str
    """
    default_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            },
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
                "filename": "logs/streamsight.log",
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
    with open(path, "w") as file:
        yaml.dump(default_config, file, default_flow_style=False)
