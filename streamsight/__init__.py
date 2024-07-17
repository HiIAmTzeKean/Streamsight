import os
import logging
import logging.config
from pathlib import Path
import yaml

from streamsight.utils.directory_tools import create_config_yaml, safe_dir

cwd = os.getcwd()
LOGGING_CONFIG = "LOGGING_CONFIG.yaml"

if not os.path.exists(LOGGING_CONFIG):
    create_config_yaml(LOGGING_CONFIG)

with open(LOGGING_CONFIG, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

dir_name = os.path.dirname(config['handlers']['file']['filename'])
safe_dir(dir_name)

logging.config.dictConfig(config)

logger = logging.getLogger(__name__)
logger.debug("Logging is configured.")
logger.info("Logging started")
logger.warning("Logging warning works")