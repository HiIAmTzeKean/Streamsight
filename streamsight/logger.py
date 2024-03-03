import os
import logging
import logging.config
import yaml

from streamsight.utils import safe_dir

LOGGING_CONFIG = "./streamsight/LOGGING_CONFIG.yaml"

with open(LOGGING_CONFIG, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

dir_name = os.path.dirname(config['handlers']['file']['filename'])
safe_dir(dir_name)

logging.config.dictConfig(config)

log = logging.getLogger(__name__)
log.debug("Logging is configured.")
log.info("Logging started")
log.warning("Logging warning works")
