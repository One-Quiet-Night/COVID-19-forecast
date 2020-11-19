import logging
import os

from onequietnight import data, features, models
from onequietnight.data.utils import to_dataframe, to_matrix
from onequietnight.env import OneQuietNightEnvironment
from onequietnight.features import transforms

__all__ = ["data", "features", "models", "transforms", "OneQuietNightEnvironment", "to_dataframe", "to_matrix"]
__version__ = "1.0.0"

# Deterimine log level, INFO by default
LOG_LEVEL = getattr(logging, os.getenv("OQN_LOG_LEVEL", "INFO").upper())

# Set log level
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# Set log stream handler formatting
console_format = logging.Formatter("%(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)
