import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def get_date_partition(env):
    if not env.write:
        return None
    partition = Path(env.base_path, env.today)
    if partition.exists():
        return partition
    else:
        logger.info(f"Making directory {str(partition)}")
        Path.mkdir(partition)
        return partition


def get_data_filename(env, name):
    return Path(name).with_suffix(".feather")


def write_data(env, data):
    assert env.write, "base_path must be specified to read and write data."
    date_partition = get_date_partition(env)
    for name, df in data.items():
        data_path = Path(date_partition, get_data_filename(env, name))
        logger.info(f"Writing to {str(data_path)}")
        df.to_feather(str(data_path))


def read_data(env, name):
    assert env.write, "base_path must be specified to read and write data."
    date_partition = get_date_partition(env)
    data_path = Path(date_partition, get_data_filename(env, name))
    if data_path.exists():
        logger.info(f"Reading from {str(data_path)}")
        return pd.read_feather(str(data_path))
    else:
        logger.info(f"Could not find data at {str(data_path)}")
        raise ValueError(f"Could not find data at {str(data_path)}")
