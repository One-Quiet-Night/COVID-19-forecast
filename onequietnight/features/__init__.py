"""Turn raw data into features for modeling."""

import logging

import pandas as pd
from onequietnight.data.utils import to_matrix
from onequietnight.features import county, national, state, transforms

logger = logging.getLogger(__name__)


def transform_data_to_features(env, data):
    out = {}
    for name, df in data.items():
        logger.info(f"Transforming {name}.")
        out[name] = to_matrix(df)
    return out


def transform_dates(env, features, dates=None):
    all_dates = pd.date_range(env.start_date, env.today, name="dates")
    dates = dates or all_dates
    out = {}
    for name, dm in features.items():
        dm.index = pd.to_datetime(dm.index)
        out[name] = dm.reindex(dates)
    return out


model_names = [national.model_name, state.model_name, county.model_name]

__all__ = [
    "transform_data_to_features",
    "transform_dates",
    "state",
    "county",
    "national",
    "transforms",
    "model_names",
]
