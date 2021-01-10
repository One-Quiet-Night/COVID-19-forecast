"""Ingest and process c3 ai datalake data.

See: https://github.com/c3-e/c3aidatalake-notebooks
"""

import logging
import re

import pandas as pd
import requests
from onequietnight.data.utils import assert_long_df

logger = logging.getLogger(__name__)


def split_evalmetrics_output_name(s):
    """Split evalmetrics string into its component parts.

    We represent evalmetrics outputs using three components: id, metric, field.
    The evalmetrics output name is a concatenation of these components joined
    by dots: '$id.$metric.$field'.

    This function splits a evalmetrics outputs string into those three
    components. For example,
    'Alabama_UnitedStates.AllSex_1_4_TotalDeaths.data' is split into
    ('Alabama_UnitedStates', 'AllSex_1_4_TotalDeaths', 'data')
    """
    if s.count(".") == 2:
        return s.split(".")
    else:
        m = re.match(r"^(.*UnitedStates)\.(.*)$", s)
        matched_id, rem = m[1], m[2]
        m = re.match(r"^(.*?)\.(.*)$", rem)
        matched_metric, matched_field = m[1], m[2]
        return [matched_id, matched_metric, matched_field]


def split_columns(df, inplace=False):
    """Splits evalmetrics columns into multiple levels.

    We represent evalmetrics outputs using three components: id, metric, field.
    The evalmetrics output name is a concatenation of these components joined
    by dots: '$id.$metric.$field'.

    This function splits the columns of a df containing evalmetrics outputs
    as columns into multiple levels. For example,
    'Alabama_UnitedStates.AllSex_1_4_TotalDeaths.data' is split into
    ('Alabama_UnitedStates', 'AllSex_1_4_TotalDeaths', 'data')
    """
    if not inplace:
        df = df.copy()
    df.columns = pd.MultiIndex.from_tuples(
        map(split_evalmetrics_output_name, df.columns),
        names=["id", "metric", "field"],
    )
    return df


def join_columns(df, inplace=False):
    """Joins evalmetrics columns split into multiple levels into one.

    This function reverses the operation for split_columns.
    """
    if not inplace:
        df = df.copy()
    df.columns = [".".join(col).strip() for col in df.columns.values]
    return df


def extract_data(df, filter_missing=True):
    """Extract data from evalmetrics output.

    An evalmetrics output contains the data as well as metadata fields
    associated with the data. This function takes a df containing evalmetrics
    output fields as columns, applies cleaning and conforming, and extracts
    the data series containing just the data field.
    """
    if filter_missing:
        df = df[df["missing"] != 100]
    return df["data"]


def process_df(df):
    """Extracts and conforms evalmetrics output.

    This function normalizes a wide form df containing evalmetrics outputs as
    columns into a long form df where each row contains an observation of a
    single case of a single variable.

    df is formatted as a DataFrame[dates, id; {metric_name}] where it is
    indexed by a multiindex containing `dates` and `id` and contains a single
    column with {metric_name} such as "Male_Under1_CovidDeaths".
    """
    assert ("dates" not in df.columns) and (
        df.index.name == "dates"
    ), "df should be evalmetrics outputs indexed by 'dates'."
    df = df.filter(regex="data|missing")
    df = split_columns(df)
    df = df.stack(["id", "metric"])
    df = extract_data(df)
    df = df.unstack("metric")
    return df


def convert_id_to_location(df, locations_df):
    """Converts c3 `id` to covid forecast hub `location_name`.

    c3 uses `id` as the PK while covid forecast hub uses `location_name`
    (labeled just `location` there instead of `location_name`) as the PK.

    locations_df contains `id` and `location_name` mapping.

    df is a long form df DataFrame[dates, id; {metric_name}].
    """
    assert_long_df(df)
    df = df.copy()
    df["location_name"] = df.index.get_level_values("id").map(
        lambda x: locations_df.set_index("id")["location_name"][x]
    )
    df = df.set_index("location_name", append=True)
    df = df.droplevel("id")
    return df


def read_data_json(typename, api, body):
    """
    read_data_json directly accesses the C3.ai COVID-19 Data Lake APIs using
    the requests library, and returns the response as a JSON, raising an error
    if the call fails for any reason.
    ------
    typename: The type you want to access, i.e. 'OutbreakLocation',
        'LineListRecord', 'BiblioEntry', etc.
    api: The API you want to access, either 'fetch' or 'evalmetrics'.
    body: The spec you want to pass. For examples, see the API documentation.
    """
    response = requests.post(
        "https://api.c3.ai/covid/api/1/" + typename + "/" + api,
        json=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )

    # if request failed, show exception
    if response.status_code != 200:
        raise Exception(response.json()["message"])

    return response.json()


def fetch(typename, body, get_all=False, remove_meta=True):
    """
    fetch accesses the C3.ai COVID-19 Data Lake using read_data_json, and
    converts the response into a Pandas dataframe. fetch is used for all
    non-timeseries data in the C3.ai COVID-19 Data Lake, and will call
    read_data as many times as required to access all of the relevant data for
    a given typename and body.
    ------
    typename: The type you want to access, i.e. 'OutbreakLocation',
        'LineListRecord', 'BiblioEntry', etc.
    body: The spec you want to pass. For examples, see the API documentation.
    get_all: If True, get all records and ignore any limit argument passed in
        the body. If False, use the limit argument passed in the body.
        The default is False.
    remove_meta: If True, remove metadata about each record. If False,
        include it. The default is True.
    """
    if get_all:
        has_more = True
        offset = 0
        limit = 2000
        df = pd.DataFrame()

        while has_more:
            body["spec"].update(limit=limit, offset=offset)
            response_json = read_data_json(typename, "fetch", body)
            new_df = pd.json_normalize(response_json["objs"])
            df = df.append(new_df)
            has_more = response_json["hasMore"]
            offset += limit

    else:
        response_json = read_data_json(typename, "fetch", body)
        df = pd.json_normalize(response_json["objs"])

    if remove_meta:
        df = df.drop(
            columns=[c for c in df.columns if ("meta" in c) | ("version" in c)]
        )

    return df


def evalmetrics(typename, body, get_all=True, remove_meta=True):
    """
    evalmetrics accesses the C3.ai COVID-19 Data Lake using read_data_json, and
    converts the response into a Pandas dataframe.
    evalmetrics is used for all timeseries data in the C3.ai COVID-19 Data
    Lake.
    ------
    typename: The type you want to access, i.e. 'OutbreakLocation',
        'LineListRecord', 'BiblioEntry', etc.
    body: The spec you want to pass. For examples, see the API documentation.
    get_all: If True, get all metrics and ignore limits on number of
        expressions and ids. If False, consider expressions and ids limits.
        The default is False.
    remove_meta: If True, remove metadata about each record. If False, include
        it. The default is True.
    """
    if get_all:
        expressions = body["spec"]["expressions"]
        ids = body["spec"]["ids"]
        df = pd.DataFrame()

        for ids_start in range(0, len(ids), 10):
            for expressions_start in range(0, len(expressions), 4):
                body["spec"].update(
                    ids=ids[ids_start : ids_start + 10],
                    expressions=expressions[expressions_start : expressions_start + 4],
                )
                response_json = read_data_json(typename, "evalmetrics", body)
                new_df = pd.json_normalize(response_json["result"])
                new_df = new_df.apply(pd.Series.explode)
                df = pd.concat([df, new_df], axis=1)

    else:
        response_json = read_data_json(typename, "evalmetrics", body)
        df = pd.json_normalize(response_json["result"])
        df = df.apply(pd.Series.explode)

    # get the useful data out
    if remove_meta:
        df = df.filter(regex="dates|data|missing")

    # only keep one date column
    date_cols = [col for col in df.columns if "dates" in col]
    keep_cols = date_cols[:1] + [col for col in df.columns if "dates" not in col]
    df = df.filter(items=keep_cols).rename(columns={date_cols[0]: "dates"})
    df["dates"] = pd.to_datetime(df["dates"])

    return df


def getprojectionhistory(body, remove_meta=True):
    """
    getprojectionhistory accesses the C3.ai COVID-19 Data Lake using
    read_data_json, and converts the response into a Pandas dataframe.
    ------
    body: The spec you want to pass. For examples, see the API documentation.
    remove_meta: If True, remove metadata about each record. If False,
        include it. The default is True.
    """
    response_json = read_data_json("outbreaklocation", "getprojectionhistory", body)
    df = pd.json_normalize(response_json)
    df = df.apply(pd.Series.explode)

    # get the useful data out
    if remove_meta:
        df = df.filter(regex="dates|data|missing|expr")

    # only keep one date column
    date_cols = [col for col in df.columns if "dates" in col]
    keep_cols = date_cols[:1] + [col for col in df.columns if "dates" not in col]
    df = df.filter(items=keep_cols).rename(columns={date_cols[0]: "dates"})
    df["dates"] = pd.to_datetime(df["dates"])

    # rename columns to simplify naming convention
    df = df.rename(columns=lambda x: x.replace(".value", ""))

    return df


def load_data(
    env, metric, interval="DAY", start_date=None, levels=["country", "state"]
):
    logger.info(f"Loading {metric} from C3 AI OutbreakLocation EvalMetrics.")
    df = evalmetrics(
        "outbreaklocation",
        {
            "spec": {
                "expressions": [metric],
                "ids": [i for level in levels for i in env.locations[level]],
                "interval": interval,
                "start": start_date or env.start_date,
                "end": env.today,
            }
        },
    )
    df = df.set_index("dates")
    df = process_df(df)
    return df
