import covidcast
import pandas as pd
from datetime import datetime
from timeit import default_timer as timer
from onequietnight.data.utils import rename_columns_df

import logging

logger = logging.getLogger("onequietnight")

metrics = {
    "Chng_SmoothedOutpatientCovid": [
        {
            "data_source": "chng",
            "signal": "smoothed_outpatient_covid",
            "geo_type": "county",
        },
        {
            "data_source": "chng",
            "signal": "smoothed_outpatient_covid",
            "geo_type": "state",
        },
    ],
    "Safegraph_CompletelyHomeProp": [
        {
            "data_source": "safegraph",
            "signal": "completely_home_prop",
            "geo_type": "county",
        },
        {
            "data_source": "safegraph",
            "signal": "completely_home_prop",
            "geo_type": "state",
        },
    ],
    "Safegraph_FullTimeWorkProp": [
        {
            "data_source": "safegraph",
            "signal": "full_time_work_prop",
            "geo_type": "county",
        },
        {
            "data_source": "safegraph",
            "signal": "full_time_work_prop",
            "geo_type": "state",
        },
    ],
    "Safegraph_MedianHomeDwellTime": [
        {
            "data_source": "safegraph",
            "signal": "median_home_dwell_time",
            "geo_type": "county",
        },
        {
            "data_source": "safegraph",
            "signal": "median_home_dwell_time",
            "geo_type": "state",
        },
    ],
    "Safegraph_PartTimeWorkProp": [
        {
            "data_source": "safegraph",
            "signal": "part_time_work_prop",
            "geo_type": "county",
        },
        {
            "data_source": "safegraph",
            "signal": "part_time_work_prop",
            "geo_type": "state",
        },
    ],
    "Ght_RawSearch": [
        {"data_source": "ght", "signal": "raw_search", "geo_type": "state"},
        {"data_source": "ght", "signal": "raw_search", "geo_type": "msa"},
    ],
    "FbSurvey_RawCli": [
        {"data_source": "fb-survey", "signal": "raw_cli", "geo_type": "county"},
        {"data_source": "fb-survey", "signal": "raw_cli", "geo_type": "state"},
    ],
    "FbSurvey_RawIli": [
        {"data_source": "fb-survey", "signal": "raw_ili", "geo_type": "county"},
        {"data_source": "fb-survey", "signal": "raw_ili", "geo_type": "state"},
    ],
    "FbSurvey_RawWcli": [
        {"data_source": "fb-survey", "signal": "raw_wcli", "geo_type": "county"},
        {"data_source": "fb-survey", "signal": "raw_wcli", "geo_type": "state"},
    ],
    "FbSurvey_RawWili": [
        {"data_source": "fb-survey", "signal": "raw_wili", "geo_type": "county"},
        {"data_source": "fb-survey", "signal": "raw_wili", "geo_type": "state"},
    ],
    "FbSurvey_RawHhCmntyCli": [
        {
            "data_source": "fb-survey",
            "signal": "raw_hh_cmnty_cli",
            "geo_type": "county",
        },
        {"data_source": "fb-survey", "signal": "raw_hh_cmnty_cli", "geo_type": "state"},
    ],
    "FbSurvey_RawNohhCmntyCli": [
        {
            "data_source": "fb-survey",
            "signal": "raw_nohh_cmnty_cli",
            "geo_type": "county",
        },
        {
            "data_source": "fb-survey",
            "signal": "raw_nohh_cmnty_cli",
            "geo_type": "state",
        },
    ],
    "DoctorVisits_SmoothedCli": [
        {
            "data_source": "doctor-visits",
            "signal": "smoothed_cli",
            "geo_type": "county",
        },
        {"data_source": "doctor-visits", "signal": "smoothed_cli", "geo_type": "state"},
    ],
}


def conform_covidcast_id(env, covidcast_df):
    locations_df = env.locations_df
    states_df = locations_df[locations_df["locationType"] == "state"].copy()
    states_df["abbr"] = states_df["abbreviation"].str.lower()
    counties_df = locations_df[locations_df["locationType"] == "county"].copy()

    msa_counties_df = locations_df[["id", "CBSACode"]].dropna()

    covidcast_geo_value_map = pd.concat(
        [
            states_df[["id", "abbr"]].rename(columns={"abbr": "geo_value"}),
            counties_df[["id", "fips.id"]].rename(columns={"fips.id": "geo_value"}),
            msa_counties_df[["id", "CBSACode"]].rename(
                columns={"CBSACode": "geo_value"}
            ),
        ]
    )

    covidcast_df = pd.merge(
        covidcast_df, covidcast_geo_value_map, on="geo_value", how="left"
    ).dropna(subset=["id"])
    return covidcast_df


def extract_df(covidcast_df):
    covidcast_df = covidcast_df.copy()
    covidcast_df = covidcast_df.rename(columns={"time_value": "dates"})
    covidcast_df = covidcast_df.sort_values("issue").groupby(["id", "dates"]).tail(1)
    df = covidcast_df[["dates", "id", "value"]]
    df = df.dropna()
    return df


def load_data(env):
    data = {}
    for name, configs in metrics.items():
        signals = []
        for config in configs:
            start = timer()
            logger.info(
                f"Loading {config['data_source']}_{config['signal']}_{config['geo_type']}"
            )

            covidcast_df = covidcast.signal(
                config["data_source"],
                config["signal"],
                datetime.strptime(env.start_date, "%Y-%m-%d"),
                datetime.strptime(env.today, "%Y-%m-%d"),
                config["geo_type"],
            )
            signals.append(covidcast_df)
            end = timer()
            logger.info("Fetched in {}".format(end - start))

        covidcast_df = pd.concat(signals)
        covidcast_df = conform_covidcast_id(env, covidcast_df)
        df = extract_df(covidcast_df)
        df = df.reset_index(drop=True)
        df = rename_columns_df(df, name)
        data[name] = df
    return data
