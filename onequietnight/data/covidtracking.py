"""Ingest Covid Tracking Project data.

Load Covid Tracking Project data from C3 AI datalake or from Covid Tracking
and convert it to C3 AI format.

Use the latter if the release is missing from the former.
"""

import pandas as pd
from onequietnight.data import c3ai
from onequietnight.data.utils import rename_columns_df

metrics = [
    "CovidTrackingProject_ConfirmedCases",
    "CovidTrackingProject_ConfirmedDeaths",
    "CovidTrackingProject_ConfirmedHospitalizations",
    "CovidTrackingProject_NegativeTests",
    "CovidTrackingProject_PendingTests",
    "CovidTrackingProject_Ventilator",
    "CovidTrackingProject_ICU",
]


def load_data_covidtracking(env):
    locations_df = env.locations_df

    df = pd.read_json("https://api.covidtracking.com/v1/us/daily.json")
    df["id"] = "UnitedStates"

    df_states = pd.read_json("https://api.covidtracking.com/v1/states/daily.json")
    df_states = pd.merge(
        df_states,
        locations_df[["abbreviation", "id"]],
        left_on="state",
        right_on="abbreviation",
        how="left",
    )

    df = pd.concat([df, df_states])
    df = df.dropna(subset=["id"])
    df["dates"] = pd.to_datetime(df["date"], format="%Y%m%d")

    covidtrackingproject_metrics = {
        "positive": "CovidTrackingProject_ConfirmedCases",
        "death": "CovidTrackingProject_ConfirmedDeaths",
        "hospitalizedCurrently": "CovidTrackingProject_ConfirmedHospitalizations",
        "onVentilatorCurrently": "CovidTrackingProject_Ventilator",
        "inIcuCurrently": "CovidTrackingProject_ICU",
        "negative": "CovidTrackingProject_NegativeTests",
        "pending": "CovidTrackingProject_PendingTests",
    }

    df = df.set_index(["dates", "id"])[covidtrackingproject_metrics.keys()]

    df = df.stack()
    df.index = df.index.rename("name", -1)
    df = df.reset_index(-1, name="value")

    df["name"] = df["name"].map(covidtrackingproject_metrics)

    df = df.sort_index()
    data = {}
    for name, g in df.groupby("name"):
        data[name] = rename_columns_df(g.drop(columns=["name"]), name)
    return data


def load_data(env):
    if env.load_data_covidtracking:
        return load_data_covidtracking(env)
    else:
        return {name: c3ai.load_data(env, name) for name in metrics}
