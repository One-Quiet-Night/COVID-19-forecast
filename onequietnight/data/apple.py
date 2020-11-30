"""Ingest Apple data.

Load Apple data from C3 AI Data Lake or from Apple and convert to C3 AI format.

Use the latter if the release is missing from the former.
"""

import io
import json
import urllib.request

import pandas as pd
import requests
from onequietnight.data import c3ai
from onequietnight.data.utils import to_dataframe

metrics = [
    "Apple_DrivingMobility",
    "Apple_TransitMobility",
    "Apple_WalkingMobility",
]


def get_apple_link():
    """Get link of Apple Mobility Trends report file
       Returns:
           link (str): link of Apple Mobility Trends report file

    See: https://github.com/ActiveConclusion/COVID19_mobility
    """
    # get link via API
    json_link = (
        "https://covid19-static.cdn-apple.com/"
        "covid19-mobility-data/current/v3/index.json"
    )
    with urllib.request.urlopen(json_link) as url:
        json_data = json.loads(url.read().decode())
    link = (
        "https://covid19-static.cdn-apple.com"
        + json_data["basePath"]
        + json_data["regions"]["en-us"]["csvPath"]
    )
    return link


def to_id(row):
    return f"{row['region_formatted']}_{row['sub_region_formatted']}_UnitedStates"


def resolve_apple_counties(df):
    df = df.copy()
    df["region_formatted"] = (
        df["region"].str.replace("Parish|County|Borough", "")
    ).str.strip()
    df["region_formatted"] = df["region_formatted"].str.replace(" ", "")
    df["sub_region_formatted"] = df["sub-region"].str.replace(" ", "")
    df["id"] = df.apply(to_id, 1)
    df["id"] = df["id"].replace(
        {
            "AnchorageMunicipality_Alaska_UnitedStates": "Anchorage_Alaska_UnitedStates",
            "Carson_Nevada_UnitedStates": "CarsonCity_Nevada_UnitedStates",
            "DoñaAna_NewMexico_UnitedStates": "DonaAna_NewMexico_UnitedStates",
            "James_Virginia_UnitedStates": "JamesCity_Virginia_UnitedStates",
            "Juneauand_Alaska_UnitedStates": "Juneau_Alaska_UnitedStates",
            "AlexandriaCity_Virginia_UnitedStates": "Alexandria_Virginia_UnitedStates",
            "BristolCity_Virginia_UnitedStates": "Bristol_Virginia_UnitedStates",
            "PortsmouthCity_Virginia_UnitedStates": "Portsmouth_Virginia_UnitedStates",
            "FredericksburgCity_Virginia_UnitedStates": "Fredericksburg_Virginia_UnitedStates",
            "HopewellCity_Virginia_UnitedStates": "Hopewell_Virginia_UnitedStates",
            "ManassasCity_Virginia_UnitedStates": "Manassas_Virginia_UnitedStates",
            "VirginiaBeachCity_Virginia_UnitedStates": "VirginiaBeach_Virginia_UnitedStates",
            "HarrisonburgCity_Virginia_UnitedStates": "Harrisonburg_Virginia_UnitedStates",
            "ManassasParkCity_Virginia_UnitedStates": "ManassasPark_Virginia_UnitedStates",
            "WinchesterCity_Virginia_UnitedStates": "Winchester_Virginia_UnitedStates",
            "WaynesboroCity_Virginia_UnitedStates": "Waynesboro_Virginia_UnitedStates",
            "CharlottesvilleCity_Virginia_UnitedStates": "Charlottesville_Virginia_UnitedStates",
            "NorfolkCity_Virginia_UnitedStates": "Norfolk_Virginia_UnitedStates",
            "ChesapeakeCity_Virginia_UnitedStates": "Chesapeake_Virginia_UnitedStates",
            "ColonialHeightsCity_Virginia_UnitedStates": "ColonialHeights_Virginia_UnitedStates",
            "WilliamsburgCity_Virginia_UnitedStates": "Williamsburg_Virginia_UnitedStates",
            "PetersburgCity_Virginia_UnitedStates": "Petersburg_Virginia_UnitedStates",
            "FallsChurchCity_Virginia_UnitedStates": "FallsChurch_Virginia_UnitedStates",
            "StauntonCity_Virginia_UnitedStates": "Staunton_Virginia_UnitedStates",
            "HamptonCity_Virginia_UnitedStates": "Hampton_Virginia_UnitedStates",
            "SalemCity_Virginia_UnitedStates": "Salem_Virginia_UnitedStates",
            "SuffolkCity_Virginia_UnitedStates": "Suffolk_Virginia_UnitedStates",
            "MartinsvilleCity_Virginia_UnitedStates": "Martinsville_Virginia_UnitedStates",
            "DanvilleCity_Virginia_UnitedStates": "Danville_Virginia_UnitedStates",
            "NewportNewsCity_Virginia_UnitedStates": "NewportNews_Virginia_UnitedStates",
            "LynchburgCity_Virginia_UnitedStates": "Lynchburg_Virginia_UnitedStates",
        }
    )

    return df.drop(columns=["region_formatted", "sub_region_formatted"])


def resolve_apple_states(df):
    df = df.copy()
    df["id"] = df["region"].str.replace(" ", "") + "_UnitedStates"
    return df


def resolve_apple_national(df):
    df = df.copy()
    df["id"] = df["region"].str.replace(" ", "")
    return df


def load_data_apple(env):
    url_req = requests.get(get_apple_link()).content
    df = pd.read_csv(io.StringIO(url_req.decode("utf-8")), low_memory=False)

    national_df = df[df["region"] == "United States"]

    counties_df = df[
        (df["country"] == "United States")
        & (df["sub-region"] != "Puerto Rico")
        & (df["geo_type"] == "county")
    ]

    states_df = df[
        (df["country"] == "United States") & (df["geo_type"] == "sub-region")
    ]

    df = pd.concat(
        [
            resolve_apple_national(national_df),
            resolve_apple_states(states_df),
            resolve_apple_counties(counties_df),
        ]
    )

    df["name"] = df["transportation_type"].map(
        {
            "driving": "Apple_DrivingMobility",
            "transit": "Apple_TransitMobility",
            "walking": "Apple_WalkingMobility",
        }
    )

    df = df.drop(
        columns=[
            "geo_type",
            "region",
            "transportation_type",
            "alternative_name",
            "sub-region",
            "country",
        ]
    )
    data = {}
    for name, g in df.groupby("name"):
        dm = g.drop(columns="name").set_index("id").T
        dm.index = dm.index.rename("dates")
        data[name] = to_dataframe(dm, name).reset_index()
    return data


def load_data(env):
    if env.load_data_apple:
        return load_data_apple(env)
    else:
        return {
            name: c3ai.load_data(env, name, levels=["country", "state", "county"])
            for name in metrics
        }
