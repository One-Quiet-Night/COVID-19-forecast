"""Ingest Google data.

Load Google data from C3 AI datalake or from Google and convert it to C3 AI format.

Use the latter if the release is missing from the former.
"""

import io

import pandas as pd
import requests
from bs4 import BeautifulSoup
from onequietnight.data import c3ai
from onequietnight.data.locations import format_county_fips_id
from onequietnight.data.utils import rename_columns_df

metrics = [
    "Google_RetailMobility",
    "Google_GroceryMobility",
    "Google_ParksMobility",
    "Google_TransitStationsMobility",
    "Google_WorkplacesMobility",
    "Google_ResidentialMobility",
]


def get_google_link():
    """Get link of Google Community Mobility report file
       Returns:
           link (str): link of Google Community report file

    See: https://github.com/ActiveConclusion/COVID19_mobility
    """
    # get webpage source
    url = "https://www.google.com/covid19/mobility/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    csv_tag = soup.find("a", {"class": "icon-link"})
    link = csv_tag["href"]
    return link


def resolve_google_ids(df, locations_df):
    national_df = df[(df["sub_region_1"].isna())].copy()

    state_df = df[~(df["sub_region_1"].isna()) & (df["sub_region_2"].isna())].copy()

    county_df = df[~(df["sub_region_2"].isna())].copy()

    county_df["fips.id"] = county_df["census_fips_code"].apply(
        lambda x: format_county_fips_id(str(int(x)))
    )
    county_df = pd.merge(
        county_df, locations_df[["fips.id", "id"]], on="fips.id", how="left"
    )

    state_df = pd.merge(
        state_df,
        locations_df[["location_name", "id"]],
        left_on=["sub_region_1"],
        right_on=["location_name"],
        how="left",
    )

    national_df["id"] = national_df["country_region"].str.replace(" ", "")
    return pd.concat([national_df, state_df, county_df])


def load_data_google(env):
    locations_df = env.locations_df

    url_req = requests.get(get_google_link()).content
    df = pd.read_csv(io.StringIO(url_req.decode("utf-8")))

    df = df[df["country_region_code"] == "US"]

    df = resolve_google_ids(df, locations_df)

    cols = [
        "date",
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
        "id",
    ]

    df = df[cols]
    df = df.rename(columns={"date": "dates"})
    df = df.set_index(["dates", "id"])

    df = df.stack()
    df.index = df.index.rename("name", level=-1)
    df = df.reset_index(-1, name="value")

    df["name"] = df["name"].map(
        {
            "retail_and_recreation_percent_change_from_baseline": "Google_RetailMobility",
            "grocery_and_pharmacy_percent_change_from_baseline": "Google_GroceryMobility",
            "parks_percent_change_from_baseline": "Google_ParksMobility",
            "transit_stations_percent_change_from_baseline": "Google_TransitStationsMobility",
            "workplaces_percent_change_from_baseline": "Google_WorkplacesMobility",
            "residential_percent_change_from_baseline": "Google_ResidentialMobility",
        }
    )

    data = {}
    for name, g in df.groupby("name"):
        data[name] = rename_columns_df(g.drop(columns=["name"]), name)

    return data


def load_data(env):
    if env.load_data_google:
        return load_data_google(env)
    else:
        return {
            name: c3ai.load_data(env, name, levels=["country", "state", "county"])
            for name in metrics
        }
