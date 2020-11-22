"""Create locations flat table.

Locations flat table is a convenience view on C3 ai location data combined with
Covid 19 Forecast Hub (for sharing forecasts) and Census Geographies (for
regional analyses).
"""
import io
import logging

import pandas as pd
import requests
from onequietnight.data.c3ai import fetch

logger = logging.getLogger(__name__)


def format_county_fips_id(fips_id):
    if pd.isna(fips_id):
        return fips_id
    if not isinstance(fips_id, str):
        fips_id = str(fips_id)
    if len(fips_id) != 5:
        fips_id = fips_id.zfill(5)
    return fips_id


def convert_c3ai_to_jhu(df, locations_df):
    return pd.merge(df, locations_df[["id", "location"]], on="id", how="left").drop(
        columns=["id"]
    )


def get_c3ai_locations():
    states = [
        "Alabama_UnitedStates",
        "Alaska_UnitedStates",
        "Arizona_UnitedStates",
        "Arkansas_UnitedStates",
        "California_UnitedStates",
        "Colorado_UnitedStates",
        "Connecticut_UnitedStates",
        "Delaware_UnitedStates",
        "DistrictofColumbia_UnitedStates",
        "Florida_UnitedStates",
        "Georgia_UnitedStates",
        "Hawaii_UnitedStates",
        "Idaho_UnitedStates",
        "Illinois_UnitedStates",
        "Indiana_UnitedStates",
        "Iowa_UnitedStates",
        "Kansas_UnitedStates",
        "Kentucky_UnitedStates",
        "Louisiana_UnitedStates",
        "Maine_UnitedStates",
        "Maryland_UnitedStates",
        "Massachusetts_UnitedStates",
        "Michigan_UnitedStates",
        "Minnesota_UnitedStates",
        "Mississippi_UnitedStates",
        "Missouri_UnitedStates",
        "Montana_UnitedStates",
        "Nebraska_UnitedStates",
        "Nevada_UnitedStates",
        "NewHampshire_UnitedStates",
        "NewJersey_UnitedStates",
        "NewMexico_UnitedStates",
        "NewYork_UnitedStates",
        "NorthCarolina_UnitedStates",
        "NorthDakota_UnitedStates",
        "Ohio_UnitedStates",
        "Oklahoma_UnitedStates",
        "Oregon_UnitedStates",
        "Pennsylvania_UnitedStates",
        "RhodeIsland_UnitedStates",
        "SouthCarolina_UnitedStates",
        "SouthDakota_UnitedStates",
        "Tennessee_UnitedStates",
        "Texas_UnitedStates",
        "Utah_UnitedStates",
        "Vermont_UnitedStates",
        "Virginia_UnitedStates",
        "Washington_UnitedStates",
        "WestVirginia_UnitedStates",
        "Wisconsin_UnitedStates",
        "Wyoming_UnitedStates",
        "UnitedStates",
    ]
    states_df = fetch(
        "outbreaklocation",
        {"spec": {"filter": " || ".join([f'id == "{x}"' for x in states])}},
        get_all=True,
    )
    county_filter = "contains(id, 'UnitedStates') && locationType == 'county'"
    counties_df = fetch(
        "outbreaklocation",
        {"spec": {"filter": county_filter}},
        get_all=True,
    )
    return states_df, counties_df


def get_forecast_hub_locations():
    url = (
        "https://raw.githubusercontent.com/reichlab/"
        "covid19-forecast-hub/master/data-locations/locations.csv"
    )
    url_req = requests.get(url).content
    locations = pd.read_csv(io.StringIO(url_req.decode("utf-8")))
    unused_locations = ["74", "11001"]  # (Minor Outlying Islands, DC clone)
    locations = locations[~locations.location.isin(unused_locations)]
    return locations


def resolve_counties(counties_df, locations):
    counties_df["fips.id"] = counties_df["fips.id"].apply(format_county_fips_id)
    counties_df = pd.merge(
        counties_df,
        locations,
        left_on="fips.id",
        right_on="location",
        how="inner",
    )
    return counties_df


def resolve_states(states_df, locations):
    states_df["name"] = states_df["name"].replace(
        {
            "UnitedStates": "US",
            "DistrictofColumbia": "District of Columbia",
            "NewHampshire": "New Hampshire",
            "NewJersey": "New Jersey",
            "NewMexico": "New Mexico",
            "NewYork": "New York",
            "NorthCarolina": "North Carolina",
            "NorthDakota": "North Dakota",
            "RhodeIsland": "Rhode Island",
            "SouthCarolina": "South Carolina",
            "SouthDakota": "South Dakota",
            "WestVirginia": "West Virginia",
        }
    )

    states_df = pd.merge(
        states_df,
        locations,
        left_on="name",
        right_on="location_name",
        how="left",
    )
    states_df["locationType"] = states_df["locationType"].fillna("state")
    return states_df


def get_cbsa():
    url = (
        "https://www2.census.gov/programs-surveys/metro-micro/"
        "geographies/reference-files/2020/delineation-files/list1_2020.xls"
    )
    url_req = requests.get(url).content
    cbsa = pd.read_excel(url_req, skiprows=2, skipfooter=4, dtype={'CBSA Code': str})
    cbsa["FIPS"] = cbsa["FIPS State Code"].astype(str).apply(
        lambda s: s.zfill(2)
    ) + cbsa["FIPS County Code"].astype(str).apply(lambda s: s.zfill(3))
    cbsa["FIPS"] = cbsa["FIPS"].replace({"11001": "11"})
    cbsa = cbsa.rename(
        columns={
            "CBSA Code": "CBSACode",
            "CBSA Title": "CBSA",
            "CSA Title": "CSA",
            "FIPS": "fips.id",
            "Metropolitan/Micropolitan Statistical Area": "CBSAType",
        }
    )
    return cbsa


def get_locations():
    logger.info("Fetching locations information.")
    states_df, counties_df = get_c3ai_locations()
    locations = get_forecast_hub_locations()

    counties_df = resolve_counties(counties_df, locations)
    states_df = resolve_states(states_df, locations)

    locations_df = pd.concat([states_df, counties_df])
    locations_df["fips.id"] = locations_df["fips.id"].combine_first(
        locations_df["location"]
    )

    cbsa = get_cbsa()
    df = pd.merge(
        locations_df,
        cbsa[["CBSACode", "CBSA", "CBSAType", "CSA", "fips.id"]],
        on="fips.id",
        how="left",
    )
    df_cols = [
        "id",
        "fips.id",
        "location_name",
        "abbreviation",
        "location",
        "locationType",
        "population",
        "latestLaborForce",
        "hospitalLicensedBeds",
        "hospitalStaffedBeds",
        "CBSACode",
        "CBSA",
        "CBSAType",
        "CSA",
    ]
    df = df[df_cols]
    return df
