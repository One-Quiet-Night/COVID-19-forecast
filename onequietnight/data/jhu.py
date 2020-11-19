"""Ingest John Hopkins University Covid-19 data.

Load covid data from C3 AI datalake or from JHU and convert it to C3 AI format.

See: https://github.com/reichlab/covid19-forecast-hub/data-truth
"""

import io

import pandas as pd
import requests
from onequietnight.data import c3ai

metrics = ["JHU_ConfirmedCases", "JHU_ConfirmedDeaths"]


def resolve_csse_id(df, locations_df, c3_id, csse_id):
    """Conforms CSSE data to C3 data.

    This function uses csse_id to identify the c3_id.
    """
    df = pd.merge(
        locations_df[[c3_id, "id"]],
        df.reset_index(),
        left_on=c3_id,
        right_on=csse_id,
        how="inner",
    )

    out = df.drop(columns=[csse_id, c3_id]).set_index("id").T
    out.index = pd.DatetimeIndex(out.index, name="dates")
    return out


def conform_csse(metric_name, state_df, county_df, locations_df):
    """Conforms CSSE data to C3 data.

    This function conforms both the state and county level CSSE data.
    """
    df = pd.concat(
        [
            resolve_csse_id(state_df, locations_df, "location_name", "index"),
            resolve_csse_id(county_df, locations_df, "fips.id", "FIPS"),
        ],
        axis=1,
    )
    df = df.stack("id").reset_index(name=metric_name)
    return df


def get_county_truth(df):
    """Format county data.

    From covid19-forecast-hub/data-truth/get-truth-data.py."""
    county = df[pd.notnull(df.FIPS)]
    county = county[(county.FIPS >= 100) & (county.FIPS < 80001)]
    county.FIPS = (county.FIPS.astype(int)).map("{:05d}".format)
    county_agg = county.groupby(["FIPS"]).sum()
    return county_agg


def get_truth(url):
    """Get data from CSSE.

    From covid19-forecast-hub/data-truth/get-truth-data.py."""
    url_req = requests.get(url).content
    df = pd.read_csv(io.StringIO(url_req.decode("utf-8")))

    # aggregate by state and nationally
    state_agg = df.groupby(["Province_State"]).sum()
    us_nat = df.groupby(["Country_Region"]).sum()
    county_agg = get_county_truth(df)
    df_state_nat = state_agg.append(us_nat)

    # drop unnecessary columns
    df_state_nat_truth = df_state_nat.drop(
        df_state_nat.columns[list(range(0, 6))], axis=1
    )
    df_county_truth = county_agg.drop(county_agg.columns[list(range(0, 5))], axis=1)

    df_state_nat_truth_cumulative = df_state_nat_truth
    df_county_truth_cumulative = df_county_truth

    df_state_nat_truth_incident = (
        df_state_nat_truth_cumulative
        - df_state_nat_truth_cumulative.shift(periods=1, axis="columns")
    )
    df_county_truth_incident = (
        df_county_truth_cumulative
        - df_county_truth_cumulative.shift(periods=1, axis="columns")
    )

    # lower bound truth values to 0.0
    df_state_nat_truth_incident = df_state_nat_truth_incident.clip(lower=0.0)
    df_county_truth_incident = df_county_truth_incident.clip(lower=0.0)

    return (
        df_state_nat_truth_cumulative,
        df_state_nat_truth_incident,
        df_county_truth_cumulative,
        df_county_truth_incident,
    )


def load_data_jhu(env):
    locations_df = env.locations_df

    county_url = (
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/"
        "master/csse_covid_19_data/csse_covid_19_time_series/"
        "time_series_covid19_deaths_US.csv"
    )
    state_nat_cum_death, _, county_cum_death, _ = get_truth(url=county_url)

    state_url = (
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/"
        "master/csse_covid_19_data/csse_covid_19_time_series/"
        "time_series_covid19_confirmed_US.csv"
    )
    state_nat_cum_case, _, county_cum_case, _ = get_truth(url=state_url)

    name_case = "JHU_ConfirmedCases"
    name_death = "JHU_ConfirmedDeaths"
    data = {
        name_case: conform_csse(
            name_case, state_nat_cum_case, county_cum_case, locations_df
        ),
        name_death: conform_csse(
            name_death, state_nat_cum_death, county_cum_death, locations_df
        ),
    }
    return data


def load_data(env):
    if env.load_data_jhu:
        return load_data_jhu(env)
    else:
        return {name: c3ai.load_data(env, name, levels=['country', 'state', 'county']) for name in metrics}
