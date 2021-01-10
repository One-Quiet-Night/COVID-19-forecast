"""Feature transforms.

All functions take a data matrix `dm` (wide format dataframe for features)
as input and return a transformed data matrix.
"""

import numpy as np
import pandas as pd
from onequietnight.data.utils import rename_columns_df, to_dataframe, to_matrix


def normalize_beds(dm, locations_df):
    df = to_dataframe(dm)
    df = pd.merge(
        rename_columns_df(df),
        locations_df[["id", "hospitalLicensedBeds"]],
        on="id",
        how="left",
    )
    df["value"] = df["value"] / df["hospitalLicensedBeds"]
    df = df.drop(columns="hospitalLicensedBeds")
    return to_matrix(df)


def normalize_cases(dm, locations_df):
    df = to_dataframe(dm)
    df = pd.merge(
        rename_columns_df(df),
        locations_df[["id", "population"]],
        on="id",
        how="left",
    )
    df["value"] = df["value"] / df["population"]
    df["value"] = df["value"] * 1e5
    df = df.drop(columns="population")
    return to_matrix(df)


def undo_normalize_cases(dm, locations_df):
    df = to_dataframe(dm)
    df = rename_columns_df(df)
    df = undo_normalize_cases_df(df, locations_df)
    return to_matrix(df)


def undo_normalize_cases_df(df, locations_df):
    df = pd.merge(
        df,
        locations_df[["id", "population"]],
        on="id",
        how="left",
    )
    df["value"] = df["value"] / 1e5
    df["value"] = df["value"] * df["population"]
    df = df.drop(columns="population")
    return df


def select_universe(dm, universe, fill_missing=False):
    """Select universe: a list of counties 'id's"""
    if fill_missing:
        missing = [x for x in universe if x not in dm.columns]
        dm = dm.assign(**{str(c): np.nan for c in missing})
        return dm[universe]
    return dm[dm.columns[dm.columns.isin(universe)]]


def winsor(x, threshold=3):
    lower = x.mean() - threshold * x.std()
    upper = x.mean() + threshold * x.std()
    return x.clip(lower, upper)


def winsor_robust(x, threshold=3):
    threshold_robust = threshold / 1.4826
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lower = q2 - threshold_robust * iqr
    upper = q2 + threshold_robust * iqr
    return x.clip(lower, upper)


def cross_section_winsor(dm, threshold=3):
    """Cross sectional winsorization"""
    assert dm.index.name == "dates"
    location = dm.mean(1)
    spread = dm.std(1)
    lower = location - threshold * spread
    upper = location + threshold * spread
    return dm.clip(lower, upper, axis=0)


def cross_section_mean(dm):
    """Compute mean across all counties."""
    df = to_dataframe(dm)
    df = df.groupby("dates")["value"].transform(np.nanmean)
    return to_matrix(df)


def cross_section_cbsa_mean(dm, locations_df):
    """Compute mean within CBSA.

    Group counties without membership as a group using groupby(dropna=False).
    """
    df = to_dataframe(dm)
    df = df.reset_index("dates").join(locations_df.set_index("id")["CBSA"])
    df = df.set_index("dates", append=True)
    df = df.groupby(["dates", "CBSA"], dropna=True)["value"].transform(np.nanmean)
    return to_matrix(df)


def cross_section_state_mean(dm):
    """Compute mean within state level."""
    df = to_dataframe(dm)
    df = df.reset_index("id")
    df["state"] = df["id"].apply(lambda s: s.split("_")[1])
    df = df.set_index("id", append=True)
    df = df.groupby(["dates", "state"])["value"].transform(np.mean)
    return to_matrix(df)


def get_state_value(dm, dm_state):
    df = to_dataframe(dm)
    df = df.reset_index("id")
    df["state"] = df["id"].apply(lambda s: "_".join(s.split("_")[1:]))
    df = df.set_index("id", append=True)
    df = df.join(to_dataframe(dm_state), on=["dates", "state"], rsuffix="_state")
    return to_matrix(df["value_state"])


def get_national_value(dm, dm_national):
    if dm_national.empty:
        return dm.assign(value=np.nan)
    df = to_dataframe(dm)
    df = pd.merge(
        df.reset_index(),
        to_dataframe(dm_national).reset_index(),
        on=["dates"],
        suffixes=("", "_national"),
    )
    return to_matrix(df.set_index(["dates", "id"])["value_national"])
