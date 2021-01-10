import numpy as np
import pandas as pd


def assert_long_df(df):
    """Assert that a df is a long df.

    long df is formatted as a DataFrame[dates, id; {metric_name}] where it is
    indexed by a multiindex containing `dates` and `id` and contains a single
    column with {metric_name} such as "Male_Under1_CovidDeaths".
    """
    has_panel_index = df.index.names == pd.core.indexes.frozen.FrozenList(["dates", "id"])
    assert (
        isinstance(df, pd.Series) and has_panel_index
    ), "df should be a metric indexed by 'dates' and 'id'."


def denormalize_data(df):
    """Converts from long form to wide form.

    This function denormalizes a long form data series to a wide form df.
    """
    assert_long_df(df)
    try:
        df = df.unstack("id")
    except ValueError:
        df = df[~df.index.duplicated(keep="last")]
        df = df.unstack("id")
    return df


def rename_columns_df(df, value="value"):
    """Rename columns of a long DataFrame to [dates, id, value].

    This is used to rename the value column as value. Columns are named
    [dates, id, <name of the data>]. This makes it tricky for writing
    certain feature engineering functions. This renames it such that the
    data column is named value.
    """
    if isinstance(df, pd.Series):
        df = df.reset_index()
    if set(["dates", "id", value]) == set(df.columns):
        return df
    assert len(df.columns) == 3 or len(df.reset_index().columns) == 3
    if len(df.columns) != 3:
        df = df.reset_index()
    else:
        df = df.copy()
    assert set(["dates", "id"]).issubset(df.columns)
    df.columns = [x if x in ["dates", "id"] else value for x in df.columns]
    return df


def to_matrix(df):
    df = rename_columns_df(df)
    dm = denormalize_data(df.set_index(["dates", "id"])["value"])
    return dm


def to_dataframe(dm, value="value"):
    df = dm.stack("id", dropna=False)
    df = rename_columns_df(df, value)
    df = df.set_index(["dates", "id"])
    return df


def clean(df):
    """General cleaning function
    Replace inf with nans.
    Drop rows or columns containing all NaNs.
    """
    out = df.replace([-np.inf, np.inf], np.nan)
    return out.dropna(1, how="all").dropna(0, how="all")
