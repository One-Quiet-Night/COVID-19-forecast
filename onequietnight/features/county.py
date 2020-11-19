import logging

import pandas as pd
from onequietnight.features.transforms import (cross_section_cbsa_mean,
                                               cross_section_mean,
                                               cross_section_state_mean,
                                               cross_section_winsor,
                                               get_state_value,
                                               normalize_cases,
                                               select_universe)

logger = logging.getLogger(__name__)

model_name = "county"


def transform_features(env, features, freq="W-SAT"):
    """Transform county-level predictors.

    Note: Do not refactor. Create features explicitly one by one instead of
    using a kitchen-sink approach.
    """
    logger.info("Processing county features.")
    dates = pd.date_range(env.start_date, env.today, freq=freq, name="dates")

    out = {}
    out["JHU_ConfirmedCases.diff(7)"] = (
        features["JHU_ConfirmedCases"].diff(7).reindex(dates)
    )
    out["JHU_ConfirmedCases.diff(7).shift(7)"] = (
        features["JHU_ConfirmedCases"].diff(7).shift(7).reindex(dates)
    )

    out["JHU_ConfirmedDeaths.diff(7)"] = (
        features["JHU_ConfirmedDeaths"].diff(7).reindex(dates)
    )
    out["JHU_ConfirmedDeaths.diff(7).shift(7)"] = (
        features["JHU_ConfirmedDeaths"].diff(7).shift(7).reindex(dates)
    )

    out["Apple_DrivingMobility.rolling(7).mean()"] = (
        features["Apple_DrivingMobility"].rolling(7).mean().reindex(dates)
    )
    out["Apple_DrivingMobility.rolling(7).mean().shift(7)"] = (
        features["Apple_DrivingMobility"].rolling(7).mean().shift(7).reindex(dates)
    )
    out["Apple_DrivingMobility.rolling(7).mean().shift(14)"] = (
        features["Apple_DrivingMobility"].rolling(7).mean().shift(14).reindex(dates)
    )

    out["Apple_WalkingMobility.rolling(7).mean()"] = (
        features["Apple_WalkingMobility"].rolling(7).mean().reindex(dates)
    )
    out["Apple_WalkingMobility.rolling(7).mean().shift(7)"] = (
        features["Apple_WalkingMobility"].rolling(7).mean().shift(7).reindex(dates)
    )
    out["Apple_WalkingMobility.rolling(7).mean().shift(14)"] = (
        features["Apple_WalkingMobility"].rolling(7).mean().shift(14).reindex(dates)
    )

    out["Apple_TransitMobility.rolling(7).mean()"] = (
        features["Apple_TransitMobility"].rolling(7).mean().reindex(dates)
    )
    out["Apple_TransitMobility.rolling(7).mean().shift(7)"] = (
        features["Apple_TransitMobility"].rolling(7).mean().shift(7).reindex(dates)
    )
    out["Apple_TransitMobility.rolling(7).mean().shift(14)"] = (
        features["Apple_TransitMobility"].rolling(7).mean().shift(14).reindex(dates)
    )

    out["Google_GroceryMobility.rolling(7).mean().shift(7)"] = (
        features["Google_GroceryMobility"].rolling(7).mean().shift(7).reindex(dates)
    )
    out["Google_GroceryMobility.rolling(7).mean().shift(14)"] = (
        features["Google_GroceryMobility"].rolling(7).mean().shift(14).reindex(dates)
    )

    out["Google_ParksMobility.rolling(7).mean().shift(7)"] = (
        features["Google_ParksMobility"].rolling(7).mean().shift(7).reindex(dates)
    )
    out["Google_ParksMobility.rolling(7).mean().shift(14)"] = (
        features["Google_ParksMobility"].rolling(7).mean().shift(14).reindex(dates)
    )

    out["Google_TransitStationsMobility.rolling(7).mean().shift(7)"] = (
        features["Google_TransitStationsMobility"]
        .rolling(7)
        .mean()
        .shift(7)
        .reindex(dates)
    )
    out["Google_TransitStationsMobility.rolling(7).mean().shift(14)"] = (
        features["Google_TransitStationsMobility"]
        .rolling(7)
        .mean()
        .shift(14)
        .reindex(dates)
    )

    out["Google_RetailMobility.rolling(7).mean().shift(7)"] = (
        features["Google_RetailMobility"].rolling(7).mean().shift(7).reindex(dates)
    )
    out["Google_RetailMobility.rolling(7).mean().shift(14)"] = (
        features["Google_RetailMobility"].rolling(7).mean().shift(14).reindex(dates)
    )

    out["Google_ResidentialMobility.rolling(7).mean().shift(7)"] = (
        features["Google_ResidentialMobility"].rolling(7).mean().shift(7).reindex(dates)
    )
    out["Google_ResidentialMobility.rolling(7).mean().shift(14)"] = (
        features["Google_ResidentialMobility"]
        .rolling(7)
        .mean()
        .shift(14)
        .reindex(dates)
    )

    out["Google_WorkplacesMobility.rolling(7).mean().shift(7)"] = (
        features["Google_WorkplacesMobility"].rolling(7).mean().shift(7).reindex(dates)
    )
    out["Google_WorkplacesMobility.rolling(7).mean().shift(14)"] = (
        features["Google_WorkplacesMobility"].rolling(7).mean().shift(14).reindex(dates)
    )
    return out


def clean_features(env, input_features):
    locations_df = env.locations_df
    universe_states = env.locations["state"]
    universe_counties = env.locations["county"]
    output_features = {}

    impute_group_1 = [
        "JHU_ConfirmedCases.diff(7)",
        "JHU_ConfirmedCases.diff(7).shift(7)",
        "JHU_ConfirmedDeaths.diff(7)",
        "JHU_ConfirmedDeaths.diff(7).shift(7)",
    ]
    for name in impute_group_1:
        logger.info(f"Processing {name}.")
        dm = input_features[name].copy()
        dm = normalize_cases(dm, locations_df)
        dm = select_universe(dm, universe_counties, fill_missing=True)
        dm = dm.clip(0)
        dm = cross_section_winsor(dm)
        dm = dm.fillna(0)
        output_features[name] = dm

    impute_group_2 = [
        "Apple_DrivingMobility.rolling(7).mean()",
        "Apple_DrivingMobility.rolling(7).mean().shift(7)",
        "Apple_DrivingMobility.rolling(7).mean().shift(14)",
        "Apple_WalkingMobility.rolling(7).mean()",
        "Apple_WalkingMobility.rolling(7).mean().shift(7)",
        "Apple_WalkingMobility.rolling(7).mean().shift(14)",
        "Apple_TransitMobility.rolling(7).mean()",
        "Apple_TransitMobility.rolling(7).mean().shift(7)",
        "Apple_TransitMobility.rolling(7).mean().shift(14)",
        "Google_GroceryMobility.rolling(7).mean().shift(7)",
        "Google_GroceryMobility.rolling(7).mean().shift(14)",
        "Google_ParksMobility.rolling(7).mean().shift(7)",
        "Google_ParksMobility.rolling(7).mean().shift(14)",
        "Google_TransitStationsMobility.rolling(7).mean().shift(7)",
        "Google_TransitStationsMobility.rolling(7).mean().shift(14)",
        "Google_RetailMobility.rolling(7).mean().shift(7)",
        "Google_RetailMobility.rolling(7).mean().shift(14)",
        "Google_ResidentialMobility.rolling(7).mean().shift(7)",
        "Google_ResidentialMobility.rolling(7).mean().shift(14)",
        "Google_WorkplacesMobility.rolling(7).mean().shift(7)",
        "Google_WorkplacesMobility.rolling(7).mean().shift(14)",
    ]

    for name in impute_group_2:
        logger.info(f"Processing {name}.")
        dm_state = select_universe(input_features[name], universe_states)
        dm = select_universe(input_features[name], universe_counties, fill_missing=True)
        dm = dm.fillna(method="ffill")
        dm = dm.fillna(cross_section_cbsa_mean(dm, locations_df))
        dm = dm.fillna(get_state_value(dm, dm_state))
        dm = dm.fillna(cross_section_state_mean(dm))
        dm = dm.fillna(cross_section_mean(dm))
        dm = cross_section_winsor(dm, 5)
        output_features[name] = dm
    return output_features
