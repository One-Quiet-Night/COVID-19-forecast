import logging

import pandas as pd
from onequietnight.data.utils import to_matrix
from onequietnight.features.transforms import (
    normalize_beds,
    normalize_cases,
    select_universe,
)

logger = logging.getLogger(__name__)

model_name = "national"


def transform_features(env, features, freq="W-SAT"):
    """Transform state-level predictors.

    Note: Do not refactor. Create features explicitly one by one instead of
    using a kitchen-sink approach.
    """
    logger.info("Processing national features.")
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

    out["CovidTrackingProject_ConfirmedCases.diff(7)"] = (
        features["CovidTrackingProject_ConfirmedCases"].diff(7).reindex(dates)
    )
    out["CovidTrackingProject_ConfirmedCases.diff(7).shift(7)"] = (
        features["CovidTrackingProject_ConfirmedCases"].diff(7).shift(7).reindex(dates)
    )

    out["CovidTrackingProject_ConfirmedDeaths.diff(7)"] = (
        features["CovidTrackingProject_ConfirmedDeaths"].diff(7).reindex(dates)
    )
    out["CovidTrackingProject_ConfirmedDeaths.diff(7).shift(7)"] = (
        features["CovidTrackingProject_ConfirmedDeaths"].diff(7).shift(7).reindex(dates)
    )

    out["CovidTrackingProject_NegativeTests.diff(7)"] = (
        features["CovidTrackingProject_NegativeTests"].diff(7).reindex(dates)
    )
    out["CovidTrackingProject_NegativeTests.diff(7).shift(7)"] = (
        features["CovidTrackingProject_NegativeTests"].diff(7).shift(7).reindex(dates)
    )

    out["CovidTrackingProject_PendingTests.rolling(7).mean()"] = (
        features["CovidTrackingProject_PendingTests"].rolling(7).mean().reindex(dates)
    )
    out["CovidTrackingProject_PendingTests.rolling(7).mean().shift(7)"] = (
        features["CovidTrackingProject_PendingTests"]
        .rolling(7)
        .mean()
        .shift(7)
        .reindex(dates)
    )

    out["CovidTrackingProject_ConfirmedHospitalizations.rolling(7).mean()"] = (
        features["CovidTrackingProject_ConfirmedHospitalizations"]
        .rolling(7)
        .mean()
        .reindex(dates)
    )
    out["CovidTrackingProject_ConfirmedHospitalizations.rolling(7).mean().shift(7)"] = (
        features["CovidTrackingProject_ConfirmedHospitalizations"]
        .rolling(7)
        .mean()
        .shift(7)
        .reindex(dates)
    )

    out["CovidTrackingProject_Ventilator.rolling(7).mean()"] = (
        features["CovidTrackingProject_Ventilator"].rolling(7).mean().reindex(dates)
    )
    out["CovidTrackingProject_Ventilator.rolling(7).mean().shift(7)"] = (
        features["CovidTrackingProject_Ventilator"]
        .rolling(7)
        .mean()
        .shift(7)
        .reindex(dates)
    )

    out["CovidTrackingProject_ICU.rolling(7).mean()"] = (
        features["CovidTrackingProject_ICU"].rolling(7).mean().reindex(dates)
    )
    out["CovidTrackingProject_ICU.rolling(7).mean().shift(7)"] = (
        features["CovidTrackingProject_ICU"].rolling(7).mean().shift(7).reindex(dates)
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

    out["Chng_SmoothedOutpatientCovid.rolling(7).mean().shift(7)"] = (
        features["Chng_SmoothedOutpatientCovid"]
        .rolling(7)
        .mean()
        .shift(7)
        .reindex(dates)
    )
    out["Chng_SmoothedOutpatientCovid.rolling(7).mean().shift(14)"] = (
        features["Chng_SmoothedOutpatientCovid"]
        .rolling(7)
        .mean()
        .shift(14)
        .reindex(dates)
    )

    out["DoctorVisits_SmoothedCli.rolling(7).mean().shift(7)"] = (
        features["DoctorVisits_SmoothedCli"].rolling(7).mean().shift(7).reindex(dates)
    )
    out["DoctorVisits_SmoothedCli.rolling(7).mean().shift(14)"] = (
        features["DoctorVisits_SmoothedCli"].rolling(7).mean().shift(14).reindex(dates)
    )

    out["FbSurvey_RawWili.rolling(7).mean()"] = (
        features["FbSurvey_RawWili"].rolling(7, min_periods=1).mean().reindex(dates)
    )
    out["FbSurvey_RawWili.rolling(7).mean().shift(7)"] = (
        features["FbSurvey_RawWili"]
        .rolling(7, min_periods=1)
        .mean()
        .shift(7)
        .reindex(dates)
    )
    out["FbSurvey_RawWcli.rolling(7).mean()"] = (
        features["FbSurvey_RawWcli"].rolling(7, min_periods=1).mean().reindex(dates)
    )
    out["FbSurvey_RawWcli.rolling(7).mean().shift(7)"] = (
        features["FbSurvey_RawWcli"]
        .rolling(7, min_periods=1)
        .mean()
        .shift(7)
        .reindex(dates)
    )
    out["FbSurvey_RawHhCmntyCli.rolling(7).mean()"] = (
        features["FbSurvey_RawHhCmntyCli"]
        .rolling(7, min_periods=1)
        .mean()
        .reindex(dates)
    )
    out["FbSurvey_RawHhCmntyCli.rolling(7).mean().shift(7)"] = (
        features["FbSurvey_RawHhCmntyCli"]
        .rolling(7, min_periods=1)
        .mean()
        .shift(7)
        .reindex(dates)
    )

    out["Ght_RawSearch.rolling(7).mean().shift(4)"] = (
        features["Ght_RawSearch"]
        .rolling(7, min_periods=1)
        .mean()
        .shift(4)
        .reindex(dates)
    )
    out["Ght_RawSearch.rolling(7).mean().shift(11)"] = (
        features["Ght_RawSearch"]
        .rolling(7, min_periods=1)
        .mean()
        .shift(11)
        .reindex(dates)
    )

    out["Safegraph_CompletelyHomeProp.rolling(7).mean().shift(4)"] = (
        features["Safegraph_CompletelyHomeProp"]
        .rolling(7, min_periods=1)
        .mean()
        .shift(4)
        .reindex(dates)
    )
    out["Safegraph_CompletelyHomeProp.rolling(7).mean().shift(11)"] = (
        features["Safegraph_CompletelyHomeProp"]
        .rolling(7, min_periods=1)
        .mean()
        .shift(11)
        .reindex(dates)
    )
    out["Safegraph_FullTimeWorkProp.rolling(7).mean().shift(4)"] = (
        features["Safegraph_FullTimeWorkProp"]
        .rolling(7, min_periods=1)
        .mean()
        .shift(4)
        .reindex(dates)
    )
    out["Safegraph_FullTimeWorkProp.rolling(7).mean().shift(11)"] = (
        features["Safegraph_FullTimeWorkProp"]
        .rolling(7, min_periods=1)
        .mean()
        .shift(11)
        .reindex(dates)
    )
    out["Safegraph_PartTimeWorkProp.rolling(7).mean().shift(4)"] = (
        features["Safegraph_PartTimeWorkProp"]
        .rolling(7, min_periods=1)
        .mean()
        .shift(4)
        .reindex(dates)
    )
    out["Safegraph_PartTimeWorkProp.rolling(7).mean().shift(11)"] = (
        features["Safegraph_PartTimeWorkProp"]
        .rolling(7, min_periods=1)
        .mean()
        .shift(11)
        .reindex(dates)
    )
    out["Safegraph_MedianHomeDwellTime.rolling(7).mean().shift(4)"] = (
        features["Safegraph_MedianHomeDwellTime"]
        .rolling(7, min_periods=1)
        .mean()
        .shift(4)
        .reindex(dates)
    )
    out["Safegraph_MedianHomeDwellTime.rolling(7).mean().shift(11)"] = (
        features["Safegraph_MedianHomeDwellTime"]
        .rolling(7, min_periods=1)
        .mean()
        .shift(11)
        .reindex(dates)
    )
    return out


def clean_features(env, input_features):
    locations_df = env.locations_df
    universe_states = env.locations["state"]
    universe_national = env.locations["country"]
    output_features = {}

    impute_group_0 = [
        "CovidTrackingProject_ConfirmedHospitalizations.rolling(7).mean()",
        "CovidTrackingProject_ConfirmedHospitalizations.rolling(7).mean().shift(7)",
        "CovidTrackingProject_ICU.rolling(7).mean()",
        "CovidTrackingProject_ICU.rolling(7).mean().shift(7)",
        "CovidTrackingProject_Ventilator.rolling(7).mean()",
        "CovidTrackingProject_Ventilator.rolling(7).mean().shift(7)",
    ]
    for name in impute_group_0:
        logger.info(f"Processing {name}.")
        dm_state = select_universe(input_features[name], universe_states)
        dm = input_features[name].copy()
        dm = select_universe(dm, universe_national, fill_missing=True)
        dm_state_sum = to_matrix(
            dm_state.sum(1)
            .reset_index(name="value")
            .assign(id=universe_national[0])
            .set_index(["dates", "id"])
        )
        dm = dm.fillna(dm_state_sum)
        dm = normalize_beds(dm, locations_df)
        dm = dm.clip(0)
        dm = dm.fillna(0)
        output_features[name] = dm

    impute_group_1 = [
        "JHU_ConfirmedCases.diff(7)",
        "JHU_ConfirmedCases.diff(7).shift(7)",
        "JHU_ConfirmedDeaths.diff(7)",
        "JHU_ConfirmedDeaths.diff(7).shift(7)",
        "CovidTrackingProject_ConfirmedCases.diff(7)",
        "CovidTrackingProject_ConfirmedCases.diff(7).shift(7)",
        "CovidTrackingProject_ConfirmedDeaths.diff(7)",
        "CovidTrackingProject_ConfirmedDeaths.diff(7).shift(7)",
        "CovidTrackingProject_NegativeTests.diff(7)",
        "CovidTrackingProject_NegativeTests.diff(7).shift(7)",
        "CovidTrackingProject_PendingTests.rolling(7).mean()",
        "CovidTrackingProject_PendingTests.rolling(7).mean().shift(7)",
    ]
    for name in impute_group_1:
        logger.info(f"Processing {name}.")
        dm_state = select_universe(input_features[name], universe_states)
        dm = input_features[name].copy()
        dm = select_universe(dm, universe_national, fill_missing=True)
        dm_state_sum = to_matrix(
            dm_state.sum(1)
            .reset_index(name="value")
            .assign(id=universe_national[0])
            .set_index(["dates", "id"])
        )
        dm = dm.fillna(dm_state_sum)
        dm = normalize_cases(dm, locations_df)
        dm = dm.clip(0)
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
        "Chng_SmoothedOutpatientCovid.rolling(7).mean().shift(7)",
        "Chng_SmoothedOutpatientCovid.rolling(7).mean().shift(14)",
        "DoctorVisits_SmoothedCli.rolling(7).mean().shift(7)",
        "DoctorVisits_SmoothedCli.rolling(7).mean().shift(14)",
        "FbSurvey_RawWili.rolling(7).mean()",
        "FbSurvey_RawWili.rolling(7).mean().shift(7)",
        "FbSurvey_RawWcli.rolling(7).mean()",
        "FbSurvey_RawWcli.rolling(7).mean().shift(7)",
        "FbSurvey_RawHhCmntyCli.rolling(7).mean()",
        "FbSurvey_RawHhCmntyCli.rolling(7).mean().shift(7)",
        "Ght_RawSearch.rolling(7).mean().shift(4)",
        "Ght_RawSearch.rolling(7).mean().shift(11)",
        "Safegraph_CompletelyHomeProp.rolling(7).mean().shift(4)",
        "Safegraph_CompletelyHomeProp.rolling(7).mean().shift(11)",
        "Safegraph_FullTimeWorkProp.rolling(7).mean().shift(4)",
        "Safegraph_FullTimeWorkProp.rolling(7).mean().shift(11)",
        "Safegraph_PartTimeWorkProp.rolling(7).mean().shift(4)",
        "Safegraph_PartTimeWorkProp.rolling(7).mean().shift(11)",
        "Safegraph_MedianHomeDwellTime.rolling(7).mean().shift(4)",
        "Safegraph_MedianHomeDwellTime.rolling(7).mean().shift(11)",
    ]

    for name in impute_group_2:
        logger.info(f"Processing {name}.")
        dm_state = select_universe(input_features[name], universe_states)
        dm = select_universe(input_features[name], universe_national, fill_missing=True)
        dm_state_mean = to_matrix(
            dm_state.mean(1)
            .reset_index(name="value")
            .assign(id=universe_national[0])
            .set_index(["dates", "id"])
        )
        dm = dm.fillna(dm_state_mean)
        dm = dm.fillna(method="ffill")
        output_features[name] = dm

    return output_features
