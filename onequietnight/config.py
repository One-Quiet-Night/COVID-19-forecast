national = dict(
    universe="national",
    train_window=16,
    feature_columns=[
        "Apple_WalkingMobility.rolling(7).mean()",
        "Apple_WalkingMobility.rolling(7).mean().shift(7)",
        "Apple_DrivingMobility.rolling(7).mean()",
        "Apple_DrivingMobility.rolling(7).mean().shift(7)",
        "Apple_TransitMobility.rolling(7).mean()",
        "Apple_TransitMobility.rolling(7).mean().shift(7)",
        "JHU_ConfirmedCases",
        "JHU_ConfirmedCases.diff(7)",
        "JHU_ConfirmedCases.diff(7).shift(7)",
    ],
)

state = dict(
    universe="state",
    train_window=20,
    feature_columns=[
        "Apple_WalkingMobility.rolling(7).mean()",
        "Apple_WalkingMobility.rolling(7).mean().shift(7)",
        "Apple_DrivingMobility.rolling(7).mean()",
        "Apple_DrivingMobility.rolling(7).mean().shift(7)",
        "Apple_TransitMobility.rolling(7).mean()",
        "Apple_TransitMobility.rolling(7).mean().shift(7)",
        "JHU_ConfirmedCases",
        "JHU_ConfirmedCases.diff(7)",
        "JHU_ConfirmedCases.diff(7).shift(7)",
        "Google_WorkplacesMobility.rolling(7).mean().shift(14)",
        "Google_WorkplacesMobility.rolling(7).mean().shift(7)",
        "CovidTrackingProject_ConfirmedHospitalizations.rolling(7).mean()",
        "CovidTrackingProject_ConfirmedHospitalizations.rolling(7).mean().shift(7)",
        "Safegraph_FullTimeWorkProp.rolling(7).mean().shift(4)",
        "Safegraph_FullTimeWorkProp.rolling(7).mean().shift(11)",
        "Safegraph_CompletelyHomeProp.rolling(7).mean().shift(4)",
        "Safegraph_CompletelyHomeProp.rolling(7).mean().shift(11)",
    ],
)

county = dict(
    universe="county",
    train_window=24,
    feature_columns=[
        "Apple_WalkingMobility.rolling(7).mean()",
        "Apple_WalkingMobility.rolling(7).mean().shift(7)",
        "Apple_DrivingMobility.rolling(7).mean()",
        "Apple_DrivingMobility.rolling(7).mean().shift(7)",
        "Apple_TransitMobility.rolling(7).mean()",
        "Apple_TransitMobility.rolling(7).mean().shift(7)",
        "JHU_ConfirmedCases",
        "JHU_ConfirmedCases.diff(7)",
        "JHU_ConfirmedCases.diff(7).shift(7)",
        "Google_WorkplacesMobility.rolling(7).mean().shift(7)",
        "Google_WorkplacesMobility.rolling(7).mean().shift(14)",
        "CovidTrackingProject_ConfirmedHospitalizations.rolling(7).mean().state",
        "CovidTrackingProject_ConfirmedHospitalizations.rolling(7).mean().shift(7).state",
        "Safegraph_FullTimeWorkProp.rolling(7).mean().shift(4)",
        "Safegraph_FullTimeWorkProp.rolling(7).mean().shift(11)",
        "Ght_RawSearch.rolling(7).mean().shift(4)",
        "Ght_RawSearch.rolling(7).mean().shift(11)",
        "JHU_ConfirmedCases.diff(7).cbsa",
        "JHU_ConfirmedCases.diff(7).shift(7).cbsa",
    ],
)

model_configs = {"national": national, "state": state, "county": county}
max_weeks_ahead = 4
