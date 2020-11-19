national = dict(
    universe="national",
    train_window=20,
    feature_columns=[
        "Apple_WalkingMobility.rolling(7).mean()",
        "Apple_WalkingMobility.rolling(7).mean().shift(7)",
        "Apple_DrivingMobility.rolling(7).mean()",
        "Apple_DrivingMobility.rolling(7).mean().shift(7)",
        "Apple_TransitMobility.rolling(7).mean()",
        "Apple_TransitMobility.rolling(7).mean().shift(7)",
        "JHU_ConfirmedCases.diff(7)",
        "JHU_ConfirmedCases.diff(7).shift(7)",
        "CovidTrackingProject_ConfirmedCases.diff(7)",
        "CovidTrackingProject_ConfirmedCases.diff(7).shift(7)",
        "CovidTrackingProject_ConfirmedHospitalizations.rolling(7).mean()",
        "CovidTrackingProject_ConfirmedHospitalizations.rolling(7).mean().shift(7)",
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
        "Google_WorkplacesMobility.rolling(7).mean().shift(14)",
        "Google_WorkplacesMobility.rolling(7).mean().shift(7)",
        "JHU_ConfirmedCases.diff(7)",
        "JHU_ConfirmedCases.diff(7).shift(7)",
        "CovidTrackingProject_ConfirmedCases.diff(7)",
        "CovidTrackingProject_ConfirmedCases.diff(7).shift(7)",
        "CovidTrackingProject_ConfirmedHospitalizations.rolling(7).mean()",
        "CovidTrackingProject_ConfirmedHospitalizations.rolling(7).mean().shift(7)",
    ],
)

county = dict(
    universe="county",
    train_window=20,
    feature_columns=[
        "Apple_WalkingMobility.rolling(7).mean()",
        "Apple_WalkingMobility.rolling(7).mean().shift(7)",
        "Apple_DrivingMobility.rolling(7).mean()",
        "Apple_DrivingMobility.rolling(7).mean().shift(7)",
        "Apple_TransitMobility.rolling(7).mean()",
        "Apple_TransitMobility.rolling(7).mean().shift(7)",
        "Google_WorkplacesMobility.rolling(7).mean().shift(14)",
        "Google_WorkplacesMobility.rolling(7).mean().shift(7)",
        "JHU_ConfirmedCases.diff(7)",
        "JHU_ConfirmedCases.diff(7).shift(7)",
    ],
)

model_configs = {"national": national, "state": state, "county": county}
max_weeks_ahead = 8
