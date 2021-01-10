"""Retrain the model and generate real-time predictions.

This script downloads the data to memory, transforms the features, generates
a training data set using historical data of features and targets, retrains
a model, and makes predictions using the latest instance of the feature values.
"""
import json
from pathlib import Path

import pandas as pd

from onequietnight.env import OneQuietNightEnvironment


def main():
    """Retrain the model and generate real-time predictions."""
    env = OneQuietNightEnvironment(Path.cwd())
    env.get_data()
    env.get_features()
    env.train_models()
    env.save_visualization_data()
    env.save_covidhub_data()


def validate_viz():
    for universe_name in ["County", "State", "Country"]:
        filename_actuals = f"JHU_IncidentCases_{universe_name}.csv"
        filepath_actuals = Path(Path.cwd(), "vis", "src", "Data", universe_name, filename_actuals)
        df_actuals = pd.read_csv(filepath_actuals)

        filename_preds = f"OQN_IncidentCasesForecast_{universe_name}.csv"
        filepath_preds = Path(Path.cwd(), "vis", "src", "Data", universe_name, filename_preds)
        df_preds = pd.read_csv(filepath_preds)

        dataEndDate = df_actuals["dates"].values[-1]
        if universe_name == "County":
            dataKingCountyEndValue = int(df_actuals["53033"].values[-1])
        elif universe_name == "State":
            dataWashingtonStateEndValue = int(df_actuals["53"].values[-1])
        forecastStartDate = df_preds["dates"].values[0]
        forecastEndDate = df_preds["dates"].values[-1]

        df = pd.concat([df_actuals, df_preds])
        df.to_csv(filepath_actuals, index=False)
    config = dict(
        dataEndDate=dataEndDate,
        dataKingCountyEndValue=dataKingCountyEndValue,
        dataWashingtonStateEndValue=dataWashingtonStateEndValue,
        forecastStartDate=forecastStartDate,
        forecastEndDate=forecastEndDate,
    )
    print(config)
    with open(Path(Path.cwd(), "vis", "src", "config.json"), "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    main()
    validate_viz()
