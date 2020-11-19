import logging

import numpy as np
import pandas as pd
from onequietnight.data.utils import to_dataframe, to_matrix
from onequietnight.features import model_names
from onequietnight.features.transforms import (normalize_cases,
                                               undo_normalize_cases,
                                               undo_normalize_cases_df)

logger = logging.getLogger(__name__)


class ForecastPipeline:

    predict_cols = [
        "forecast_date",
        "target",
        "target_end_date",
        "id",
        "type",
        "quantile",
        "value",
    ]

    def __init__(
        self, env, universe=None, n_week_ahead=1, train_window=20, feature_columns=[]
    ):
        self.env = env
        self.universe = universe
        self.n_week_ahead = n_week_ahead
        self.train_window = train_window
        self.feature_columns = feature_columns

        self.dates = pd.date_range(
            env.start_date, env.today, freq="W-SAT", name="dates"
        )
        self.df = self.get_data()

    def get_features(self):
        """Return the set of features in a dataframe indexed by id and dates."""
        assert (
            self.universe in model_names
        ), f"Universe must be one of {str(model_names)}."
        logger.info("Loading features.")
        return pd.concat(
            [
                to_dataframe(dm, name)
                for name, dm in self.env.features[self.universe].items()
            ],
            axis=1,
            join="outer",
        )

    def get_target(self):
        """Compute new cases per 100k people per week shifted by t_shift."""
        logger.info("Creating target.")
        env = self.env
        t_shift = self.n_week_ahead
        all_dates = pd.date_range(env.start_date, env.today, name="dates")
        dates = pd.date_range(env.start_date, env.today, freq="W-SAT", name="dates")
        df = env.data["JHU_ConfirmedCases"].copy()
        dm = to_matrix(df)
        dm.index = pd.to_datetime(dm.index)
        dm = dm.reindex(all_dates)
        dm = dm.reindex(dates)
        dm = dm.diff(1).clip(0)
        dm = normalize_cases(dm, env.locations_df)
        target = dm.shift(-t_shift)
        target = to_dataframe(target, "target")
        return target

    def get_data(self):
        features_df = self.get_features()
        target = self.get_target()
        return features_df.join(target, how="inner")

    def get_model(self):
        from onequietnight.models.models.bayesian import ClippedModel
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([("scaler", StandardScaler()), ("model", ClippedModel())])

    def fit(self, instance_offset=0):
        train_window = self.train_window
        dates = self.dates
        feature_columns = self.feature_columns
        offset = self.n_week_ahead
        train_dates = dates[
            instance_offset - train_window - offset : instance_offset - offset
        ]
        logger.info(f"Training on window [{train_dates[0]}, {train_dates[-1]}]")
        train = self.df.loc[train_dates]
        y_train = train["target"].values
        X_train = train[feature_columns].values
        self.model = self.get_model()
        self.model.fit(X_train, y_train)

    def predict(self, offset=0, instance_offset=0, should_undo_normalize_cases=False):
        assert hasattr(self, "model")
        instance_date = self.dates[instance_offset - 1 - offset]
        instance = self.df.loc[[instance_date]]
        instance_X = instance[self.feature_columns].values

        logger.info(f"Predicting on {instance_date}.")
        logger.debug(f"Instance {instance.head().to_markdown()}.")
        predictions = self.model.predict(instance_X)
        predictions_df = pd.Series(predictions, index=instance.index)
        predictions_dm = to_matrix(predictions_df)

        if should_undo_normalize_cases:
            predictions_dm = undo_normalize_cases(predictions_dm, self.env.locations_df)
        predictions_dm["target"] = f"{self.n_week_ahead} wk ahead inc case"
        predictions_dm["forecast_date"] = self.env.today
        predictions_dm["target_end_date"] = instance_date + pd.tseries.offsets.Week(
            self.n_week_ahead, weekday=5
        )
        predictions_dm = predictions_dm.set_index(
            ["forecast_date", "target", "target_end_date"]
        )
        predictions_df = predictions_dm.stack()
        predictions_df = predictions_df.reset_index(name="value")
        predictions_df["type"] = "point"
        predictions_df["quantile"] = np.nan
        return predictions_df[self.predict_cols]

    def predict_proba(
        self, offset=0, instance_offset=0, should_undo_normalize_cases=False
    ):
        assert hasattr(self, "model")
        instance_date = self.dates[instance_offset - 1 - offset]
        instance = self.df.loc[[instance_date]]
        instance_X = instance[self.feature_columns].values

        logger.info(f"Predicting on {instance_date}.")
        logger.debug(f"Instance {instance.head().to_markdown()}.")
        predictions = self.model.predict_proba(instance_X)
        predictions_df = pd.DataFrame(predictions)
        predictions_df.index = instance.index

        predictions_df = predictions_df.stack()
        predictions_df.index = predictions_df.index.set_names("quantile", -1)
        predictions_df = predictions_df.reset_index(name="value")

        if should_undo_normalize_cases:
            predictions_df = undo_normalize_cases_df(
                predictions_df, self.env.locations_df
            )

        predictions_df["target"] = f"{self.n_week_ahead} wk ahead inc case"
        predictions_df["forecast_date"] = self.env.today
        predictions_df["type"] = "quantile"
        predictions_df["target_end_date"] = instance_date + pd.tseries.offsets.Week(
            self.n_week_ahead, weekday=5
        )
        return predictions_df[self.predict_cols]
