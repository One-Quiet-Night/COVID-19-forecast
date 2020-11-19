import logging
from pathlib import Path

import joblib
import pandas as pd

from onequietnight.config import max_weeks_ahead, model_configs
from onequietnight.data import apple, covidtracking, google, jhu
from onequietnight.data.io import get_date_partition, read_data, write_data
from onequietnight.data.locations import convert_c3ai_to_jhu, get_locations
from onequietnight.data.utils import to_dataframe, to_matrix
from onequietnight.features import (county, national, state,
                                    transform_data_to_features,
                                    transform_dates)
from onequietnight.features.transforms import normalize_cases, select_universe
from onequietnight.models.forecast import ForecastPipeline

logger = logging.getLogger(__name__)


def locations_map(locations_df):
    """Return a locations map:
    {
        'country': ['UnitedStates'],
        'county': [
            'Abbeville_SouthCarolina_UnitedStates',
            'Acadia_Louisiana_UnitedStates',
            ...
        ],
        'state': [
            'Alabama_UnitedStates',
            'Alaska_UnitedStates',
            ...
        ],
    }
    """
    return locations_df.groupby("locationType").id.apply(list).to_dict()


class OneQuietNightEnvironment:
    """Environment for data engineering, feature engineering, and modeling.
    ---
    base_path: optional pathlib.Path object where data can be cached for future
        analysis. If this is not provided, all data will be fetched and
        stored in memory only. If this is provided, some data will be persisted
        to disk under the base_path.
    today: optional isoformat date string such as "2020-11-18" to override
        today value. This is used to cut off the "end" date for the c3ai data
        and to infer the epidemiological weeks for the forecast date and the
        target end date of the forecast.
    """

    locations_filename = "locations.feather"
    features_filename = "feature_store.joblib"
    models_filename = "model_store.joblib"
    start_date = "2020-01-20"
    load_data_jhu = True
    load_data_covidtracking = True
    load_data_google = True
    load_data_apple = True

    def __init__(self, base_path=None, today=None):
        self.base_path = base_path
        self.today = today or pd.Timestamp.today().date().isoformat()
        self.write = isinstance(self.base_path, Path)
        if self.write:
            Path.mkdir(self.base_path, exist_ok=True)
        self.locations_df = self.get_or_create_locations_df()
        self.locations = locations_map(self.locations_df)
        self.data = {}

    def get_or_create_locations_df(self):
        if self.write:
            locations_df_path = Path(get_date_partition(self), self.locations_filename)
            if Path.exists(locations_df_path):
                return pd.read_feather(locations_df_path)
            else:
                locations_df = get_locations()
                logger.info(f"Writing to {locations_df_path}")
                locations_df.to_feather(locations_df_path)
                return locations_df
        else:
            return get_locations()

    def get_data(self):
        if self.write:
            for source in [jhu, apple, google, covidtracking]:
                try:
                    for name in source.metrics:
                        self.data[name] = read_data(self, name)
                except ValueError:
                    source_data = source.load_data(self)
                    write_data(self, source_data)
                    self.data = {**self.data, **source_data}
        else:
            for source in [jhu, apple, google, covidtracking]:
                source_data = source.load_data(self)
                self.data = {**self.data, **source_data}

    def get_features(self):
        features_df_path = Path(get_date_partition(self), self.features_filename)
        if self.write and features_df_path.exists():
            logger.info(f"Reading features from {str(features_df_path)}.")
            self.features = joblib.load(str(features_df_path))
        else:
            features = transform_data_to_features(self, self.data)
            features = transform_dates(self, features)

            self.features = {}
            for model in [national, state, county]:
                model_features = model.transform_features(self, features)
                model_features = model.clean_features(self, model_features)
                self.features[model.model_name] = model_features

            if self.write:
                logger.info(f"Writing features to {str(features_df_path)}.")
                joblib.dump(self.features, str(features_df_path))

    def train_models(self, instance_offset=0):
        models_df_path = Path(get_date_partition(self), self.models_filename)
        if not instance_offset and self.write and models_df_path.exists():
            logger.info(f"Reading models from {str(models_df_path)}.")
            self.models = joblib.load(str(models_df_path))
        else:
            self.models = {"national": {}, "state": {}, "county": {}}
            for name, config in model_configs.items():
                logger.info(f"Training {name} models.")
                for n_week_ahead in range(1, max_weeks_ahead + 1):
                    model_pipeline = ForecastPipeline(
                        self, **config, **dict(n_week_ahead=n_week_ahead)
                    )
                    model_pipeline.fit(instance_offset=instance_offset)
                    self.models[name][n_week_ahead] = model_pipeline

            if not instance_offset and self.write:
                logger.info(f"Writing models to {str(models_df_path)}.")
                joblib.dump(self.models, str(models_df_path))

    def predict(
        self,
        should_predict_proba=False,
        should_undo_normalize_cases=False,
        instance_offset=0,
    ):
        forecasts = {}
        for name in model_configs:
            forecast = []
            for n_week_ahead in range(1, max_weeks_ahead + 1):
                n_week_ahead_preds = self.models[name][n_week_ahead].predict(
                    should_undo_normalize_cases=should_undo_normalize_cases,
                    instance_offset=instance_offset,
                )
                forecast.append(n_week_ahead_preds)
                if should_predict_proba:
                    n_week_ahead_preds_proba = self.models[name][
                        n_week_ahead
                    ].predict_proba(
                        should_undo_normalize_cases=should_undo_normalize_cases,
                        instance_offset=instance_offset,
                    )
                    forecast.append(n_week_ahead_preds_proba)
            forecasts[name] = pd.concat(forecast)
        return forecasts

    def save_covidhub_data(self, instance_offset=0):
        forecasts = self.predict(
            should_predict_proba=False,
            should_undo_normalize_cases=True,
            instance_offset=instance_offset,
        )
        forecasts_df = pd.concat(forecasts.values())
        forecasts_df = pd.merge(
            forecasts_df, self.locations_df[["id", "location"]], on="id", how="left"
        ).drop(columns=["id"])
        cols = [
            "forecast_date",
            "target",
            "target_end_date",
            "location",
            "type",
            "quantile",
            "value",
        ]
        forecasts_df = forecasts_df[cols]
        forecasts_df = forecasts_df.sort_values(
            ["target", "target_end_date", "location", "type", "quantile"]
        )
        forecasts_df["value"] = forecasts_df["value"].round()
        forecasts_df["value"] = forecasts_df["value"].clip(0)

        if instance_offset == 0:
            filename = f"{self.today}-OneQuietNight.csv"
        else:
            filename = f"Backfill-{instance_offset}-{self.today}-OneQuietNight.csv"
        filepath = Path(get_date_partition(self), filename)
        logger.info(f"Writing to {filepath}")
        forecasts_df.to_csv(filepath, index=False)

    def save_visualization_data(self):
        forecasts = self.predict()
        for name, df in forecasts.items():
            if name == "national":
                name = "country"
            df = convert_c3ai_to_jhu(df, self.locations_df)
            df = df.rename(columns={"target_end_date": "dates"})
            df = df.set_index(["dates", "location"])["value"].unstack()
            filename = f"OQN_IncidentCasesForecast_{name.capitalize()}.csv"
            filepath = Path(get_date_partition(self), filename)
            logger.info(f"Writing to {filepath}")
            df.to_csv(filepath)

        dm = self.get_new_cases_per_100k()
        universes = (
            self.locations_df.groupby("locationType")["fips.id"].apply(list).to_dict()
        )
        for universe_name, universe in universes.items():
            filename = f"JHU_IncidentCases_{universe_name.capitalize()}.csv"
            filepath = Path(get_date_partition(self), filename)
            logger.info(f"Writing to {filepath}")
            select_universe(dm, universe).to_csv(filepath)

    def get_new_cases_per_100k(self):
        dates = pd.date_range(self.start_date, self.today, freq="W-SAT", name="dates")
        dm = to_matrix(self.data["JHU_ConfirmedCases"])

        dm = dm.reindex(dates, method="ffill").diff(1)
        dm = normalize_cases(dm, self.locations_df)
        df = (
            pd.merge(
                to_dataframe(dm).reset_index(),
                self.locations_df[["id", "location"]],
                on="id",
                how="left",
            )
            .drop(columns=["id"])
            .rename(columns={"location": "id"})
        )
        dm = to_matrix(df)
        return dm
