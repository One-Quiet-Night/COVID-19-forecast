# OneQuietNight Covid-19 Forecast

| Forecast           | National, state, and county numbers of new COVID-19 cases per week for next 4 weeks. |
:------------------- |:---------------------------------------------------- |
| **Authors**        | Areum Jo (areumjo1@gmail.com), Jae Cho (jaehun.cho@gmail.com) |
| **Last Updated**   | 2020-12-05                                           |
| **Paper**          | [OneQuietNight Covid-19 Forecast](docs/OQN.pdf) |

OneQuietNight Covid-19 Forecast uses scientifically-driven machine learning models to accurately predict the spread of Covid-19 infections using real-time data from the [C3 AI Covid-19 Data Lake](https://c3.ai/customers/covid-19-data-lake/). OneQuietNight forecasts the number of new Covid-19 cases per week for the next 4 weeks at the national, state, and county levels.

We publish the forecast through a [web application](https://one-quiet-night.github.io/vis/) and submit them to the [CDC](https://www.cdc.gov/coronavirus/2019-ncov/covid-data/mathematical-modeling.html) to help inform public health decision making.


## Runbook

Install:
```
git clone https://github.com/One-Quiet-Night/COVID-19-forecast
cd COVID-19-forecast
python setup.py install
```

Retrain the model and generate real-time predictions:
```
python main.py
```

## Basic usage

`OneQuietNightEnvironment` is the main entry point for the program.

```python
from onequietnight.env import OneQuietNightEnvironment
env = OneQuietNightEnvironment()
```

`OneQuietNightEnvironment()` takes two optional parameters:
- `base_path`: pathlib.Path object where data can be cached for future analysis. If this is not provided, all data will be fetched and stored in memory only. If this is provided, some data will be persisted to disk under the base_path.
- `today`: optional isoformat date string such as "2020-11-18" to override today value. This is used to cut off the "end" date for the c3ai data and to infer the epidemiological weeks for the forecast date and the target end date of the forecast.


`OneQuietNightEnvironment` contains the following functions:
- `get_or_create_locations_df`: Create the `location` data. The `location` data
joins three pieces of data together:
    - The tabular location data about the country, states, and counties from the C3 AI Covid-19 Data Lake `OutbreakLocation` `fetch` API.
    - The [Covid-19 Forecast Hub](https://github.com/reichlab/covid19-forecast-hub) location data (for publication).
    - The [Census Metropolitan and Micropolitan Statstical Area Reference File](https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html) to resolve county CBSA membership.
- `get_data`: Download source data.
- The following data are downloaded from the C3 AI Covid-19 Data Lake `OutbreakLocation` `evalmetrics` API.
    - Apple_DrivingMobility
    - Apple_TransitMobility
    - Apple_WalkingMobility
    - CovidTrackingProject_ConfirmedCases
    - CovidTrackingProject_ConfirmedDeaths
    - CovidTrackingProject_ConfirmedHospitalizations
    - CovidTrackingProject_NegativeTests
    - CovidTrackingProject_PendingTests
    - Google_GroceryMobility
    - Google_ParksMobility
    - Google_ResidentialMobility
    - Google_RetailMobility
    - Google_TransitStationsMobility
    - Google_WorkplacesMobility
    - JHU_ConfirmedCases
    - JHU_ConfirmedDeaths
    - These data are sourced from [Apple](https://covid19.apple.com/mobility), [Covid Tracking Project](https://covidtracking.com/), [Google](https://www.google.com/covid19/mobility/), and [JHU](https://github.com/CSSEGISandData/COVID-19).
    - We also implement fall backs that download from these sources and process them to the C3 AI Covid-19 Data Lake schema and format. The fall backs can be used when there are latency issues or missing releases with one of the sources. These controls are exposed as class variables in `OneQuietNightEnvironment`.
- The following data are sourced from [covidcast](https://cmu-delphi.github.io/delphi-epidata/api/covidcast.html).
    - Safegraph_FullTimeWorkProp
    - Safegraph_CompletelyHomeProp
    - Ght_RawSearch
- `get_features`: Transform source data to input features for modeling. It currently produces three sets of features for the three models that we have at each geographic hierarchical level.
- `train_models`: Trains the machine learning algorithms using the features. We implement a model pipeline to expose the data to the models and to handle the fit and predict processes. The pipeline class can be extended to implement additional models for use with the C3 AI Covid-19 Data Lake data sets.
- `save_visualization_data`: Make predictions using the latest features. Generate csv files for OneQuietNight web application.
- `save_covidhub_data`: Make predictions using the latest features. Generate csv files for Covid-19 Forecast Hub submissions.

When `base_path` is specified, the program will cache each of these data to the following structure under the base path on the first run `today`. Next time the functions above are called with `today` value (e.g. calling the program twice on `2020-11-18`) would load the data from local storage rather than the remote APIs.

```
└── data
    ├── [data_name].feather         <- Historical time-series data from c3 ai saved in feather format.
    ├── locations.feather           <- Tabular location dimension table.
    ├── feature_store.joblib        <- Transformed time-series data.
    ├── model_store.joblib          <- Model parameters.
    ├── [today]-OneQuietNight.csv   <- Forecast output for Covid-19 Forecast Hub submission.
    ├── JHU_[target].csv            <- JHU target values for OneQuietNight web application.
    └── OQN_[forecast].csv          <- Forecast output for OneQuietNight web application.
```
