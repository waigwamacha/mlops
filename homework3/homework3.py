## INGEST DATA

import requests
from io import BytesIO
from typing import List

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    #dfs: List[pd.DataFrame] = []

    #for year, months in [(2023, (1, 3))]:
    #    for i in range(*months):
    response = requests.get(
                'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
            )

    if response.status_code != 200:
        raise Exception(response.text)

    df = pd.read_parquet(BytesIO(response.content))
            #dfs.append(df)

    return df



## PREPARE DATA

import pandas as pd

from your_first_project.utils.data_prep.cleaning import clean
from your_first_project.utils.data_prep.feature_engineering import combine_features
from your_first_project.utils.data_prep.feature_selector import select_features
#from mlops.utils.data_prep.splitters import split_on_value

if 'transformer' not in globals():
    from mage_ai.data_prep.decorators import transformer


@transformer
def read_dataframe(df: pd.DataFrame):

    #df = df.drop(['VendorID'], axis=1)

    #df = clean(df)
    #df = combine_features(df)
    #df = select_features(df)

    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


## TRAIN MODEL


import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def train(df: pd.DataFrame):
    
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    train_dicts = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    target = 'duration'
    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)

    RMSE = mean_squared_error(y_train, y_pred, squared=False)

    print(f"{lr.intercept_}, {RMSE}")

    return dv, lr 


## REGISTER MODEL


from sklearn.linear_model import LinearRegression

import mlflow
import mlflow.sklearn


EXPERIMENT_NAME = "Linear regression"
mlflow.set_experiment(EXPERIMENT_NAME)
vectorizer_path = ''

print(f"MLFlow Version: {mlflow.__version__}")

@data_exporter
def export_data(dv, lr):

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    mlflow.search_experiments()
    dv, lr = artifacts

    with mlflow.start_run(run_name=EXPERIMENT_NAME):
        # Log the model
        print('entered mlflow')
        mlflow.sklearn.log_model(lr, "model")
        mlflow.log_metric('rmse',8.16)
        # Save the DictVectorizer to a file and log it as an artifact
        vectorizer_path = "dict_vectorizer.pkl"
        joblib.dump(dv, vectorizer_path)
        mlflow.log_artifact(vectorizer_path, "dict_vectorizer")
        print('logged siccesfully')



