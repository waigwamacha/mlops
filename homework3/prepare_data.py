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
