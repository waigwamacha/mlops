
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



