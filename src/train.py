# src/train.py
# Trains the model using decisions from config.yaml.
# Logs everything to MLflow. Registers the model in the MLflow registry.
# Run: python src/train.py
# Output (printed to stdout for pipeline.sh to capture):
#   RUN_ID=abc123 CV_MAE=1.823

import os
import tempfile
import joblib
import pandas as pd
import mlflow
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def train():
    config = load_config()

    df = pd.read_csv(config["data"]["processed_path"])
    feat_cols  = config["features"]["feature_cols"]
    target_col = config["features"]["target_col"]
    X, y = df[feat_cols], df[target_col]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    params = config["model"]["params"]
    model_name = config["model"]["name"]

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_param("n_features",   len(feat_cols))
        mlflow.log_param("n_train_rows", len(df))
        mlflow.log_param("feature_list", ",".join(feat_cols))

        model = GradientBoostingRegressor(**params)

        tscv = TimeSeriesSplit(n_splits=5)
        maes = []
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds = model.predict(X.iloc[val_idx])
            mae   = mean_absolute_error(y.iloc[val_idx], preds)
            maes.append(mae)
            mlflow.log_metric("fold_mae", mae, step=fold)

        cv_mae = sum(maes) / len(maes)
        mlflow.log_metric("cv_mae", cv_mae)

        model.fit(X, y)
        train_r2 = r2_score(y, model.predict(X))
        mlflow.log_metric("train_r2", train_r2)

        # Save the model as a PLAIN artifact file (joblib), NOT via
        # mlflow.sklearn.log_model()'s registered_model_name flow. MLflow 3.x
        # routes log_model() through a "Logged Model" entity whose artifacts
        # are proxied via a separate store — a known bug (mlflow/mlflow#16429)
        # means these become undownloadable through a local file-based
        # mlflow-artifacts server, regardless of whether you later load by
        # alias, version, or run_id. A plain joblib file avoids that path
        # entirely; we still get full registry versioning/aliasing by
        # registering the version explicitly below.
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.joblib")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="pickled_model")

        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            pass  # already exists — fine

        client.create_model_version(
            name=model_name,
            source=f"runs:/{run.info.run_id}/pickled_model",
            run_id=run.info.run_id,
        )

        print(f"RUN_ID={run.info.run_id} CV_MAE={cv_mae:.6f}")
        return run.info.run_id, cv_mae


if __name__ == "__main__":
    train()