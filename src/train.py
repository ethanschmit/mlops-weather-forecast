# src/train.py
# Trains the model using decisions from config.yaml.
# Logs everything to MLflow. Registers the model in the MLflow registry.
# Run: python src/train.py
# Output (printed to stdout for pipeline.sh to capture):
#   RUN_ID=abc123 CV_MAE=1.823

import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def train():
    config = load_config()

    # Load processed features
    df = pd.read_csv(config["data"]["processed_path"])
    feat_cols  = config["features"]["feature_cols"]
    target_col = config["features"]["target_col"]
    X, y = df[feat_cols], df[target_col]

    # Connect to MLflow — must be running (mlflow server --host 127.0.0.1 --port 5000)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    params = config["model"]["params"]

    with mlflow.start_run() as run:
        # Log everything that defines this training run
        mlflow.log_params(params)
        mlflow.log_param("n_features",   len(feat_cols))
        mlflow.log_param("n_train_rows", len(df))
        mlflow.log_param("feature_list", ",".join(feat_cols))

        model = GradientBoostingRegressor(**params)

        # Time-series cross validation — same setup as notebook
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

        # Final fit on ALL data before registering
        model.fit(X, y)
        train_r2 = r2_score(y, model.predict(X))
        mlflow.log_metric("train_r2", train_r2)

        # Register in MLflow Model Registry
        # This is what the serving layer reads from
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=config["model"]["name"],
        )

        # Print in a parseable format for pipeline.sh
        print(f"RUN_ID={run.info.run_id} CV_MAE={cv_mae:.6f}")
        return run.info.run_id, cv_mae


if __name__ == "__main__":
    train()