# src/drift_report.py
# Compares the most recent window_days of data against all older data:
# data drift (feature distributions) + model drift (prediction/target
# relationship — is the model's error pattern changing over time).
# Run: python src/drift_report.py
# Output: reports/latest_drift_report.html (overwritten daily) + dated copy

from pathlib import Path
from datetime import date
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()
    feat_cols   = config["features"]["feature_cols"]
    window_days = config["drift"]["window_days"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    model_name = config["model"]["name"]

    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
    except Exception as e:
        print(f"No Production model yet ({e}) — skipping drift report until first promotion.")
        exit(0)

    df = pd.read_csv(config["data"]["processed_path"], parse_dates=["time"])
    reference = df.iloc[:-window_days].reset_index(drop=True)
    current   = df.iloc[-window_days:].reset_index(drop=True)

    reference["prediction"] = model.predict(reference[feat_cols])
    current["prediction"]   = model.predict(current[feat_cols])

    column_mapping = ColumnMapping(
        target="target",
        prediction="prediction",
        datetime="time",
        numerical_features=feat_cols,
    )

    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    Path("reports").mkdir(exist_ok=True)
    report.save_html(f"reports/drift_report_{date.today().isoformat()}.html")
    report.save_html("reports/latest_drift_report.html")

    drift_detected = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
    print(f"Drift report saved → reports/latest_drift_report.html")
    print(f"Dataset drift detected: {drift_detected}")