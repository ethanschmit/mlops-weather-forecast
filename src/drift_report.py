# src/drift_report.py
# Compares this calendar month's data against the SAME calendar month last
# year. Scores against the CHAMPION model, loaded as a plain joblib artifact
# (see train.py note — avoids MLflow 3.x's Logged Model download bug).
# Run: python src/drift_report.py
# Output: reports/latest_drift_report.html (overwritten daily) + a dated copy

import os
from pathlib import Path
from datetime import date
import joblib
import pandas as pd
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()
    feat_cols  = config["features"]["feature_cols"]
    model_name = config["model"]["name"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    try:
        champion_version = client.get_model_version_by_alias(model_name, "champion")
        local_dir = download_artifacts(run_id=champion_version.run_id, artifact_path="pickled_model")
        model = joblib.load(os.path.join(local_dir, "model.joblib"))
    except Exception as e:
        print(f"No champion model yet ({e}) — skipping drift report.")
        Path("reports").mkdir(exist_ok=True)
        Path("reports/SKIPPED.txt").write_text(f"Drift report skipped: {e}\n")
        exit(0)

    df = pd.read_csv(config["data"]["processed_path"], parse_dates=["time"])

    today = date.today()
    current   = df[(df["time"].dt.year == today.year)     & (df["time"].dt.month == today.month)]
    reference = df[(df["time"].dt.year == today.year - 1) & (df["time"].dt.month == today.month)]

    if len(current) < 5 or len(reference) < 5:
        print(f"Not enough data for month-over-year comparison "
              f"(current: {len(current)} rows, same month last year: {len(reference)} rows) — skipping.")
        Path("reports").mkdir(exist_ok=True)
        Path("reports/SKIPPED.txt").write_text(
            f"Drift report skipped: current={len(current)} rows, reference={len(reference)} rows\n"
        )
        exit(0)

    current   = current.reset_index(drop=True)
    reference = reference.reset_index(drop=True)

    reference["prediction"] = model.predict(reference[feat_cols])
    current["prediction"]   = model.predict(current[feat_cols])

    data_definition = DataDefinition(
        numerical_columns=feat_cols,
        regression=[Regression(target="target", prediction="prediction")],
    )
    reference_dataset = Dataset.from_pandas(reference, data_definition=data_definition)
    current_dataset   = Dataset.from_pandas(current,   data_definition=data_definition)

    report = Report([DataDriftPreset(), RegressionPreset()])
    my_eval = report.run(current_dataset, reference_dataset)

    Path("reports").mkdir(exist_ok=True)
    my_eval.save_html(f"reports/drift_report_{today.isoformat()}.html")
    my_eval.save_html("reports/latest_drift_report.html")

    print(f"Drift report saved — {today.strftime('%B %Y')} vs {today.replace(year=today.year-1).strftime('%B %Y')}")