# src/drift_report.py
# Compares this calendar month's data against the SAME calendar month last
# year — a fairer comparison than last-N-days-vs-everything, since it removes
# seasonality as a false-positive drift source (May 2026 vs May 2025, not
# May 2026 vs the entire 10-year history).
# Run: python src/drift_report.py
# Output: reports/latest_drift_report.html (overwritten daily) + a dated copy


from pathlib import Path
from datetime import date
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
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
 
    # Load via run_id, NOT models:/name@alias directly — MLflow 3.x has a
    # known bug where alias-based artifact resolution fails to find files
    # that load fine by run_id or version number (see mlflow/mlflow#16429).
    try:
        champion_version = client.get_model_version_by_alias(model_name, "champion")
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{champion_version.version}")
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
    # current dataset is the FIRST argument, reference is the SECOND
    my_eval = report.run(current_dataset, reference_dataset)
 
    Path("reports").mkdir(exist_ok=True)
    my_eval.save_html(f"reports/drift_report_{today.isoformat()}.html")
    my_eval.save_html("reports/latest_drift_report.html")
 
    print(f"Drift report saved — {today.strftime('%B %Y')} vs {today.replace(year=today.year-1).strftime('%B %Y')}")