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
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset


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
        print(f"No champion model yet ({e}) — skipping drift report until first promotion.")
        Path("reports").mkdir(exist_ok=True)
        Path("reports/SKIPPED.txt").write_text(f"Drift report skipped: {e}\n")
        exit(0)

    df = pd.read_csv(config["data"]["processed_path"], parse_dates=["time"])
    reference = df.iloc[:-window_days].reset_index(drop=True)
    current   = df.iloc[-window_days:].reset_index(drop=True)

    reference["prediction"] = model.predict(reference[feat_cols])
    current["prediction"]   = model.predict(current[feat_cols])

    data_definition = DataDefinition(
        numerical_columns=feat_cols,
        regression=[Regression(target="target", prediction="prediction")],
    )
    reference_dataset = Dataset.from_pandas(reference, data_definition=data_definition)
    current_dataset   = Dataset.from_pandas(current,   data_definition=data_definition)

    report = Report([DataDriftPreset(), RegressionPreset()])
    # current dataset is the FIRST argument, reference is the SECOND — opposite
    # of the old ColumnMapping-era API, easy to get backwards
    my_eval = report.run(current_dataset, reference_dataset)

    Path("reports").mkdir(exist_ok=True)
    my_eval.save_html(f"reports/drift_report_{date.today().isoformat()}.html")
    my_eval.save_html("reports/latest_drift_report.html")

    print("Drift report saved → reports/latest_drift_report.html")