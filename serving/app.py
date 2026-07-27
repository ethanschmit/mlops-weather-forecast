import os
import joblib
import mlflow
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

with open("config.yaml") as f:
    config = yaml.safe_load(f)

FEATURE_COLS = config["features"]["feature_cols"]
MODEL_NAME   = config["model"]["name"]

app = FastAPI(
    title="Weather Forecast API",
    description="Predicts tomorrow's max temperature",
    version="1.0",
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    config.get("mlflow", {}).get("tracking_uri", "http://127.0.0.1:5000"),
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()
champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
local_dir = download_artifacts(run_id=champion_version.run_id, artifact_path="pickled_model")
model = joblib.load(os.path.join(local_dir, "model.joblib"))
print(f"Loaded model: {MODEL_NAME} (champion, run {champion_version.run_id})")

class WeatherInput(BaseModel):
    temperature_2m_max: float
    temperature_2m_min: float
    wind_speed_10m_max: float
    shortwave_radiation_sum: float
    precipitation_sum: float
    day_of_year: int


class PredictionResponse(BaseModel):
    predicted_tmax_tomorrow_celsius: float
    model_name: str


@app.get("/health")
def health():
    """Used by Docker and load balancers to check the container is alive."""
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: WeatherInput):
    try:
        df = pd.DataFrame([data.dict()])[FEATURE_COLS]
        pred = float(model.predict(df)[0])
        return PredictionResponse(
            predicted_tmax_tomorrow_celsius=round(pred, 2),
            model_name=MODEL_NAME,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))