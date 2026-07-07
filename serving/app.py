# serving/app.py
# FastAPI app that loads the Production model from MLflow registry
# and serves predictions via HTTP.
# The model loaded is ALWAYS whatever is currently tagged "Production"
# in the registry — so when evaluate.py promotes a new model,
# this app serves it on the next restart.

import mlflow.sklearn
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

FEATURE_COLS = config["features"]["feature_cols"]
MODEL_NAME   = config["model"]["name"]

app = FastAPI(
    title="Weather Forecast API",
    description="Predicts tomorrow's max temperature",
    version="1.0",
)

# Load model at startup — reads from MLflow registry
# If you update the Production model, restart the container
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
print(f"Loaded model: {MODEL_NAME} (Production)")


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