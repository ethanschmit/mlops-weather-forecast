# src/ingest.py
# Pulls latest weather data from Open-Meteo and saves to data/raw/
# Run: python src/ingest.py

import sys
from pathlib import Path
# Add project root to path so we can import config utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import pandas as pd
import yaml
from datetime import date, timedelta
from pathlib import Path


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def fetch_weather(config: dict) -> pd.DataFrame:
    cfg = config["data"]
    # Uses the Historical Weather API (archive-api), not /v1/forecast:
    # /v1/forecast only serves ~16 days of forecast — not enough for days_back=365.
    # ERA5 reanalysis (which archive-api is built on) has a ~5 day processing
    # delay, so end_date can't be today — pull up to 7 days ago to be safe.
    end_date   = date.today() - timedelta(days=7)
    start_date = end_date - timedelta(days=cfg["days_back"])

    params = {
        "latitude":  cfg["latitude"],
        "longitude": cfg["longitude"],
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",       # note: underscore between "wind" and "speed"
            "shortwave_radiation_sum",
        ],
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "timezone":   cfg["timezone"],
    }

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    daily = response.json()["daily"]
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


if __name__ == "__main__":
    config = load_config()
    raw_path = Path(config["data"]["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching weather data...")
    df = fetch_weather(config)
    df.to_csv(raw_path, index=False)
    print(f"Saved {len(df)} rows → {raw_path}")