# src/features.py
# Applies feature engineering decisions from the notebook.
# The logic here is the FROZEN result of notebook cell 3+4 — no experimentation.
# Run: python src/features.py

import pandas as pd
import yaml
from pathlib import Path


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)

    # This project's final feature set is all same-day raw values plus
    # day_of_year for seasonality — lag/rolling features were tried in the
    # notebook (Cell 3/4) but did not survive feature selection, so there
    # is no lag/rolling logic to replicate here. See config.yaml's
    # features.feature_cols for the exact list this model expects.

    # Calendar feature — captures seasonality
    df["day_of_year"] = df["time"].dt.dayofyear

    # Target: next day's max temperature
    df["target"] = df["temperature_2m_max"].shift(-1)

    # Drop rows missing either features or target
    df = df.dropna().reset_index(drop=True)
    return df


if __name__ == "__main__":
    config = load_config()

    raw = pd.read_csv(config["data"]["raw_path"], parse_dates=["time"])
    processed = build_features(raw, config)

    out_path = Path(config["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out_path, index=False)

    print(f"Features built: {processed.shape}")
    print(f"Columns: {list(processed.columns)}")