def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)

    # Calendar feature — captures seasonality
    df["day_of_year"] = df["time"].dt.dayofyear

    # Target: next day's max temperature
    df["target"] = df["temperature_2m_max"].shift(-1)

    return df




###########################################################
############## Must still update this #####################
###########################################################
