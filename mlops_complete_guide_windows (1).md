# MLOps: Complete Guide — Notebook to Production
### One document. Work through it top to bottom.

> **This version is adapted for Windows + Cursor + Git Bash.** Every command below uses bash syntax and works as-is in Git Bash. If you ever switch to Cursor's PowerShell terminal instead, commands will need translating (e.g. `\` line-continuations become `` ` ``, `source .venv/Scripts/activate` becomes `.venv\Scripts\Activate.ps1`).

---

## What You Are Building

A weather forecasting model that:
- Pulls daily data from a free API (no account needed)
- Is experimented on in a Jupyter notebook
- Gets cleaned up into production scripts
- Is tracked and versioned with MLflow
- Is served via a FastAPI endpoint inside Docker
- Retrains automatically via GitHub Actions every day

This lives in **its own GitHub repo** — `mlops-weather-forecast`. When you build a second model (finance, fraud, etc.) it gets its own repo. Your GitHub profile becomes a clean list of standalone projects.

---

## Part 0: One-Time Machine Setup

Do this once. Never again.

### Install These Tools

| Tool | What it does | Download |
|---|---|---|
| Cursor | Your editor | cursor.com |
| Docker Desktop | Runs containers | docker.com/products/docker-desktop |
| Git for Windows | Version control + Git Bash terminal | git-scm.com |
| Python 3.12 | Language (avoid 3.14 — see note below) | python.org/downloads/windows |

**Cursor Extensions** — install from the Extensions panel (blocks icon, left sidebar):
- Python (by Microsoft)
- Jupyter (by Microsoft)
- Docker (by Microsoft)
- GitLens
- YAML (by Red Hat)

**Set Cursor's default terminal to Git Bash:** `Ctrl+Shift+P` → `Terminal: Select Default Profile` → choose **Git Bash**. This means every command in this guide (written in bash syntax) works as-is, with no translation to PowerShell needed.

**Why Python 3.12, not the newest version?** MLflow (and a lot of production ML tooling) tends to lag a version or two behind Python's latest release. As of this guide, Python 3.14 breaks MLflow's server with an `ImportError: cannot import name 'Traversable' from importlib.abc` — a known compatibility bug, not something you did wrong. Installing 3.12 alongside any newer version you may already have is safe; use the `py` launcher (`py -3.12 ...`, `py -0` to list installed versions) to be explicit about which one a command uses.

**Fix Windows' terminal encoding, once, so emoji output from libraries (MLflow prints a 🏃 on every run) doesn't crash your scripts:**
```bash
echo 'export PYTHONIOENCODING=utf-8' >> ~/.bashrc
source ~/.bashrc
```
Without this, you'll eventually hit `UnicodeEncodeError: 'charmap' codec can't encode character...` — Windows' default terminal encoding (`cp1252`) can't display emoji, and some libraries print them. This is a one-time fix that applies to every future terminal session.

---

### Daily Startup Routine (do this every time you sit down to work)

Since MLflow and Docker are **local processes tied to your laptop** — not cloud services — they stop the moment you close Docker Desktop, shut down, or close the terminal tab they're running in. GitHub Actions is the one exception (it runs on GitHub's own servers, independent of your laptop — see Part 8).

**Every work session, in order:**

1. **Start MLflow** (Terminal Tab 1 — leave running):
```bash
cd ~/Projects/mlops-weather-forecast
source .venv/Scripts/activate
mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "127.0.0.1:*,localhost:*,host.docker.internal:*"
```
`--host 0.0.0.0` and `--allowed-hosts` (rather than the simpler `--host 127.0.0.1`) are required so that **both** your local Python scripts *and* a Docker container can reach this server without a 403 "Invalid Host header" rejection.

2. **If you want the Docker container running too:**
```bash
docker start weather-api
```
(Only works if you've already built and run it once with `docker run -d -p 8000:8000 --name weather-api weather-forecast:latest` — see Part 5.) This will fail with a connection error if step 1 isn't already up, since the container loads its model from MLflow at startup.

3. **Confirm everything is alive:**
```bash
curl http://127.0.0.1:5000        # MLflow UI
curl http://localhost:8000/health # your API, if running
```

**When you're done for the day:** just close things down, nothing to clean up — `Ctrl+C` on the MLflow tab, `docker stop weather-api` if you started the container. Tomorrow, repeat from step 1.

---

### Configure Git (one-time)

Open a terminal (any terminal, not inside a project yet):

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

This stamps every commit you make with your identity. You only do this once per machine.

### Create a GitHub Account

Go to github.com and create an account if you don't have one. This is where your repos will live publicly.

---

## Part 1: Create the Project

### Step 1.1 — Create the Repo on GitHub First

Go to github.com → click the `+` icon → New repository.

- Name: `mlops-weather-forecast`
- Description: `End-to-end MLOps pipeline: daily weather data → GBM forecast → FastAPI → Docker`
- Set to **Public** (this is your portfolio)
- Tick **Add a README file**
- Click **Create repository**

**Why GitHub first?** Because you clone it to your machine. This means Git tracking is set up automatically — you don't have to `git init` and connect remotes manually.

### Step 1.2 — Clone it to Your Machine

In Cursor: `Ctrl+Shift+P` → type `Git: Clone` → paste your repo URL (looks like `https://github.com/yourusername/mlops-weather-forecast.git`) → choose where to save it → click Open when prompted.

Or in terminal:

```bash
cd ~/Documents          # or wherever you keep projects
git clone https://github.com/yourusername/mlops-weather-forecast.git
cd mlops-weather-forecast
cursor .                # opens this folder in Cursor
```

You now have a local copy of the repo, already connected to GitHub.

### Step 1.3 — Create the Folder Structure

Open the Cursor terminal with Ctrl+` (backtick) — make sure it's Git Bash (see Part 0). Run this entire block:

```bash
mkdir -p notebooks \
         src \
         data/raw \
         data/processed \
         serving \
         .github/workflows

touch notebooks/.gitkeep \
      src/__init__.py \
      src/ingest.py \
      src/features.py \
      src/train.py \
      src/evaluate.py \
      serving/app.py \
      serving/Dockerfile \
      pipeline.sh \
      config.yaml \
      requirements.txt \
      .gitignore
```

**Why terminal instead of clicking?** One command, reproducible on any machine. On a server or in CI there is no GUI — terminal is the only option.

### Step 1.4 — Create .gitignore

Open `.gitignore` and paste this. These are files/folders that must **never** be committed to GitHub:

```
# Data — never commit raw or processed data
data/

# Python environment — machine-specific, not portable
.venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# MLflow tracking — lives locally or on a server, not in git
mlruns/
mlartifacts/

# OS files
.DS_Store
Thumbs.db

# Secrets — never ever commit these
.env
*.key
```

**Why does this matter?** If you commit your `data/` folder you'll push megabytes of CSV files to GitHub. If you commit `.env` files you'll expose API keys publicly. The `.gitignore` prevents both.

### Step 1.5 — Set Up Python Environment

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate

pip install mlflow scikit-learn pandas requests \
            fastapi uvicorn pytest pyyaml jupyter matplotlib seaborn

pip freeze > requirements.txt
```

**Why a virtual environment?** It's an isolated Python installation just for this project. If you `pip install` globally, packages from different projects clash. The `.venv` keeps everything clean and reproducible — anyone can run `pip install -r requirements.txt` and get the exact same setup.

In Cursor: `Ctrl+Shift+P` → `Python: Select Interpreter` → choose the `.venv` one (requires the Python extension by Microsoft to be installed — Extensions icon, search "Python"). Cursor will now activate it automatically per this project folder.

### Step 1.6 — First Git Push

You've done real work — create the structure and commit it:

```bash
git add .
git commit -m "feat: initial project structure and dependencies"
git push origin main
```

**Breaking this down:**
- `git add .` — stages everything (tells git "track these changes")
- `git commit -m "..."` — saves a snapshot with a message
- `git push origin main` — sends that snapshot to GitHub

Go to your GitHub repo page and refresh — you'll see all your files there.

**When to commit:** After every meaningful chunk of work. Not every line, not every hour — after logical units. Good commit messages are: `feat: add feature engineering`, `fix: handle missing values in rolling mean`, `refactor: move MLflow utils to config`.

---

## Part 2: The Config File — Decisions Live Here, Not in Code

Before writing any model code, create `config.yaml`. This is one of the most important professional habits. Every number that could change (feature list, model params, thresholds) lives here. Scripts read from it. When you retrain with different params you change the config, not the code.

```yaml
# config.yaml

model:
  name: "weather-forecast"
  algorithm: "GradientBoostingRegressor"
  params:
    n_estimators: 200       # filled in AFTER notebook experimentation
    max_depth: 4            # filled in AFTER notebook experimentation
    learning_rate: 0.05     # filled in AFTER notebook experimentation
    random_state: 42

data:
  latitude: -33.92          # Cape Town
  longitude: 18.42
  timezone: "Africa/Johannesburg"
  days_back: 3650           # ~10 years — pulled from the Historical Weather API
  raw_path: "data/raw/weather.csv"
  processed_path: "data/processed/features.csv"

features:
  target_col: "target"
  feature_cols:             # filled in AFTER notebook experimentation
    - temperature_2m_max
    - temperature_2m_min
    - wind_speed_10m_max
    - shortwave_radiation_sum
    - precipitation_sum
    - day_of_year

evaluation:
  mae_threshold: 3.0        # filled in AFTER notebook experimentation
  improvement_pct: 0.05     # new model must beat prod by at least 5%

mlflow:
  experiment_name: "weather-forecast"
  tracking_uri: "http://127.0.0.1:5000"
```

Notice the comments — the params are placeholders. The notebook (Part 3) is where you discover the real values. Then you come back and fill them in here.

---

## Part 3: The Notebook — Where All Thinking Happens

**This is the answer to your question about where model values come from.**

The notebook is a scratchpad. You try things, plot things, compare things. It is intentionally messy. At the end of the notebook you have made four concrete decisions that get written into `config.yaml`. The production scripts then just re-execute those decisions on new data automatically.

Start MLflow's tracking server in one terminal tab before opening the notebook — you want it logging from the start:

```bash
# Terminal Tab 1 — leave this running the whole time
source .venv/Scripts/activate
mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "127.0.0.1:*,localhost:*,host.docker.internal:*"
```

Open a second tab, start Jupyter:

```bash
# Terminal Tab 2
source .venv/Scripts/activate
jupyter notebook
```

Create `notebooks/01_experimentation.ipynb`. Work through each cell:

---

### Notebook Cell 1: Pull and Inspect Raw Data

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta

# Pull a full year of daily weather data — Cape Town
# Uses the Historical Weather API (archive-api), NOT /v1/forecast:
# /v1/forecast only serves ~16 days of forecast, not a full year of history.
# ERA5 reanalysis data (which the archive API is built on) has a ~5 day
# processing delay, so end_date can't be today — pull up to 7 days ago to be safe.
end_date = date.today() - timedelta(days=7)
start_date = end_date - timedelta(days=365)

response = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
    "latitude": -33.92,
    "longitude": 18.42,
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",       # note: underscore between "wind" and "speed"
        "shortwave_radiation_sum",
    ],
    "start_date": start_date.isoformat(),
    "end_date": end_date.isoformat(),
    "timezone": "Africa/Johannesburg",
})

response.raise_for_status()   # throws an error if the API call failed
df = pd.DataFrame(response.json()["daily"])
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

print(f"Shape: {df.shape}")
print(f"Date range: {df['time'].min()} → {df['time'].max()}")
print(df.head())
```

**What you are looking for:** How many rows, what columns, what date range. Does the data look sensible?

**Full list of available variables and what they mean:** see Open-Meteo's own docs at [open-meteo.com/en/docs/historical-weather-api](https://open-meteo.com/en/docs/historical-weather-api) — scroll to "Daily Parameter Definition" for a plain-language description of every field (units, what it measures). Worth resisting the urge to pull in every variable available — see the note at the end of Part 3 on why a short, deliberately-chosen feature list beats an exhaustive one.

---

### Notebook Cell 2: Understand the Data Quality

```python
print("=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Basic Statistics ===")
print(df.describe())

# Plot the target variable — what are we predicting?
plt.figure(figsize=(14, 4))
plt.plot(df["time"], df["temperature_2m_max"], linewidth=0.8)
plt.title("Daily Max Temperature — do I see seasonality? gaps? outliers?")
plt.ylabel("°C")
plt.tight_layout()
plt.show()
```

**What you are looking for:**
- Missing values: if a column has >10% missing, it may not be usable
- Seasonality: you can see it in the temperature wave — this tells you calendar features (month, day_of_year) will be useful
- Outliers: any spikes that look like data errors

---

### Notebook Cell 3: Build Candidate Features

```python
df_feat = df.copy().sort_values("time").reset_index(drop=True)

# Lag features — yesterday's values predict tomorrow
# You MUST shift before computing lags to prevent data leakage
for lag in [1, 2, 3, 7]:
    df_feat[f"tmax_lag{lag}"] = df_feat["temperature_2m_max"].shift(lag)
    df_feat[f"tmin_lag{lag}"] = df_feat["temperature_2m_min"].shift(lag)
    df_feat[f"precip_lag{lag}"] = df_feat["precipitation_sum"].shift(lag)

# Rolling stats — smooth out noise, capture trends
# .shift(1) BEFORE .rolling() — critical to not leak today's value into the window
df_feat["tmax_roll2_mean"] = df_feat["temperature_2m_max"].shift(1).rolling(2).mean()
df_feat["tmax_roll2_std"]  = df_feat["temperature_2m_max"].shift(1).rolling(2).std()
df_feat["tmax_roll4_mean"] = df_feat["temperature_2m_max"].shift(1).rolling(4).mean()

# Calendar features — capture seasonality
df_feat["day_of_year"] = df_feat["time"].dt.dayofyear
df_feat["month"]       = df_feat["time"].dt.month
df_feat["week"]        = df_feat["time"].dt.isocalendar().week.astype(int)

# Pass-through features from the raw data
# wind_speed_10m_max, shortwave_radiation_sum, precipitation_sum, and
# temperature_2m_min/max are already daily aggregates — use as-is.
# They are lag-safe as SAME-DAY predictors: today's conditions are fully
# known at prediction time, so using them to predict TOMORROW's max is fine.

# Target: NEXT day's max temperature
# Shift by -1 so that for each row, target = what we want to predict
df_feat["target"] = df_feat["temperature_2m_max"].shift(-1)

# Drop rows where we can't form complete features OR target
df_clean = df_feat.dropna().reset_index(drop=True)
print(f"Rows after dropping NaN: {len(df_clean)} (lost {len(df_feat)-len(df_clean)} to lag/rolling warmup)")
print(df_clean.head(3))
```

**Why `.shift(1)` before `.rolling()`?** This is the data leakage point. If you compute a rolling mean including today's value, and today's value is correlated with tomorrow's target (which it is — temperature is autocorrelated), your model learns from the future. Always shift before rolling in time series.

> **Note on this project's actual result:** after running Cell 4's importance check below with 10 years of data, the raw same-day values (`temperature_2m_max`, `temperature_2m_min`, `wind_speed_10m_max`, `shortwave_radiation_sum`, `precipitation_sum`) plus `day_of_year` outperformed all the engineered lag/rolling features — none of the lag or rolling columns made the final cut. That's a legitimate outcome, not a mistake: with more data, the model found the raw values carried cleaner signal than smoothed/lagged versions. The lag/rolling code above is still worth knowing how to build — it's the kind of feature engineering that *does* often help on other datasets — but it didn't survive selection for this one.

---

### Notebook Cell 4: Feature Selection — Which Features Actually Help?

```python
from sklearn.ensemble import GradientBoostingRegressor

# Define all candidates you created above
candidate_features = [
    "tmax_lag1", "tmax_lag2", "tmax_lag3", "tmax_lag7",
    "tmin_lag1", "tmin_lag2", "tmin_lag3",
    "precip_lag1", "precip_lag2", "precip_lag3",
    "tmax_roll2_mean", "tmax_roll2_std", "tmax_roll4_mean",
    "temperature_2m_max", "temperature_2m_min",
    "wind_speed_10m_max", "shortwave_radiation_sum", "precipitation_sum",
    "day_of_year", "month", "week",
]

X_all = df_clean[candidate_features]
y     = df_clean["target"]

# Quick correlation check — which features are linearly related to the target?
correlations = X_all.corrwith(y).abs().sort_values(ascending=False)
print("Correlation with target:\n", correlations)
```

```python
# Train a quick model on ALL features and look at importances
quick_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
quick_model.fit(X_all, y)

importances = pd.Series(quick_model.feature_importances_, index=candidate_features)
importances.sort_values().plot(kind="barh", figsize=(8, 6), color="steelblue")
plt.title("Feature Importances — drop anything near zero")
plt.xlabel("Importance score")
plt.tight_layout()
plt.show()

# Print the bottom ones explicitly
print("\nLowest importance features (candidates to drop):")
print(importances.sort_values().head(5))
```

**What you are doing here:** You look at this chart and make a judgment call. Features with importance near 0.00 are noise — they add complexity without adding prediction power. Typically you'd drop anything below 0.01 or 0.02.

**What actually happened on this project, with 10 years of Cape Town data:** the raw same-day values — `temperature_2m_max`, `temperature_2m_min`, `wind_speed_10m_max`, `shortwave_radiation_sum`, `precipitation_sum` — dominated the importance chart, along with `day_of_year` for seasonality. All the engineered lag features (`tmax_lag1`, etc.) and rolling stats (`tmax_roll2_mean`, etc.) scored near zero once the raw values were available — the model found today's actual conditions more informative than smoothed or lagged versions of them. `month` and `week` also contributed almost nothing once `day_of_year` was included (redundant seasonality signal) → dropped.

**This is where your feature list for `config.yaml` comes from.** You make the decision here, once, in the notebook. Don't assume lags/rolling stats will always win just because they're common in time-series tutorials — always let the actual importance chart decide for your specific dataset.

---

### Notebook Cell 5: Compare Algorithms

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import numpy as np

# The FINAL feature list — after dropping weak ones above
# On this project, the raw same-day values won out over every
# engineered lag/rolling feature (see Cell 4 note above)
final_features = [
    "temperature_2m_max", "wind_speed_10m_max", "shortwave_radiation_sum",
    "temperature_2m_min", "day_of_year", "precipitation_sum",
]

X = df_clean[final_features]
y = df_clean["target"]

# TimeSeriesSplit is critical — never use regular KFold on time series data
# Regular KFold shuffles randomly, so fold 3 could train on data from the future
# TimeSeriesSplit always trains on past, validates on future — mirrors real deployment
tscv = TimeSeriesSplit(n_splits=5)

models_to_try = {
    "Ridge":  Ridge(),
    "RF":     RandomForestRegressor(n_estimators=100, random_state=42),
    "GBM":    GradientBoostingRegressor(n_estimators=100, random_state=42),
}

print("Algorithm comparison (CV MAE ± std):")
results = {}
for name, model in models_to_try.items():
    scores = cross_val_score(
        model, X, y, cv=tscv, scoring="neg_mean_absolute_error"
    )
    mae_mean = -scores.mean()
    mae_std  = scores.std()
    results[name] = mae_mean
    print(f"  {name:8s}: {mae_mean:.3f}°C ± {mae_std:.3f}")

winner = min(results, key=results.get)
print(f"\nWinner: {winner}")
# → This tells you which algorithm goes into config.yaml
```

---

### Notebook Cell 6: Tune the Winning Algorithm

```python
from sklearn.model_selection import GridSearchCV

# GBM won (most likely). Now find the best hyperparameters.
param_grid = {
    "n_estimators":  [100, 200, 300],
    "max_depth":     [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X, y)

print(f"\nBest params:  {grid_search.best_params_}")
print(f"Best CV MAE:  {-grid_search.best_score_:.3f}°C")

# EXAMPLE OUTPUT (your numbers will differ slightly):
# Best params:  {'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 200}
# Best CV MAE:  1.698°C
#
# ↑ These go directly into config.yaml under model.params
# ↑ This is where the "model parameter values" come from
```

---

### Notebook Cell 7: Final Holdout Validation

```python
from sklearn.metrics import mean_absolute_error, r2_score

# Simulate what will happen in production:
# Train on historical data, evaluate on the most recent unseen data
# Use a genuine time-based split — last 20% of dates for holdout
cutoff = int(len(df_clean) * 0.80)

X_train = df_clean[final_features].iloc[:cutoff]
X_test  = df_clean[final_features].iloc[cutoff:]
y_train = df_clean["target"].iloc[:cutoff]
y_test  = df_clean["target"].iloc[cutoff:]

best_params = grid_search.best_params_
final_model = GradientBoostingRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)
preds = final_model.predict(X_test)

holdout_mae = mean_absolute_error(y_test, preds)
holdout_r2  = r2_score(y_test, preds)

print(f"Holdout MAE: {holdout_mae:.3f}°C")
print(f"Holdout R²:  {holdout_r2:.3f}")

# Plot actual vs predicted on the holdout period
test_dates = df_clean["time"].iloc[cutoff:]
plt.figure(figsize=(14, 4))
plt.plot(test_dates, y_test.values, label="Actual", linewidth=1)
plt.plot(test_dates, preds,         label="Predicted", linewidth=1, alpha=0.8)
plt.title(f"Holdout Period — MAE: {holdout_mae:.2f}°C, R²: {holdout_r2:.3f}")
plt.legend()
plt.tight_layout()
plt.show()

# The holdout MAE is what you set as mae_threshold in config.yaml
# Because: a freshly retrained model on new data should at least match
# the performance you saw during experimentation
print(f"\n→ Set config.yaml mae_threshold to approximately: {holdout_mae + 0.5:.1f}")
# Adding 0.5°C buffer — allows a slightly worse model to still pass the gate
```

---

### Notebook Cell 8: Log This Experiment to MLflow

Even in the notebook, log to MLflow. This creates a record of your experimentation session in the same UI you'll use for production runs.

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("weather-forecast")

with mlflow.start_run(run_name="notebook-experimentation"):
    mlflow.log_params(best_params)
    mlflow.log_param("features_used", len(final_features))
    mlflow.log_metric("holdout_mae", holdout_mae)
    mlflow.log_metric("holdout_r2",  holdout_r2)
    mlflow.log_metric("cv_mae", -grid_search.best_score_)

    # Log the final model — not registering it yet (that's for the pipeline)
    mlflow.sklearn.log_model(final_model, artifact_path="model")

    print("Logged to MLflow ✓")
    print("Open http://127.0.0.1:5000 to see this run")
```

Go to `http://127.0.0.1:5000` now. You'll see your experiment, the run, all params and metrics. This is exactly what Azure ML's UI looks like — same concept.

---

### Notebook Summary: Four Decisions Made

At the end of your notebook you know:

| Decision | Where it came from | Goes into |
|---|---|---|
| Feature list | Cell 4 importance chart | `config.yaml: features.feature_cols` |
| Algorithm | Cell 5 comparison | `config.yaml: model.algorithm` |
| Hyperparameters | Cell 6 grid search | `config.yaml: model.params` |
| MAE threshold | Cell 7 holdout + buffer | `config.yaml: evaluation.mae_threshold` |

Go update `config.yaml` now with these real values before writing any production scripts.

**Git checkpoint — commit the notebook:**

```bash
git add notebooks/01_experimentation.ipynb config.yaml
git commit -m "feat: experimentation notebook, feature selection, hyperparameter tuning"
git push origin main
```

**Why commit the notebook?** It documents *why* you made those decisions. A hiring manager can look at it and see your thought process. It also means if something breaks in production later you can trace back to the reasoning.

---

## Part 4: Production Scripts — Crystallising the Notebook Decisions

The scripts do not explore or experiment. They re-execute the decisions you made in the notebook. Every script is runnable standalone from the command line.

### src/ingest.py

```python
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
```

Test it:

```bash
python src/ingest.py
# Should print: Saved 365 rows → data/raw/weather.csv
cat data/raw/weather.csv | head -5
```

---

### src/features.py

```python
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
```

Test it:

```bash
python src/features.py
```

---

### src/train.py

```python
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

    # Connect to MLflow — must be running (mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "127.0.0.1:*,localhost:*,host.docker.internal:*")
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
```

Test it (MLflow must still be running in Tab 1):

```bash
python src/train.py
# Should print something like:
# RUN_ID=3a7f2b1c8d4e CV_MAE=1.823456
```

Check `http://127.0.0.1:5000` → you'll see the run logged, and under "Models" you'll see `weather-forecast` registered.

---

### src/evaluate.py

```python
# src/evaluate.py
# Compares new model against current production model in the registry.
# Promotes if better, rejects if not.
# Run: python src/evaluate.py --run_id <id> --mae <value>
# Exit code: 0 = promoted, 1 = rejected (used by pipeline.sh)

import sys
import argparse
import yaml
import mlflow
from mlflow.tracking import MlflowClient


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def promote_model(run_id: str, new_mae: float, config: dict) -> bool:
    client     = MlflowClient()
    model_name = config["model"]["name"]
    threshold  = config["evaluation"]["mae_threshold"]
    min_improv = config["evaluation"]["improvement_pct"]

    # Gate 1: absolute quality — reject garbage models regardless of comparison
    if new_mae > threshold:
        print(f"REJECTED: CV MAE {new_mae:.3f} > threshold {threshold}")
        return False

    # Gate 2: compare against current production model (if one exists)
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            prod_run = client.get_run(prod_versions[0].run_id)
            prod_mae = float(prod_run.data.metrics["cv_mae"])

            required_mae = prod_mae * (1 - min_improv)
            if new_mae > required_mae:
                print(f"REJECTED: {new_mae:.3f} not enough better than prod {prod_mae:.3f} "
                      f"(needed < {required_mae:.3f})")
                return False

            # Archive the old production model before promoting new one
            client.transition_model_version_stage(
                name=model_name,
                version=prod_versions[0].version,
                stage="Archived",
            )
            print(f"Archived previous prod model v{prod_versions[0].version} (MAE: {prod_mae:.3f})")

    except Exception as e:
        print(f"No existing prod model ({e}) — promoting directly")

    # Promote the newest registered version
    new_versions = client.get_latest_versions(model_name, stages=["None"])
    if not new_versions:
        print("ERROR: No model version found in 'None' stage to promote")
        return False

    client.transition_model_version_stage(
        name=model_name,
        version=new_versions[-1].version,
        stage="Production",
    )
    print(f"PROMOTED: v{new_versions[-1].version} → Production (MAE: {new_mae:.3f})")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--mae",    required=True, type=float)
    args = parser.parse_args()

    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    success = promote_model(args.run_id, args.mae, config)
    sys.exit(0 if success else 1)
```

---

### Git checkpoint — commit all scripts:

```bash
git add src/
git commit -m "feat: production scripts — ingest, features, train, evaluate"
git push origin main
```

---

## Part 5: The Serving Layer — FastAPI + Docker

### serving/app.py

```python
# serving/app.py
# FastAPI app that loads the Production model from MLflow registry
# and serves predictions via HTTP.
# The model loaded is ALWAYS whatever is currently tagged "Production"
# in the registry — so when evaluate.py promotes a new model,
# this app serves it on the next restart.

import mlflow.sklearn
import pandas as pd
import yaml
import os
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

# Read tracking URI from environment variable first — this lets Docker
# override it (to host.docker.internal:5000, via the Dockerfile's ENV line)
# without touching this code. Falls back to config.yaml / localhost for
# plain local (non-Docker) runs.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    config.get("mlflow", {}).get("tracking_uri", "http://127.0.0.1:5000"),
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
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
```

Before running this, make sure `serving/` is an importable Python package — create an empty `__init__.py` inside it if you haven't already (same idea as `src/__init__.py` from Part 0's folder structure):
```bash
touch serving/__init__.py
```

Test it locally first, before Docker. **Run this from the project root, not from inside `serving/`** — `config.yaml` is loaded with a relative path, so the terminal's current folder needs to be the project root for `load_config()`/the config-loading line in `app.py` to find it:

```bash
uvicorn serving.app:app --reload --port 8000
```

Note the `serving.` prefix (dot, not slash) — this tells uvicorn "find `app` inside the `serving` package," while your terminal stays at the root so `config.yaml` resolves correctly. If you instead see `FileNotFoundError: config.yaml`, it means you're running from inside `serving/` — `cd` back to the project root and use the command above.

In a new tab:

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"weather-forecast"}

# Test prediction (use real values from your processed CSV)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature_2m_max": 22.4, "temperature_2m_min": 12.1,
    "wind_speed_10m_max": 14.3, "shortwave_radiation_sum": 18.5,
    "precipitation_sum": 0.0, "day_of_year": 180
  }'
# {"predicted_tmax_tomorrow_celsius": 21.87, "model_name": "weather-forecast"}
```

Stop uvicorn with `Ctrl+C`.

---

### serving/requirements.txt

This is a **separate, leaner file from the root `requirements.txt`** — it only lists what `app.py` actually imports at runtime (no Jupyter, matplotlib, pytest, etc., which the dev environment needs but the container doesn't). This also avoids dragging in Windows-only packages (`pywin32`, `pywinpty`) that `pip freeze` on Windows can silently include and that fail to build on Linux.

```
mlflow
scikit-learn
pandas
pyyaml
fastapi
uvicorn
```

**Best practice for pinning versions:** build once with the unpinned list above, then capture the exact, Linux-verified versions from inside the container itself (not from your Windows `.venv`):
```bash
docker run --rm weather-forecast:latest pip freeze > serving/requirements.txt
```
This guarantees every version in the file actually installed and worked together on Linux — the same OS the container runs on — rather than reusing a Windows-side freeze that may not have matching Linux builds.

### serving/Dockerfile

```dockerfile
# serving/Dockerfile
# Packages the FastAPI app + its environment into a portable container.
# The container connects to MLflow on startup to load the Production model.

FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies FIRST — Docker caches this layer
# so rebuilds are fast if you only change code, not requirements
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY serving/app.py .
COPY config.yaml .
COPY src/ ./src/

# Where MLflow tracking server lives — overridden at runtime if needed
# host.docker.internal resolves to your host machine from inside a container
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

EXPOSE 8000

# Healthcheck: Docker will ping /health every 30s
# Container is marked unhealthy if this fails 3 times
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Note the `FROM python:3.12-slim`** — this must match the Python version your `requirements.txt` was generated against. Since this project's `.venv` runs 3.12 (see Part 0's note on why 3.14 was avoided), the container needs 3.12 too, or pinned packages like `numpy` may not have a matching build for the container's Python version.

Build and run the Docker image:

```bash
# Run from the PROJECT ROOT (not from inside serving/)
# Because the Dockerfile COPYs src/ and config.yaml which are at root level

docker build -f serving/Dockerfile -t weather-forecast:latest .

# Run it DETACHED (-d) so it keeps running in the background,
# independent of this terminal tab staying open, and give it a
# memorable --name so you can start/stop/check it later without
# rebuilding: docker start weather-api / docker stop weather-api
docker run -d -p 8000:8000 --name weather-api weather-forecast:latest
```

**Prerequisite: your MLflow server must already be running** (see Part 3/setup) before starting the container — `app.py` needs to reach it at startup to load the "Production" model. Also, MLflow needs to be started with `--host 0.0.0.0` and an `--allowed-hosts` list that includes `host.docker.internal:*`, or the container's request will be rejected with a 403 "Invalid Host header" error. See the "Daily Startup Routine" note in Part 0 for the exact command.

**Check the container is alive:**
```bash
docker logs weather-api
```
Look for `Loaded model: weather-forecast (Production)` and `Application startup complete`.

While it's running, open a new tab:

```bash
docker ps                           # see running containers
curl http://localhost:8000/health   # still works — now from inside Docker
```

**What just happened:** Your model is now running in a fully isolated container. It has its own OS, its own Python, its own dependencies. It connects to your local MLflow server on startup to load the Production model. This container could now be deployed to any cloud provider unchanged.

**Understanding the Docker build output:** Each line in the Dockerfile is a "layer". Docker caches layers. If you change only `app.py` and rebuild, Docker reuses the cached `pip install` layer — rebuild takes seconds not minutes. That's why dependencies are copied and installed before the application code.

Stop the container with `Ctrl+C`.

**Git checkpoint:**

```bash
git add serving/
git commit -m "feat: FastAPI serving layer and Dockerfile"
git push origin main
```

---

## Part 6: The Pipeline Script — Chaining Everything Together

```bash
#!/bin/bash
# pipeline.sh
# Runs the full retraining pipeline: ingest → features → train → evaluate
# Called by GitHub Actions on a schedule, or manually with: bash pipeline.sh
#
# set -e means: if any command returns a non-zero exit code, stop immediately
# Without this, a failed ingest would silently continue to train on stale data
set -e
set -o pipefail    # catch errors inside pipes too

echo "======================================"
echo "PIPELINE START: $(date)"
echo "======================================"

echo ""
echo "[1/4] Ingesting data..."
python src/ingest.py

echo ""
echo "[2/4] Building features..."
python src/features.py

echo ""
echo "[3/4] Training model..."
# Capture the output line that contains RUN_ID and CV_MAE
TRAIN_OUTPUT=$(python src/train.py)
echo "$TRAIN_OUTPUT"

# Parse RUN_ID and MAE from the output
# train.py prints exactly: RUN_ID=abc123 CV_MAE=1.823456
RUN_ID=$(echo "$TRAIN_OUTPUT" | grep -oP 'RUN_ID=\K\S+')
MAE=$(echo "$TRAIN_OUTPUT"    | grep -oP 'CV_MAE=\K\S+')

echo ""
echo "[4/4] Evaluating and promoting (run=$RUN_ID, mae=$MAE)..."
python src/evaluate.py --run_id "$RUN_ID" --mae "$MAE"

echo ""
echo "======================================"
echo "PIPELINE COMPLETE: $(date)"
echo "======================================"
```

Make it executable and test it end to end:

```bash
chmod +x pipeline.sh
bash pipeline.sh
```

You should see all four steps execute and the model get promoted. Check `http://127.0.0.1:5000` — you'll see the new run.

---

## Part 7: GitHub Actions — Automating the Pipeline

This is the file that makes retraining happen without you touching anything.

```yaml
# .github/workflows/daily_retrain.yml
# Runs the full retraining pipeline every day at 6am UTC
# Also runnable manually from the GitHub Actions UI

name: Daily Model Retraining

on:
  schedule:
    - cron: "0 6 * * *"    # 6am UTC every day
  workflow_dispatch:         # manual trigger from GitHub UI

jobs:
  retrain:
    runs-on: ubuntu-latest   # GitHub provides a fresh Linux VM for each run

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        # Pulls your repo onto the GitHub runner machine

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Start MLflow server
        run: |
          mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "127.0.0.1:*,localhost:*,host.docker.internal:*" &
          sleep 5    # give it time to start before the pipeline calls it
        # The & runs it in the background

      - name: Run retraining pipeline
        run: bash pipeline.sh

      - name: Save MLflow runs as artifact
        uses: actions/upload-artifact@v4
        if: always()   # save even if pipeline failed — for debugging
        with:
          name: mlflow-runs-${{ github.run_number }}
          path: mlruns/
          retention-days: 30
```

**Important note:** In this local/free setup, the MLflow server is started fresh each run and the model artefacts are saved as GitHub Action artifacts. In production on Azure, MLflow would point to a persistent server and artefacts would be stored in Azure Blob Storage — but the `pipeline.sh` and all the Python scripts are identical.

Commit and push this:

```bash
git add pipeline.sh .github/
git commit -m "feat: orchestration pipeline and GitHub Actions daily schedule"
git push origin main
```

Go to your GitHub repo → Actions tab. You'll see the workflow listed. Click "Run workflow" to trigger it manually and watch it execute.

---

## Part 8: The Complete Flow — What Happens Every Day

```
6:00 AM UTC — GitHub Actions wakes up
      │
      ▼
Checks out your repo onto a fresh Linux VM
      │
      ▼
[1] src/ingest.py
    → calls Open-Meteo API
    → saves data/raw/weather.csv (now includes yesterday's data)
      │
      ▼
[2] src/features.py
    → reads data/raw/weather.csv
    → applies the SAME feature engineering as the notebook
    → saves data/processed/features.csv
      │
      ▼
[3] src/train.py
    → reads data/processed/features.csv
    → trains GBM with config.yaml params on ALL available data
    → logs run to MLflow (params, metrics, model artifact)
    → registers new model version in MLflow registry (stage: None)
    → prints RUN_ID and CV_MAE
      │
      ▼
[4] src/evaluate.py
    → compares new CV_MAE against Production model's CV_MAE
    → if better by ≥5% AND below absolute threshold:
          archives old Production model
          promotes new version to Production
    → if not better: rejects, Production model unchanged
      │
      ▼
serving/app.py (already running in Docker)
    → on next restart: loads whatever is now tagged "Production"
    → serves /predict with the updated model
```

**Scoring (inference) is separate from this entire flow.** The Docker container sits running all day, serving predictions from whatever model is in Production. The pipeline above is purely about replacing that model when a better version is found.

---

## Part 9: Git Workflow — When to Commit What

| When | Command | Message format |
|---|---|---|
| After creating project structure | `git add . && git commit` | `chore: initial project structure` |
| After notebook experimentation | `git add notebooks/ config.yaml` | `feat: experimentation, feature selection, tuning` |
| After writing each script | `git add src/` | `feat: add training script` |
| After fixing a bug | `git add <file>` | `fix: handle NaN in rolling features` |
| After changing config | `git add config.yaml` | `config: update mae_threshold to 2.8` |
| Before pushing to production | `git push origin main` | (after commit) |

**Never commit:** `data/`, `.venv/`, `mlruns/`, `.env` files — your `.gitignore` prevents this.

**Branch workflow (once comfortable):** Create a branch for experiments: `git checkout -b experiment/try-xgboost`. Work, commit, push. Then merge via a Pull Request on GitHub. This keeps `main` clean and always deployable.

---

## Part 10: What to Build Next (Second Repo)

When you start `mlops-finance-forecast`, the process is identical:

```bash
cd ~/Documents
git clone https://github.com/yourusername/mlops-finance-forecast.git
cd mlops-finance-forecast
code .
# ... same structure, different ingest.py, different features, different config.yaml
```

Your notebook experimentation process is the same. Your pipeline.sh is the same. Your Dockerfile is the same. Only the data source and feature engineering change.

This is why the structure matters — you are building reusable muscle memory, not one-off scripts.

---

## Quick Reference Card

| Task | Command |
|---|---|
| Activate environment | `source .venv/Scripts/activate` |
| Start MLflow UI | `mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "127.0.0.1:*,localhost:*,host.docker.internal:*"` |
| Start Jupyter | `jupyter notebook` |
| Run pipeline manually | `bash pipeline.sh` |
| Build Docker image | `docker build -f serving/Dockerfile -t weather-forecast:latest .` |
| Run Docker container | `docker run -p 8000:8000 weather-forecast:latest` |
| See running containers | `docker ps` |
| Stop a container | `docker stop <container_id>` |
| Test API health | `curl http://localhost:8000/health` |
| Commit and push | `git add . && git commit -m "message" && git push origin main` |
| See git history | `git log --oneline` |
| See what's changed | `git status` |
| View MLflow UI | `http://127.0.0.1:5000` |
| View API docs | `http://localhost:8000/docs` |
