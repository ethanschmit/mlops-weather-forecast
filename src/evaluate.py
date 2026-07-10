# src/evaluate.py
# Compares new model against the current "champion" in the registry.
# Promotes by reassigning the champion alias if better, rejects if not.
# Run: python src/evaluate.py --run_id <id> --mae <value>
# Exit code: 0 = promoted, 1 = rejected (pipeline.sh treats both as valid outcomes)

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

    # Gate 1: absolute quality
    if new_mae > threshold:
        print(f"REJECTED: CV MAE {new_mae:.3f} > threshold {threshold}")
        return False

    # Gate 2: compare against whatever version currently holds "champion"
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
        champion_mae = float(client.get_run(champion.run_id).data.metrics["cv_mae"])

        required_mae = champion_mae * (1 - min_improv)
        if new_mae > required_mae:
            print(f"REJECTED: {new_mae:.3f} not enough better than champion {champion_mae:.3f} "
                  f"(needed < {required_mae:.3f})")
            return False

        client.set_model_version_tag(model_name, champion.version, "status", "archived")
        print(f"Archived previous champion v{champion.version} (MAE: {champion_mae:.3f})")

    except mlflow.exceptions.MlflowException as e:
        print(f"No existing champion ({e}) — promoting directly")

    # The version train.py just registered is always the highest version number
    latest = max(client.search_model_versions(f"name='{model_name}'"), key=lambda v: int(v.version))

    client.set_registered_model_alias(model_name, "champion", latest.version)
    client.set_model_version_tag(model_name, latest.version, "status", "champion")
    print(f"PROMOTED: v{latest.version} → champion (MAE: {new_mae:.3f})")
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