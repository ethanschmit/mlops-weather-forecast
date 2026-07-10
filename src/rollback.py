# src/rollback.py
# Lists all registered model versions with tags + who holds "champion", or
# rolls back by reassigning the champion alias to a specific version.
# Run: python src/rollback.py               (lists versions)
#      python src/rollback.py --version 4   (makes v4 champion)

import argparse
import yaml
import mlflow
from mlflow.tracking import MlflowClient


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def list_versions(client: MlflowClient, model_name: str):
    try:
        champion_version = client.get_model_version_by_alias(model_name, "champion").version
    except mlflow.exceptions.MlflowException:
        champion_version = None

    versions = sorted(client.search_model_versions(f"name='{model_name}'"), key=lambda v: int(v.version))
    print(f"{'Version':<8}{'Alias':<12}{'CV_MAE':<10}Run ID")
    for v in versions:
        mae = client.get_run(v.run_id).data.metrics.get("cv_mae", float("nan"))
        alias = "champion" if v.version == champion_version else ""
        print(f"{v.version:<8}{alias:<12}{mae:<10.3f}{v.run_id}")


def rollback(client: MlflowClient, model_name: str, target_version: str):
    try:
        current = client.get_model_version_by_alias(model_name, "champion")
        client.set_model_version_tag(model_name, current.version, "status", "archived")
        print(f"Archived current champion v{current.version}")
    except mlflow.exceptions.MlflowException:
        pass

    client.set_registered_model_alias(model_name, "champion", target_version)
    client.set_model_version_tag(model_name, target_version, "status", "champion")
    print(f"PROMOTED: v{target_version} → champion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=False, help="Version to roll back to. Omit to list versions.")
    args = parser.parse_args()

    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()
    model_name = config["model"]["name"]

    if args.version:
        rollback(client, model_name, args.version)
    else:
        list_versions(client, model_name)