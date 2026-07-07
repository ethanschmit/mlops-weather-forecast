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