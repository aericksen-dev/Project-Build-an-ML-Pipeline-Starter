import json

import mlflow
import hydra
import tempfile, json, os

from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]

# This automatically reads in the configuration
@hydra.main(config_path=".", config_name="config", version_base="1.1")
def go(config: DictConfig):

    proj_root = get_original_cwd()

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(proj_root, "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Cleaned sample of Airbnb NYC data",
                    "min_price": config["basic_cleaning"]["min_price"],
                    "max_price": config["basic_cleaning"]["max_price"],
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(proj_root, "src", "data_check"),
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["basic_cleaning"]["min_price"],
                    "max_price": config["basic_cleaning"]["max_price"],
                },
            )

        if "data_split" in active_steps:
            comp_repo = config["main"]["components_repository"]
            # if it's a local folder like "components", make absolute (if you implemented proj_root earlier)
            if not (comp_repo.startswith("http") or comp_repo.endswith(".git")):
                comp_repo = os.path.join(proj_root, comp_repo)

            _ = mlflow.run(
                f"{comp_repo}/train_val_test_split",
                "main",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                },
            )

        if "train_random_forest" in active_steps:
            step_path = os.path.join(proj_root, "src", "train_random_forest")

            # Convert Hydra DictConfig -> plain dict -> JSON file for the step
            rf_cfg_dict = OmegaConf.to_container(config["modeling"]["random_forest"], resolve=True)

            with tempfile.TemporaryDirectory() as td:
                rf_cfg_path = os.path.join(td, "rf_config.json")
                with open(rf_cfg_path, "w") as f:
                    json.dump(rf_cfg_dict, f)

                _ = mlflow.run(
                    step_path,
                    "main",
                    parameters={
                        # required by the step’s MLproject
                        "trainval_artifact": "trainval_data.csv:latest",
                        "val_size": config["modeling"]["val_size"],
                        "random_seed": config["modeling"]["random_seed"],
                        "stratify_by": config["modeling"]["stratify_by"],
                        "rf_config": rf_cfg_path,  # <— pass JSON file path
                        "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                        "output_artifact": "random_forest_export",  # <— rubric wants this name
                    },
                )

        if "test_regression_model" in active_steps:
            comp_repo = config["main"]["components_repository"]
            if not (comp_repo.startswith("http") or comp_repo.endswith(".git")):
                comp_repo = os.path.join(proj_root, comp_repo)

            _ = mlflow.run(
                f"{comp_repo}/test_regression_model",
                "main",
                parameters={
                    "mlflow_model": "model_export:prod",  # or model_export:prod if that's your name
                    "test_dataset": "test_data.csv:latest",  # <-- correct param name
                },
            )


if __name__ == "__main__":
    go()
