# Databricks notebook source
# DBTITLE 1,Import libraries
import json
import os

from databricks.sdk.runtime import dbutils  # type: ignore
from edna.mlops.mlflow.helper import build_run_source_path, get_exp_run
from edna.mlops.mlflow.helper import create_registered_model
from mlflow.client import MlflowClient
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning

disable_warnings(InsecureRequestWarning)

# COMMAND ----------
# DBTITLE 1,Get parameters
notebook_parameters = dbutils.notebook.entry_point.getCurrentBindings()
config = json.loads(notebook_parameters["json_parameters"])

# COMMAND ----------
# DBTITLE 1,Get `train` task outputs
experiment_run_id = dbutils.jobs.taskValues.get(
    taskKey="train", key="experiment_run_id"
)
experiment_name = dbutils.jobs.taskValues.get(taskKey="train", key="experiment_name")
metric_name = dbutils.jobs.taskValues.get(taskKey="train", key="metric_name")
model_name = f"{os.environ['MLFLOW_NAME_PREFIX']}.{config['model_exp_name']}"

# COMMAND ----------
# DBTITLE 1,Get the best model from the last hyperparameter search
run = get_exp_run(
    experiment_name=experiment_name,
    version="best_metric",
    metric_name=metric_name,
    parent_id=experiment_run_id,
)

# COMMAND ----------
# DBTITLE 1,Register the model
# At least at this time, we will just be always updating our "champion" model with
# the latest version.

# Get the registered model
# client = MlflowClient()
# mdl = client.get_registered_model(model_name)
client = MlflowClient()
mdl = create_registered_model(
    name=model_name,
    description=config["model_description"],
    raise_if_already_exists=False,
    overwrite=False,
    overwrite_existing_tags_and_description=False,
)

# Register the new version of the model
mv = client.create_model_version(
    name=model_name,  # exp name same as model name
    source=build_run_source_path(run),
    run_id=run.info.run_id,
)

try:
    client.set_registered_model_alias(mv.name, "champion", mv.version)
except Exception:
    client.set_model_version_tag(mv.name, mv.version, "alias", "champion")

dbutils.jobs.taskValues.set(key="model_name", value=mdl.name)
dbutils.jobs.taskValues.set(key="model_alias", value="champion")
dbutils.jobs.taskValues.set(key="model_version", value=mv.version)