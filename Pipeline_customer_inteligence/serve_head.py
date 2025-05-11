# Databricks notebook source
# DBTITLE 1,Import libraries
import json

from databricks.sdk.runtime import dbutils  # type: ignore
from edna.mlops.databricks import serve_model
from edna.mlops.mlflow.clone_model import clone_model

# from edna.mlops.mlflow.clone_model import clone_model
from mlflow.tracking import MlflowClient

notebook_parameters = dbutils.notebook.entry_point.getCurrentBindings()
config = json.loads(notebook_parameters["json_parameters"])

# COMMAND ----------

# DBTITLE 1,Load model name from last task
model_name = dbutils.jobs.taskValues.get(taskKey="register", key="model_name")
model_alias = dbutils.jobs.taskValues.get(taskKey="register", key="model_alias")

# COMMAND ----------

# DBTITLE 1,Clone the model from the MLOps registry to the local databricks registry
mlops_client = MlflowClient()
dbx_client = MlflowClient("databricks")
model_versions = clone_model(mlops_client, dbx_client, model_name, False)

# COMMAND ----------

# DBTITLE 1, Serve the model
model_version = model_versions[model_alias].version
serve_model(
    endpoint_name=model_name.replace(".", "-").replace("|", "-"),
    served_entities=[{
        "entity_name": model_versions[model_alias].name,
        "entity_version": model_version,
        "name": f"primary-{model_version}",
        "workload_size": config["serve_endpoint_size"],
        "scale_to_zero_enabled": False,
    }],
    traffic_config={
        "routes": [{
            "served_model_name": f"primary-{model_version}",
            "traffic_percentage": 100,
        }]
    },
)
