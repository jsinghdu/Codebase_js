# Databricks notebook source
# DBTITLE 1,Import libraries
import json
import os

from databricks.sdk.runtime import dbutils  # type: ignore
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning

from src.train import train

disable_warnings(InsecureRequestWarning)

# COMMAND ----------
# DBTITLE 1,Get parameters
notebook_parameters = dbutils.notebook.entry_point.getCurrentBindings()
config = json.loads(notebook_parameters["json_parameters"])


# COMMAND ----------
# DBTITLE 1,Train the model
# Run train function
experiment_run_id, experiment_name, metric_name = train(
    poly_order_range=config["polynomial_order_range"]
)

print(f"Experiment run ID: {experiment_run_id}")

# COMMAND ----------
# DBTITLE 1,Return the experiment run ID
dbutils.jobs.taskValues.set(key="experiment_run_id", value=experiment_run_id)
dbutils.jobs.taskValues.set(key="experiment_name", value=experiment_name)
dbutils.jobs.taskValues.set(key="metric_name", value=metric_name)
