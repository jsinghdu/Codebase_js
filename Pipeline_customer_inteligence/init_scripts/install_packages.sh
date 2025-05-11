#!/bin/bash

/databricks/python/bin/pip install edna.mlops --upgrade
/databricks/python/bin/pip install databricks-sdk --upgrade
/databricks/python/bin/pip install \
  rich \
  "pandas>=2.0" \
  mlflow

echo "Appending ETC_EXTRA_HOSTS to /etc/hosts"
echo "" >> /etc/hosts
echo "# extra hosts from private endpoints" >> /etc/hosts
echo "" >> /etc/hosts
echo "$ETC_EXTRA_HOSTS" >> /etc/hosts

echo "Printing out /etc/hosts"
cat /etc/hosts
