#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Make sure MLFLOW_TRACKING_URI is set
echo "MLFLOW_TRACKING_URI is set to $MLFLOW_TRACKING_URI"

# Or in Python
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Make sure MLFLOW_TRACKING_URI is set
echo "OLLAMA_BASE_URI is set to $OLLAMA_BASE_URI"

# Or in Python
python -c "import os; print(os.getenv('OLLAMA_BASE_URI'))"
