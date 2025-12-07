#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Make sure MLFLOW_TRACKING_URI is set
export MLFLOW_TRACKING_URI="http://localhost:5000"
echo "MLFLOW_TRACKING_URI is set to $MLFLOW_TRACKING_URI"

python evaluate_ollama.py