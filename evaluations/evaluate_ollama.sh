#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Make sure MLFLOW_TRACKING_URI is set
echo "MLFLOW_TRACKING_URI is set to $MLFLOW_TRACKING_URI"
echo "OLLAMA_BASE_URI is set to $OLLAMA_BASE_URI"

python evaluate_ollama.py