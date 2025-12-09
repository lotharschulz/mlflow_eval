#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make sure MLFLOW_TRACKING_URI is set
echo "MLFLOW_TRACKING_URI is set to $MLFLOW_TRACKING_URI"
echo "OLLAMA_BASE_URI is set to $OLLAMA_BASE_URI"

# Run the Python script from the same directory as this script
python "$SCRIPT_DIR/evaluate_ollama.py"