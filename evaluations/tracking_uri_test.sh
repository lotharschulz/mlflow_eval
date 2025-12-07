#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Check the variable is set
echo $MLFLOW_TRACKING_URI

# Or in Python
python -c "import mlflow; print(mlflow.get_tracking_uri())"