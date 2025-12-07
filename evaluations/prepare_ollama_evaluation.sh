#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Make sure MLFLOW_TRACKING_URI is set
export MLFLOW_TRACKING_URI="http://localhost:5000"
echo "MLFLOW_TRACKING_URI is set to $MLFLOW_TRACKING_URI"

# Check if Ollama latest version is installed, if not install it
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed. Checking version..."
    
    # Get current installed version
    CURRENT_VERSION=$(ollama --version | grep -oP 'ollama version is \K[0-9.]+' || echo "0.0.0")
    echo "Current version: $CURRENT_VERSION"
    
    # Get latest version from GitHub API
    LATEST_VERSION=$(curl -s https://api.github.com/repos/ollama/ollama/releases/latest | grep -oP '"tag_name": "v\K[0-9.]+' || echo "999.999.999")
    echo "Latest version: $LATEST_VERSION"
    
    # Compare versions and update if needed
    if [ "$CURRENT_VERSION" != "$LATEST_VERSION" ]; then
        echo "Updating Ollama from $CURRENT_VERSION to $LATEST_VERSION..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "Ollama is up to date."
    fi
fi

# pull models
#ollama pull llama4
#ollama pull llama3.3
ollama pull llama3.2
#ollama pull mistral
#ollama pull dolphin3
#ollama pull deepseek-r1
#ollama pull deepseek-v3.1


# Verify it's running
ollama list