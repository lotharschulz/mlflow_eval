#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Detected OS: $OS"

# Make sure MLFLOW_TRACKING_URI is set
export MLFLOW_TRACKING_URI="http://localhost:5000"
echo "MLFLOW_TRACKING_URI is set to $MLFLOW_TRACKING_URI"

# Check if Ollama latest version is installed, if not install it
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing Ollama..."
    if [ "$OS" == "mac" ]; then
        # On macOS, use Homebrew or download the app
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "Please install Ollama manually from https://ollama.com/download/mac"
            exit 1
        fi
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
else
    echo "Ollama is already installed. Checking version..."
    
    if [ "$OS" == "mac" ]; then
        # macOS-specific version extraction using sed
        CURRENT_VERSION=$(ollama --version 2>/dev/null | sed -n 's/.*version is \([0-9.]*\).*/\1/p' | head -n 1)
        # If empty, default to 0.0.0
        CURRENT_VERSION=${CURRENT_VERSION:-0.0.0}
        echo "Current version: $CURRENT_VERSION"
        
        # Get latest version from GitHub API using sed
        LATEST_VERSION=$(curl -s https://api.github.com/repos/ollama/ollama/releases/latest | sed -n 's/.*"tag_name": "v\([0-9.]*\)".*/\1/p' || echo "999.999.999")
        echo "Latest version: $LATEST_VERSION"
        
        # Compare versions and update if needed
        if [ "$CURRENT_VERSION" != "$LATEST_VERSION" ]; then
            echo "Updating Ollama from $CURRENT_VERSION to $LATEST_VERSION..."
            if command -v brew &> /dev/null && brew list ollama &> /dev/null; then
                # Ollama is installed via Homebrew
                brew upgrade ollama
            else
                # Ollama is installed as standalone app - download and install the update
                echo "Downloading Ollama update..."
                TEMP_ZIP="/tmp/Ollama-darwin.zip"
                curl -L -o "$TEMP_ZIP" "https://ollama.com/download/Ollama-darwin.zip"

                echo "Installing update..."
                # Extract to /Applications
                unzip -o "$TEMP_ZIP" -d /Applications

                # Clean up
                rm "$TEMP_ZIP"
                
                echo "Ollama has been updated to $LATEST_VERSION"
            fi
        else
            echo "Ollama is up to date."
        fi
    else
        # Linux-specific version extraction using grep -P
        CURRENT_VERSION=$(ollama --version | grep -oP 'ollama version is \K[0-9.]+' || echo "0.0.0")
        echo "Current version: $CURRENT_VERSION"
        
        # Get latest version from GitHub API using grep -P
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
fi

# pull models
echo "Pulling Ollama models..."

#echo "pull llama3.3:70b ------------------------------------------------------"
#ollama pull llama3.3:70b

echo "pull llama3.2:3b ------------------------------------------------------"
ollama pull llama3.2:3b

echo "pull llama3.2:1b ------------------------------------------------------"
ollama pull llama3.2:1b

echo "pull mistral:7b ------------------------------------------------------"
ollama pull mistral:7b

echo "pull dolphin3:8b ------------------------------------------------------"
ollama pull dolphin3:8b

echo "pull deepseek-r1:7b ------------------------------------------------------"
ollama pull deepseek-r1:7b

#echo "pull gemma3:27b ------------------------------------------------------"
#ollama pull gemma3:27b

echo "pull nomic-embed-text:v1.5 ------------------------------------------------------"
ollama pull nomic-embed-text:v1.5

#echo "pull llama4:16x17b ------------------------------------------------------"
#ollama pull llama4:16x17b

# Verify it's running
ollama list