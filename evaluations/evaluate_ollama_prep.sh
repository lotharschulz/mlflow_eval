#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Check if Ollama is installed, if not install it
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

# Install and start Ollama (if not already done)
ollama pull llama3.2

# Verify it's running
ollama list