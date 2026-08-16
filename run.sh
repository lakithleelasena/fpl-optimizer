#!/bin/bash
set -e

cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Install/update dependencies using venv's pip directly
echo "Installing dependencies..."
.venv/bin/pip install -r requirements.txt -q

# Start the app
echo "Starting FPL Optimizer at http://localhost:8000"
.venv/bin/uvicorn main:app --reload --port 8000
