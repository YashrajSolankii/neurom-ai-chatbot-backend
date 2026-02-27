#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status.
set -o errexit

echo "Render Start Script: Installing dependencies..."
pip install -r requirements.txt

# This command runs the one-time database creation if the directory doesn't exist.
# On Render, the file system is ephemeral, so this might run on every deploy/restart
# if the persisted DB is not part of the deployed files. Including it in git is key.
if [ ! -d "chroma_db_api_neuroum" ]; then
    echo "Chroma DB not found. Running core_logic.py to create it..."
    python core_logic.py
else
    echo "Chroma DB found. Skipping creation."
fi

echo "Starting Uvicorn server..."
# Render provides the $PORT environment variable.
# The --host 0.0.0.0 is crucial to bind to the network interface inside the container.
uvicorn main_api:app --host 0.0.0.0 --port $PORT
