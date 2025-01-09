#!/bin/bash

# Set up a virtual environment if one doesn't exist
if [ ! -d "/home/site/wwwroot/antenv" ]; then
    python3 -m venv /home/site/wwwroot/antenv
fi

# Activate the virtual environment
source /home/site/wwwroot/antenv/bin/activate

# Install the required dependencies
pip install --upgrade pip
pip install -r /home/site/wwwroot/requirements.txt

# Run Streamlit on port 8505
streamlit run /home/site/wwwroot/RSS-ModelCompared.py --server.port=8505 --server.address=0.0.0.0
