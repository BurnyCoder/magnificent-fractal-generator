#!/bin/bash

# Magnificent Fractal Art Generator launcher script

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if virtual environment exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please install python3-venv package and try again."
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/installed.flag" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        touch venv/installed.flag
    else
        echo "Failed to install requirements."
        exit 1
    fi
fi

# Create directories if they don't exist
mkdir -p static/saved_fractals

# Run the application
echo "Starting Magnificent Fractal Art Generator..."
python app.py

# Deactivate virtual environment on exit
deactivate 