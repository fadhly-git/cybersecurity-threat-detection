#!/bin/bash
# install.sh - Fixed for Python 3.11

echo "=========================================="
echo "CYBERSECURITY ML - macOS Installation"
echo "=========================================="

# Check if Python 3.11 exists
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "Using Python 3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo "Using Python 3.10"
else
    echo "ERROR: Python 3.10 or 3.11 required!"
    echo "Install with: brew install python@3.11"
    exit 1
fi

$PYTHON_CMD --version

# Create virtual environment with specific Python
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate
echo "Activating environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo "=========================================="
echo "Installation complete!"
echo ""
echo "To activate: source venv/bin/activate"
echo "To run: python main.py"
echo "=========================================="