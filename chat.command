#!/bin/bash
# This script runs the main tool using the virtual environment
cd "$(dirname "$0")"
echo ""
echo "=========================================="
echo "booting up neuron-x. Please be patient..."
echo "=========================================="
echo ""
./.venv/bin/python gemini_interface.py "$@"
echo ""
echo "=========================================="
echo "neuron-x has been shut down."
echo "=========================================="
echo ""
