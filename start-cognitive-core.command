#!/bin/bash
# This script starts the watcher in the background using the virtual environment
cd "$(dirname "$0")"
# Check for existing processes and stop them
if pgrep -f "consciousness_loop.py" > /dev/null; then
    echo "Stopping existing Cognitive Core instances..."
    pkill -f "consciousness_loop.py"
    # Wait for them to exit (give them time to consolidate)
    sleep 10
fi

nohup ./.venv/bin/python -u consciousness_loop.py > nohup.out 2>&1 &
osascript -e 'display notification "The Cognitive Core has started." with title "NEURON-X"'