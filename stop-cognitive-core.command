#!/bin/bash
# This script finds and stops the watcher process
pkill -f "consciousness_loop.py"
osascript -e 'display notification "The Cognitive Core has been stopped." with title "NEURON-X"'