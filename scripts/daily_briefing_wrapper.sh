#!/bin/bash
cd /Users/corneliusbolten/Projekte/neuron-x
export $(grep -v '^#' .env | xargs)
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 scripts/daily_briefing_trigger.py >> daily_briefing.log 2>&1
