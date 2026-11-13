#!/usr/bin/env bash
echo "Starting Medanta Hospital - Cash & Service Counter Operations Monitor on http://localhost:8195..."
cd "$(dirname "$0")/backend"
python3 counter_tracker.py
