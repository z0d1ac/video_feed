#!/bin/bash

# Enable job control
set -m

# Start Main Flask Application (Port 5050)
echo "Starting Video Feed App..."
python app.py

# Wait for any process to exit
wait -n
  
# Exit with status of process that exited first
exit $?
