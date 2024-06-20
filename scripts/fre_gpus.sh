#!/bin/bash

# Get the PIDs of processes using the GPUs
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)

# Loop through each PID
for pid in $pids; do
  # Check if the process exists
  if ps -p $pid > /dev/null; then
    # Kill the process
    kill -9 $pid
    echo "Killed process $pid"
  else
    echo "Process $pid does not exist"
  fi
done

# Verify
nvidia-smi
