#!/bin/bash

max_jobs=8
job_count=0
pids=()

# Function to launch a single calculation
run_calculation() {
  local index=$1
  echo "Starting training $index"
  mkdir $index
  (cd $index && python3 ../MNIST_train.py $index > output.txt)
  echo "Finished training $index"
}

# Loop through all calculations
for i in {0..199}; do
  # Launch the calculation in the background and get its PID
  run_calculation "$i" &
  pid=$!
  pids+=("$pid")

  # Increment the job count
  ((job_count++))

  # If the maximum number of concurrent jobs is reached, wait for the first one to finish
  if [[ $job_count -ge $max_jobs ]]; then
    wait "${pids[0]}"
    # Remove the finished job's PID from the list and decrement the job count
    pids=("${pids[@]:1}")
    ((job_count--))
  fi
done

# Wait for any remaining jobs to finish
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "All training completed."

python3 ./binning.py > binningResult.txt

echo "Binning completed."


