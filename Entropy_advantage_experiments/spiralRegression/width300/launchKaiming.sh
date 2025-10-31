#!/bin/bash

max_jobs=4
gpu_count=1

nvidia-cuda-mps-control -d


for wlData in 100 200 500 1000
do

# adam training
mkdir $wlData
mkdir $wlData/adamKaiming
cd $wlData/adamKaiming
cat ../../train.py.template | sed "s/aaaaa/$wlData/g" > ./train.py

job_count=0
pids=()

# Function to launch a single calculation
run_calculation() {
  local index=$1
  echo "Starting training $index"
  mkdir $index
  (cd $index && CUDA_VISIBLE_DEVICES=$((index%gpu_count)) python3 ../train.py $index > output.txt)
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

echo "Training completed for" $i
python3 ../../binning.py
echo "Binning completed."

cd ../..
done
