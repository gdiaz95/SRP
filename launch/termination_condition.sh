#!/bin/bash
set -e
cd /home/gabriel/dpth_copy
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepthinking
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7
combinations=$(python /home/gabriel/main.py)

echo "Running test_model.py with num_chunks=1 and overlap=0"
python test_model.py problem.numchunks=1 problem.overlap=0

IFS=$'\n'  # Set the internal field separator to newline
for combo in $combinations; do
  IFS=',' read -r num_chunks overlap <<< "$combo"
  echo "Running test_model.py with num_chunks=$num_chunks and overlap=$overlap"
  python test_model.py problem.numchunks="$num_chunks" problem.overlap="$overlap"
done
