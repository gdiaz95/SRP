#!/bin/bash
conda activate deepthinking
cd ..

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null
then
    export CUDA_VISIBLE_DEVICES=0
    echo "CUDA is available. Using First GPU available."
else
    echo "CUDA is not available. Running on CPU."
fi

for i in {2..8}
do
    echo "Taking time for $i terminals."
    python test_model.py time_evaluation=True problem.data_type=${i}_green
done

echo "Done taking time for all terminals."