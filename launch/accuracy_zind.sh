#!/bin/bash
# conda activate deepthinking
cd ..

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null
then
    export CUDA_VISIBLE_DEVICES=0
    echo "CUDA is available. Using First GPU available."
else
    echo "CUDA is not available. Running on CPU."
fi

for i in {5..7}
do
    echo "Accuracy for $i terminals."
    python test_model.py time_evaluation=False accuracy_evaluation=True problem.data_type=zind/big_floorplans/${i}_green
done

echo "Done taking accuracy for all terminals."