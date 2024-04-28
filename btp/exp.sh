#!/bin/bash

# Array of step values
steps=(1 2 4 8 16 32)

# Array of models
models=("mistral" "llama2" "llama2chat")

# Array of optimizers
opts=("Adam" "SGD")

# Nested loop over the arrays
for model in "${models[@]}"
do
   for opt in "${opts[@]}"
   do
      for step in "${steps[@]}"
      do
         # Run the experiment with the current model, optimizer, and step value
         python run.py --model $model --lr 0.00001 --steps $step --opt $opt
      done
   done
done