#!/bin/bash

#SBATCH --job-name=matrix_transposition
#SBATCH --output=matrix_transposition.out
#SBATCH --error=matrix_transposition.err
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

./main

