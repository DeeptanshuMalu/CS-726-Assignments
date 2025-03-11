#!/bin/bash

# List of n_steps values
STEPS_VALUES=(10 50 100 150 200 500)

# List of lbeta,ubeta pairs
BETA_PAIRS=(
    "0.0001 0.02"
    "0.001 0.2"
    "0.00001 0.002"
    "0.00001 0.02"
    "0.0001 0.2"
    "0.00001 0.2"
)

# First loop: Fixed beta pair (0.0001, 0.02) with varying T values
echo "Running with fixed beta pair (0.0001, 0.02) and varying T values"
for steps in "${STEPS_VALUES[@]}"; do
    echo "Running with n_steps=$steps, lbeta=0.0001, ubeta=0.02"
    
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset moons --n_steps "$steps" --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --n_samples 5000
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset circles --n_steps "$steps" --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --n_samples 5000
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset manycircles --n_steps "$steps" --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --n_samples 5000
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset blobs --n_steps "$steps" --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --n_samples 5000
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset helix --n_steps "$steps" --n_dim 3 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --n_samples 5000
done

# Second loop: Fixed T=200 with varying beta pairs
echo "Running with fixed n_steps=200 and varying beta pairs"
for beta_pair in "${BETA_PAIRS[@]}"; do
    read -r lbeta ubeta <<< "$beta_pair"
    
    echo "Running with n_steps=200, lbeta=$lbeta, ubeta=$ubeta"
    
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset moons --n_steps 200 --n_dim 2 --lbeta "$lbeta" --ubeta "$ubeta" --epochs 100 --batch_size 64 --lr 0.001 --n_samples 5000
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset circles --n_steps 200 --n_dim 2 --lbeta "$lbeta" --ubeta "$ubeta" --epochs 100 --batch_size 64 --lr 0.001 --n_samples 5000
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset manycircles --n_steps 200 --n_dim 2 --lbeta "$lbeta" --ubeta "$ubeta" --epochs 100 --batch_size 64 --lr 0.001 --n_samples 5000
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset blobs --n_steps 200 --n_dim 2 --lbeta "$lbeta" --ubeta "$ubeta" --epochs 100 --batch_size 64 --lr 0.001 --n_samples 5000
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset helix --n_steps 200 --n_dim 3 --lbeta "$lbeta" --ubeta "$ubeta" --epochs 100 --batch_size 64 --lr 0.001 --n_samples 5000
done