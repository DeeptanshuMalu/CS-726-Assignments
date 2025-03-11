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

# Fixed beta pair with varying steps
echo "Running with fixed beta pair (lbeta=0.0001, ubeta=0.02) and varying n_steps"
for steps in "${STEPS_VALUES[@]}"; do
    echo "Running with n_steps=$steps, lbeta=0.0001, ubeta=0.02"
    
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode train --dataset moons --n_steps "$steps" --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode train --dataset circles --n_steps "$steps" --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode train --dataset manycircles --n_steps "$steps" --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode train --dataset blobs --n_steps "$steps" --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode train --dataset helix --n_steps "$steps" --n_dim 3 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001
done

# Fixed n_steps with varying beta pairs
echo "Running with fixed n_steps=200 and varying beta pairs"
for beta_pair in "${BETA_PAIRS[@]}"; do
    read -r lbeta ubeta <<< "$beta_pair"
    
    echo "Running with n_steps=200, lbeta=$lbeta, ubeta=$ubeta"
    
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode train --dataset moons --n_steps 200 --n_dim 2 --lbeta "$lbeta" --ubeta "$ubeta" --epochs 100 --batch_size 64 --lr 0.001
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode train --dataset circles --n_steps 200 --n_dim 2 --lbeta "$lbeta" --ubeta "$ubeta" --epochs 100 --batch_size 64 --lr 0.001
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode train --dataset manycircles --n_steps 200 --n_dim 2 --lbeta "$lbeta" --ubeta "$ubeta" --epochs 100 --batch_size 64 --lr 0.001
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode train --dataset blobs --n_steps 200 --n_dim 2 --lbeta "$lbeta" --ubeta "$ubeta" --epochs 100 --batch_size 64 --lr 0.001
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode train --dataset helix --n_steps 200 --n_dim 3 --lbeta "$lbeta" --ubeta "$ubeta" --epochs 100 --batch_size 64 --lr 0.001
done
