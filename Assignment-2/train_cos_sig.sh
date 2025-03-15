#!/bin/bash

echo "Training DDPM on synthetic datasets with cosine scheduler"
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode train --dataset moons --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler cosine
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode train --dataset circles --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler cosine
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode train --dataset manycircles --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler cosine
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode train --dataset blobs --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler cosine
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode train --dataset helix --n_steps 200 --n_dim 3 --epochs 100 --batch_size 64 --lr 0.001 --scheduler cosine

echo "Training DDPM on synthetic datasets with sigmoid scheduler"
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode train --dataset moons --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler sigmoid
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode train --dataset circles --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler sigmoid
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode train --dataset manycircles --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler sigmoid
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode train --dataset blobs --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler sigmoid
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode train --dataset helix --n_steps 200 --n_dim 3 --epochs 100 --batch_size 64 --lr 0.001 --scheduler sigmoid