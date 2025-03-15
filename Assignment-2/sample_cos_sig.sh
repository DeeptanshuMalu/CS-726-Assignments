#!/bin/bash

echo "Sampling DDPM on synthetic datasets with cosine scheduler"
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode sample --dataset moons --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler cosine --n_samples 5000
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode sample --dataset circles --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler cosine --n_samples 5000
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode sample --dataset manycircles --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler cosine --n_samples 5000
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode sample --dataset blobs --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler cosine --n_samples 5000
CUDA_VISIBLE_DEVICES=3 python ddpm.py --mode sample --dataset helix --n_steps 200 --n_dim 3 --epochs 100 --batch_size 64 --lr 0.001 --scheduler cosine --n_samples 5000

echo "Sampling DDPM on synthetic datasets with sigmoid scheduler"
CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset moons --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler sigmoid --n_samples 5000
CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset circles --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler sigmoid --n_samples 5000
CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset manycircles --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler sigmoid --n_samples 5000
CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset blobs --n_steps 200 --n_dim 2 --epochs 100 --batch_size 64 --lr 0.001 --scheduler sigmoid --n_samples 5000
CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample --dataset helix --n_steps 200 --n_dim 3 --epochs 100 --batch_size 64 --lr 0.001 --scheduler sigmoid --n_samples 5000