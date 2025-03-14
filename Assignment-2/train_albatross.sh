CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode train --dataset albatross --n_steps 150 --n_dim 64 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001
