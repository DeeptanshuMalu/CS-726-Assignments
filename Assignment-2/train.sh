CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode train --dataset moons --n_steps 2000 --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 10 --batch_size 5 --lr 0.00001 
