CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample_svdd --dataset moons --n_steps 150 --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --n_samples 20000
CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample_svdd --dataset circles --n_steps 150 --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --n_samples 20000
CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample_svdd --dataset manycircles --n_steps 150 --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --n_samples 20000
CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample_svdd --dataset blobs --n_steps 150 --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --n_samples 20000
CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample_svdd --dataset helix --n_steps 150 --n_dim 3 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --n_samples 20000
