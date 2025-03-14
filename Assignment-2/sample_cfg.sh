GUIDANCE_SCALES=(0.25 0.5 1 2 4)

for scale in "${GUIDANCE_SCALES[@]}"; do
    echo "Running with guidance scale=$scale"
    
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample_cfg --dataset moons --n_steps 150 --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --p_uncond 0.2 --n_samples 20000 --guidance_scale "$scale"
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample_cfg --dataset circles --n_steps 150 --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --p_uncond 0.2 --n_samples 20000 --guidance_scale "$scale"
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample_cfg --dataset manycircles --n_steps 150 --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --p_uncond 0.2 --n_samples 20000 --guidance_scale "$scale"
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample_cfg --dataset blobs --n_steps 150 --n_dim 2 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --p_uncond 0.2 --n_samples 20000 --guidance_scale "$scale"
    CUDA_VISIBLE_DEVICES=1 python ddpm.py --mode sample_cfg --dataset helix --n_steps 150 --n_dim 3 --lbeta 0.0001 --ubeta 0.02 --epochs 100 --batch_size 64 --lr 0.001 --p_uncond 0.2 --n_samples 20000 --guidance_scale "$scale"
done
