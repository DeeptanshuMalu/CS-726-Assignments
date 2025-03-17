import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt
import numpy as np
from ddpm import NoiseScheduler, DDPM

@torch.no_grad()
def sample_albatross(model, n_samples, noise_scheduler, priors):
    init_sample = priors
    T = len(noise_scheduler)

    for t in range(T-1, -1, -1):
        t_batch = torch.ones(n_samples).to(device)*t
        alpha_t = 1-noise_scheduler.betas.to(device)[t].view(-1, 1)
        alpha_bar_t = noise_scheduler.alphas.to(device)[t].view(-1, 1)
        t_batch = t_batch.float().reshape(-1, 1)
        init_sample = 1/torch.sqrt(alpha_t) * (init_sample - (1-alpha_t)/torch.sqrt(1-alpha_bar_t) * model(init_sample, t_batch))

    return init_sample


if __name__ == "__main__":
    seed=42

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    utils.seed_everything(seed)

    n_dim = 64
    n_steps = 150
    lbeta = 0.0001
    ubeta = 0.02
    dataset_name = "albatross"
    run_name = f"exps/ddpm_{n_dim}_{n_steps}_{lbeta}_{ubeta}_{dataset_name}"

    priors = torch.from_numpy(np.load('data/albatross_prior_samples.npy')).to(torch.float32).to(device)
    print(priors.shape)

    n_samples = priors.shape[0]

    model = DDPM(n_dim=n_dim, n_steps=n_steps).to(device)
    model.load_state_dict(torch.load(f'albatross_model.pth', map_location=device))

    noise_scheduler = NoiseScheduler(num_timesteps=n_steps, beta_start=lbeta, beta_end=ubeta)

    data_X, data_y = dataset.load_dataset(dataset_name)
    # can split the data into train and test -- for evaluation later
    data_X = data_X.to(device)

    samples = sample_albatross(model, n_samples, noise_scheduler, priors)

    subsample_size = 250
    emd_list = []
    # nll_list = []
    for i in range(5):
        print(f"Sample {i}")
        subsample_data_X = utils.sample(data_X.to("cpu").numpy(), size = subsample_size)
        subsample_samples = utils.sample(samples.to("cpu").numpy(), size = subsample_size)
        emd = utils.get_emd(subsample_data_X, subsample_samples)
    #     nll = utils.get_nll(data_X, samples)
        print(f'{i} EMD {emd}')
    #     print(f'{i} NLL {nll}')
        emd_list.append(emd)
    #     nll_list.append(nll)
    emd_avg = sum(emd_list)/len(emd_list)
    # nll_avg = sum(nll_list)/len(nll_list)
    print(f"EMD: {emd_avg}")
    # print(f"NLL: {nll_avg}")


    np.save(f'albatross_samples_reproduce.npy', samples.cpu().numpy())

    # torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')