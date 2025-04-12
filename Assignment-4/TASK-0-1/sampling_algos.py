import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# You can import any other torch modules you need below #
from get_results import EnergyRegressor, FEAT_DIM
from sklearn.manifold import TSNE
from time import time


##########################################################

# Other settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Set random seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Define two classes for Algo-1 and Algo-2 ---
##################################################
# Your code for Task-1 goes here

@torch.no_grad()
def gen_tsne_plot_sklearn(samples, dim=2, algo=1):    
    if dim not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3")
    
    samples_tensor = torch.cat(samples, dim=0).cpu().numpy()
    
    tsne = TSNE(n_components=dim, random_state=SEED)
    samples_nd = tsne.fit_transform(samples_tensor)
    
    if dim == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(samples_nd[:, 0], samples_nd[:, 1], s=5)
        plt.title('t-SNE visualization of samples (2D)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.savefig(f'tsne_plot_sklearn_2d_{algo}.png')
    else:  # dim == 3
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples_nd[:, 0], samples_nd[:, 1], samples_nd[:, 2], s=5)
        ax.set_title('t-SNE visualization of samples (3D)')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.savefig(f'tsne_plot_sklearn_3d_{algo}.png')

class Algo1_Sampler:
    def __init__(self, model, N=10000, burn_in=2000, tau=0.01):
        self.model = model
        self.N = N
        self.burn_in = burn_in
        self.tau = torch.tensor(tau)

    def sample(self):
        X0 = torch.randn(1, FEAT_DIM).to(DEVICE)
        samples = [X0]

        start_time = time()
        for i in range(self.N):
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.N}")
            X_prev = samples[-1].detach().requires_grad_(True)
            energy = self.model(X_prev)
            grad = torch.autograd.grad(energy, X_prev)[0]
            X_prop = X_prev - self.tau/2 * grad + torch.sqrt(self.tau) * torch.randn_like(X_prev)
            energy_prop = self.model(X_prop)
            grad_prop = torch.autograd.grad(energy_prop, X_prop)[0]
            log_q_X_prev_X_prop = -(torch.norm(X_prev - X_prop + grad_prop * self.tau / 2) ** 2) / (4 * self.tau)
            log_q_X_prop_X_prev = -(torch.norm(X_prop - X_prev + grad * self.tau / 2) ** 2) / (4 * self.tau)
            acceptance_ratio = torch.minimum(torch.tensor(1.0), torch.exp(energy - energy_prop + log_q_X_prev_X_prop - log_q_X_prop_X_prev))
            if torch.rand(1).item() < acceptance_ratio:
                samples.append(X_prop)
            else:
                samples.append(X_prev)
            if i == self.burn_in-1:
                end_burn_in = time()
        
        end_time = time()

        print(f"Burn-in time: {end_burn_in - start_time:.2f} seconds")
        print(f"Sampling time: {end_time - end_burn_in:.2f} seconds")
        print(f"Total time: {end_time - start_time:.2f} seconds")

        return samples[self.burn_in+1:]

class Algo2_Sampler:
    def __init__(self, model, N=10000, burn_in=2000, tau=0.01):
        self.model = model
        self.N = N
        self.burn_in = burn_in
        self.tau = torch.tensor(tau)

    def sample(self):
        X0 = torch.randn(1, FEAT_DIM).to(DEVICE)
        samples = [X0]

        start_time = time()

        for i in range(self.N):
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.N}")
            X_prev = samples[-1].detach().requires_grad_(True)
            energy = self.model(X_prev)
            grad = torch.autograd.grad(energy, X_prev)[0]
            X_new = X_prev - self.tau/2 * grad + torch.sqrt(self.tau) * torch.randn_like(X_prev)
            samples.append(X_new)
            if i == self.burn_in-1:
                end_burn_in = time()

        end_time = time()

        print(f"Burn-in time: {end_burn_in - start_time:.2f} seconds")
        print(f"Sampling time: {end_time - end_burn_in:.2f} seconds")
        print(f"Total time: {end_time - start_time:.2f} seconds")
            
        return samples[self.burn_in+1:]


# --- Main Execution ---
if __name__ == "__main__":
    model = EnergyRegressor(FEAT_DIM).to(DEVICE)
    model.load_state_dict(torch.load("../trained_model_weights.pth"))

    N = 10000
    burn_in = 2000
    tau = 0.01

    algo1 = Algo1_Sampler(model, N, burn_in, tau)
    samples = algo1.sample()
    gen_tsne_plot_sklearn(samples, dim=2, algo=1)
    gen_tsne_plot_sklearn(samples, dim=3, algo=1)

    algo2 = Algo2_Sampler(model, N, burn_in, tau)
    samples = algo2.sample()
    gen_tsne_plot_sklearn(samples, dim=2, algo=2)
    gen_tsne_plot_sklearn(samples, dim=3, algo=2)