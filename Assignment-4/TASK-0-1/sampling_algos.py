import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# You can import any other torch modules you need below #
from get_results import EnergyRegressor, FEAT_DIM
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
def gen_tsne_plot(samples, dim=2, algo=1):
    if dim not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3")
        
    # Convert list of tensors to a single tensor and move to CPU
    samples_tensor = torch.cat(samples, dim=0).cpu().numpy()
    
    # Custom t-SNE implementation
    def compute_pairwise_distances(X):
        sum_X = np.sum(np.square(X), axis=1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return D
    
    def compute_joint_probabilities(D, perplexity=30.0, epsilon=1e-8):
        # Calculate joint probabilities p_ij from distances
        n = D.shape[0]
        P = np.zeros((n, n))
        
        # Binary search for sigma (bandwidth)
        for i in range(n):
            beta_min, beta_max = -np.inf, np.inf
            beta = 1.0
            
            # Binary search for beta
            for _ in range(50):
                # Compute conditional probabilities with current beta
                D_i = D[i, np.arange(n) != i]
                P_i = np.exp(-D_i * beta)
                sum_P_i = np.sum(P_i)
                
                if sum_P_i < epsilon:
                    beta_min = beta
                    beta = (beta + beta_max) / 2.0 if beta_max != np.inf else beta * 2
                else:
                    H_i = np.log(sum_P_i) + beta * np.sum(D_i * P_i) / sum_P_i
                    H_diff = H_i - np.log(perplexity)
                    
                    if np.abs(H_diff) < 1e-5:
                        break
                    
                    if H_diff > 0:
                        beta_min = beta
                        beta = (beta + beta_max) / 2.0 if beta_max != np.inf else beta * 2
                    else:
                        beta_max = beta
                        beta = (beta + beta_min) / 2.0 if beta_min != -np.inf else beta / 2
                
                P_i = P_i / sum_P_i
                P[i, np.arange(n) != i] = P_i
        
        # Symmetrize the probabilities
        P = (P + P.T) / (2*n)
        P = np.maximum(P, epsilon)
        return P
    
    def custom_tsne(X, n_components=2, perplexity=30.0, n_iter=1000, learning_rate=200.0, random_state=None):
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = X.shape[0]
        
        # Initialize low-dimensional representation
        Y = np.random.randn(n_samples, n_components) * 0.0001
        
        # Compute pairwise distances and joint probabilities
        D = compute_pairwise_distances(X)
        P = compute_joint_probabilities(D, perplexity)
        
        # Gradient descent
        for i in range(n_iter):
            # Compute pairwise affinities in low-dimensional space
            sum_Y = np.sum(np.square(Y), axis=1)
            num = -2.0 * np.dot(Y, Y.T)
            num = 1.0 / (1.0 + np.add(np.add(num, sum_Y).T, sum_Y))
            np.fill_diagonal(num, 0)
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)
            
            # Compute gradient
            PQ_diff = P - Q
            grad = np.zeros((n_samples, n_components))
            for j in range(n_samples):
                grad[j] = 4.0 * np.sum(np.tile(PQ_diff[:, j] * num[:, j], (n_components, 1)).T * (Y[j] - Y), axis=0)
            
            # Update Y
            Y = Y - learning_rate * grad
            
            # Center Y
            Y = Y - np.mean(Y, axis=0)
            
            if (i + 1) % 100 == 0:
                print(f"t-SNE iteration {i + 1}/{n_iter}")
        
        return Y
    
    # Apply custom t-SNE
    samples_nd = custom_tsne(samples_tensor, n_components=dim, perplexity=30, 
                             n_iter=1000, random_state=SEED)
    
    # Plotting
    if dim == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(samples_nd[:, 0], samples_nd[:, 1], s=5)
        plt.title('Custom t-SNE visualization of samples (2D)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.savefig(f'tsne_plot_custom_2d_{algo}.png')
    else:  # dim == 3
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples_nd[:, 0], samples_nd[:, 1], samples_nd[:, 2], s=5)
        ax.set_title('Custom t-SNE visualization of samples (3D)')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.savefig(f'tsne_plot_custom_3d_{algo}.png')

@torch.no_grad()
def gen_tsne_plot_sklearn(samples, dim=2, algo=1):
    from sklearn.manifold import TSNE
    
    if dim not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3")
    
    # Convert list of tensors to a single tensor and move to CPU
    samples_tensor = torch.cat(samples, dim=0).cpu().numpy()
    
    # Apply t-SNE using sklearn
    tsne = TSNE(n_components=dim, perplexity=30, n_iter=1000, random_state=SEED)
    samples_nd = tsne.fit_transform(samples_tensor)
    
    # Plotting
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
        count = 0
        for i in range(self.N):
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.N}")
            X_prev = samples[-1].detach().requires_grad_(True)
            energy = self.model(X_prev)
            grad = torch.autograd.grad(energy, X_prev)[0]
            X_prop = X_prev - self.tau * grad + torch.sqrt(self.tau) * torch.randn_like(X_prev)
            energy_prop = self.model(X_prop)
            grad_prop = torch.autograd.grad(energy_prop, X_prop)[0]
            log_q_X_prev_X_prop = -(torch.norm(X_prev - X_prop + grad_prop * self.tau / 2) ** 2) / (4 * self.tau)
            log_q_X_prop_X_prev = -(torch.norm(X_prop - X_prev + grad * self.tau / 2) ** 2) / (4 * self.tau)
            log_acceptance_ratio = min(1, torch.exp(energy - energy_prop + log_q_X_prev_X_prop - log_q_X_prop_X_prev))
            if torch.rand(1).item() < log_acceptance_ratio:
                samples.append(X_prop)
                count += 1
            else:
                samples.append(X_prev)
        
        end_time = time()
        print(f"Accepted {count} out of {self.N} proposals.")
        return samples[self.burn_in+1:], end_time - start_time

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

        end_time = time()
            
        return samples[self.burn_in+1:], end_time - start_time


# --- Main Execution ---
if __name__ == "__main__":
    model = EnergyRegressor(FEAT_DIM).to(DEVICE)
    model.load_state_dict(torch.load("../trained_model_weights.pth"))

    N = 10000
    burn_in = 2000
    tau = 0.01

    algo1 = Algo1_Sampler(model, N, burn_in, tau)
    samples, time_diff = algo1.sample()
    print(f"Sampling time 1: {time_diff:.2f} seconds")
    # gen_tsne_plot(samples, dim=2, algo=1)
    # gen_tsne_plot(samples, dim=3)
    gen_tsne_plot_sklearn(samples, dim=2, algo=1)
    gen_tsne_plot_sklearn(samples, dim=3, algo=1)

    algo2 = Algo2_Sampler(model, N, burn_in, tau)
    samples, time_diff = algo2.sample()
    print(f"Sampling time 2: {time_diff:.2f} seconds")
    # gen_tsne_plot(samples, dim=2)
    # gen_tsne_plot(samples, dim=3)
    gen_tsne_plot_sklearn(samples, dim=2, algo=2)
    gen_tsne_plot_sklearn(samples, dim=3, algo=2)