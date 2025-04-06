import numpy as np
import matplotlib.pyplot as plt

def branin_hoo(x):
    """Calculate the Branin-Hoo function value for given input."""
    x1, x2 = x
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    d = 6
    e = 10
    f = 1 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - d)**2 + e * (1 - f) * np.cos(x1) + e

# Kernel Functions (Students implement)
def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    """Compute the RBF kernel."""    
    x1_expanded = x1[:, np.newaxis, :]  # Shape: (n1, 1, d)
    x2_expanded = x2[np.newaxis, :, :]  # Shape: (1, n2, d)
    
    dists = np.sum((x1_expanded - x2_expanded) ** 2, axis=2)
    
    K = sigma_f**2 * np.exp(-0.5 * dists / length_scale**2)
    
    return K

def matern_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, nu=1.5):
    """Compute the Matérn kernel (nu=1.5)."""
    x1_expanded = x1[:, np.newaxis, :]  # Shape: (n1, 1, d)
    x2_expanded = x2[np.newaxis, :, :]  # Shape: (1, n2, d)
    
    dists = np.sqrt(np.sum((x1_expanded - x2_expanded) ** 2, axis=2))
    
    K = sigma_f**2 * (1 + np.sqrt(3) * dists / length_scale) * np.exp(-np.sqrt(3) * dists / length_scale)
    
    return K

def rational_quadratic_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, alpha=1.0):
    """Compute the Rational Quadratic kernel."""
    x1_expanded = x1[:, np.newaxis, :]  # Shape: (n1, 1, d)
    x2_expanded = x2[np.newaxis, :, :]  # Shape: (1, n2, d)
    
    dists = np.sum((x1_expanded - x2_expanded) ** 2, axis=2)
    
    K = sigma_f**2 * (1 + dists / (2 * alpha * length_scale**2)) ** (-alpha)
    
    return K

def log_marginal_likelihood(x_train, y_train, kernel_func, length_scale, sigma_f, noise=1e-4):
    """Compute the log-marginal likelihood."""
    y_train = y_train.reshape(-1, 1)
    K = kernel_func(x_train, x_train, length_scale, sigma_f) + noise * np.eye(len(x_train))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    log_likelihood = -0.5 * np.dot(y_train.T, alpha)
    log_likelihood -= np.sum(np.log(np.diagonal(L)))
    log_likelihood -= len(x_train) / 2 * np.log(2 * np.pi)
    return log_likelihood.flatten()[0]

def optimize_hyperparameters(x_train, y_train, kernel_func, noise=1e-4):
    """Optimize hyperparameters using grid search."""
    length_scale_range = np.logspace(-5, 5, 20)
    sigma_f_range = np.logspace(-5, 5, 20)
    noise_range = [1e-4]
    
    best_log_likelihood = -np.inf
    best_params = None
    
    for length_scale in length_scale_range:
        for sigma_f in sigma_f_range:
            for noise in noise_range:
                log_likelihood = log_marginal_likelihood(x_train, y_train, kernel_func, length_scale, sigma_f, noise)
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_params = (length_scale, sigma_f, noise)

    return best_params

def gaussian_process_predict(x_train, y_train, x_test, kernel_func, length_scale=1.0, sigma_f=1.0, noise=1e-4):
    """Perform GP prediction."""
    y_train = y_train.reshape(-1, 1)
    K = kernel_func(x_train, x_train, length_scale, sigma_f) + noise * np.eye(len(x_train))
    K_s = kernel_func(x_train, x_test, length_scale, sigma_f)
    K_ss = kernel_func(x_test, x_test, length_scale, sigma_f) + 1e-8 * np.eye(len(x_test))
    
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    
    y_mean = K_s.T @ alpha
    v = np.linalg.solve(L, K_s)
    y_var = K_ss - v.T @ v
    
    return y_mean.flatten(), np.sqrt(np.diag(y_var))

# Acquisition Functions (Simplified, no erf)
def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Expected Improvement acquisition function."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    z = (mu - y_best - xi) / sigma
    Phi_z = 1 / (1 + np.exp(-1.702 * z))
    phi_z = 1/np.sqrt(2 * np.pi) * np.exp(-0.5 * z**2)
    return (mu - y_best - xi) * Phi_z + sigma * phi_z

def probability_of_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Probability of Improvement acquisition function."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    z = (mu - y_best - xi) / sigma
    Phi_z = 1 / (1 + np.exp(-1.702 * z))
    return Phi_z

def plot_graph(x1_grid, x2_grid, z_values, x_train, title, filename):
    """Create and save a contour plot."""
    plt.figure(figsize=(10, 8))
    plt.contourf(x1_grid, x2_grid, z_values, levels=50, cmap='viridis')
    plt.colorbar(label='Function Value')
    plt.scatter(x_train[:, 0], x_train[:, 1], c='red', s=30, label='Sample Points', marker='x')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig("plots/"+filename)
    plt.close()    

def main():
    """Main function to run GP with kernels, sample sizes, and acquisition functions."""
    np.random.seed(0)
    n_samples_list = [10, 20, 50, 100]
    kernels = {
        'rbf': (rbf_kernel, 'RBF'),
        'matern': (matern_kernel, 'Matern (nu=1.5)'),
        'rational_quadratic': (rational_quadratic_kernel, 'Rational Quadratic')
    }
    acquisition_strategies = {
        'EI': expected_improvement,
        'PI': probability_of_improvement
    }
    
    x1_test = np.linspace(-5, 10, 100)
    x2_test = np.linspace(0, 15, 100)
    x1_grid, x2_grid = np.meshgrid(x1_test, x2_test)
    x_test = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    true_values = np.array([branin_hoo([x1, x2]) for x1, x2 in x_test]).reshape(x1_grid.shape)
    
    for kernel_name, (kernel_func, kernel_label) in kernels.items():
        for n_samples in n_samples_list:
            x_train = np.random.uniform(low=[-5, 0], high=[10, 15], size=(n_samples, 2))
            y_train = np.array([branin_hoo(x) for x in x_train])
            
            print(f"\nKernel: {kernel_label}, n_samples = {n_samples}")
            length_scale, sigma_f, noise = optimize_hyperparameters(x_train, y_train, kernel_func)
            
            for acq_name, acq_func in acquisition_strategies.items():
                x_train_current = x_train.copy()
                y_train_current = y_train.copy()
                
                y_mean, y_std = gaussian_process_predict(x_train_current, y_train_current, x_test, 
                                                        kernel_func, length_scale, sigma_f, noise)
                y_mean_grid = y_mean.reshape(x1_grid.shape)
                y_std_grid = y_std.reshape(x1_grid.shape)
                
                if acq_func is not None:
                    # Hint: Find y_best, apply acq_func, select new point, update training set, recompute GP
                    y_best = np.max(y_train_current)
                    acq_values = acq_func(y_mean, y_std, y_best)
                    x_next = x_test[np.argmax(acq_values)]
                    y_next = branin_hoo(x_next)
                    x_train_current = np.vstack((x_train_current, x_next))
                    y_train_current = np.append(y_train_current, y_next)
                    y_mean_next, y_std_next = gaussian_process_predict(x_train_current, y_train_current, x_test, kernel_func, length_scale, sigma_f, noise)
                    y_mean_grid = y_mean_next.reshape(x1_grid.shape)
                    y_std_grid = y_std_next.reshape(x1_grid.shape)
                
                acq_label = '' if acq_name == 'None' else f', Acq={acq_name}'
                plot_graph(x1_grid, x2_grid, true_values, x_train_current,
                          f'True Branin-Hoo Function (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'true_function_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_mean_grid, x_train_current,
                          f'GP Predicted Mean (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_mean_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_std_grid, x_train_current,
                          f'GP Predicted Std Dev (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_std_{kernel_name}_n{n_samples}_{acq_name}.png')

if __name__ == "__main__":
    main()