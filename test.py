import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# 1. Define target distribution and score function
def true_pdf(x, nu):
    const = gamma((nu + 1) / 2) / (np.sqrt(nu * np.pi) * gamma(nu / 2))
    return const * (1 + (x**2) / nu) ** (-(nu + 1) / 2)

def grad_log_p(x, nu):
    """Analytical gradient derived: -(\nu + 1)x / (\nu + x^2)"""
    return - (nu + 1) * x / (nu + x**2)

# 2. Unadjusted Langevin Algorithm (ULA)
def sample_ula(nu, gamma_step, num_samples, x0=0.0):
    samples = np.zeros(num_samples)
    x = x0
    for t in range(num_samples):
        # Proposal step using drift and Brownian motion noise
        noise = np.random.normal(0, np.sqrt(2 * gamma_step))
        x = x + gamma_step * grad_log_p(x, nu) + noise
        samples[t] = x
    return samples

# 3. Metropolis-Adjusted Langevin Algorithm (MALA)
def sample_mala(nu, gamma_step, num_samples, x0=0.0):
    samples = np.zeros(num_samples)
    x = x0
    acceptances = 0
    
    # Helper to calculate the transition probability density q(x_prime | x)
    def log_q(x_prime, x):
        mean = x + gamma_step * grad_log_p(x, nu)
        return -((x_prime - mean)**2) / (4 * gamma_step)
        
    for t in range(num_samples):
        # Propose a candidate
        noise = np.random.normal(0, np.sqrt(2 * gamma_step))
        x_prop = x + gamma_step * grad_log_p(x, nu) + noise
        
        # Calculate acceptance probability in log-space to maintain stability
        log_target_prop = -(nu + 1)/2 * np.log(1 + x_prop**2 / nu)
        log_target_curr = -(nu + 1)/2 * np.log(1 + x**2 / nu)
        
        log_alpha = (log_target_prop + log_q(x, x_prop)) - (log_target_curr + log_q(x_prop, x))
        
        if np.log(np.random.uniform(0, 1)) < log_alpha:
            x = x_prop
            acceptances += 1
            
        samples[t] = x
        
    print(True, f"MALA Acceptance Rate for nu={nu}, gamma={gamma_step}: {acceptances/num_samples:.2%}")
    return samples

# 4. Simulation and Visualization Execution
num_samples = 20000
nu_values = [2, 10]          # Low nu (heavy tails) vs High nu (near-Gaussian)
gamma_values = [0.1, 0.8]     # Small step size vs Big step size

fig, axes = plt.subplots(len(nu_values), len(gamma_values), figsize=(14, 10), sharex=False)

for i, nu in enumerate(nu_values):
    for j, gamma_step in enumerate(gamma_values):
        ax = axes[i, j]
        
        # Run samplers
        ula_samples = sample_ula(nu, gamma_step, num_samples)
        mala_samples = sample_mala(nu, gamma_step, num_samples)
        
        # Plot evaluation curves
        x_arr = np.linspace(-6, 6, 500)
        ax.plot(x_arr, true_pdf(x_arr, nu), 'k-', lw=2.5, label='True PDF')
        ax.hist(ula_samples, bins=70, density=True, alpha=0.5, label='ULA', color='crimson')
        ax.hist(mala_samples, bins=70, density=True, alpha=0.5, label='MALA', color='teal')
        
        ax.set_title(r"$\nu$ = {}, $\gamma$ = {}".format(nu, gamma_step), fontsize=12)
        ax.set_xlim(-6, 6)
        ax.grid(True, alpha=0.3)
        if i == 0 and j == 0:
            ax.legend()

plt.tight_layout()
plt.show()