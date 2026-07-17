import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def grad_log_target(x, nu):
    return(-(nu+1)*x / (nu + x**2))

def pdf(x, nu):
    a = gamma((nu+1)/2)
    b = np.sqrt(nu*np.pi)*gamma(nu/2)
    c = (1 + (x**2)/nu) ** (-(nu+1)/2)
    return (a/b)*c

def log_pdf(x, nu):
    #log_const = np.log(gamma((nu+1)/2)) - 0.5*np.log(nu*np.pi) - np.log(gamma(nu/2)) # it cancels out afterwards
    return - ((nu+1) / 2) * np.log(1+ (x**2)/nu) # + log_const


# 1. ULA
def ula(nu, gam_step, n):
    samples = np.zeros(n)
    for i in range(1,n):
        current = samples[i-1]
        grad_curr = grad_log_target(current, nu)

        new = current + gam_step*grad_curr + np.sqrt(2*gam_step)*np.random.normal(0, 1)
        samples[i] = new
    return samples


# 2. MALA
def mala(nu, gam_step, n):
    samples = np.zeros(n)
    for i in range(1,n):
        current = samples[i-1]
        grad_curr = grad_log_target(current, nu)

        new = current + gam_step*grad_curr + np.sqrt(2*gam_step)*np.random.normal(0, 1)
        grad_new = grad_log_target(new, nu)

        mu_curr = new + gam_step*grad_new
        mu_new = current + gam_step*grad_curr
        log_q_new_given_curr = -((new - mu_new)**2) / (4*gam_step)
        log_q_curr_given_new = -((current - mu_curr)**2) / (4*gam_step) 

        log_r = (log_pdf(new, nu) + log_q_curr_given_new) - (log_pdf(current, nu) + log_q_new_given_curr)
        
        u = np.random.uniform(0,1)
        if np.log(u) < log_r:
            samples[i] = new
        else:
            samples[i] = current
    return samples

# --- Parameters ---
n = 50000          # total iter nb
burnin = 10000     
nu_values = [2, 5, 10]
gamma_values = [0.1, 0.5, 0.9]


#########  plot instructions
fig, axes = plt.subplots(len(nu_values), len(gamma_values), figsize=(14, 10), sharex=False)

for i, nu in enumerate(nu_values):
    for j, gamma_step in enumerate(gamma_values):
        ax = axes[i, j]
        
        # Run samplers
        ula_samples = ula(nu, gamma_step, n)
        mala_samples = mala(nu, gamma_step, n)
        
        # Plot evaluation curves
        x_arr = np.linspace(-6, 6, 500)
        ax.plot(x_arr, pdf(x_arr, nu), 'k-', lw=2.5, label='True PDF')
        ax.hist(ula_samples[burnin:], bins=70, density=True, alpha=0.5, label='ULA', color='crimson')
        ax.hist(mala_samples[burnin:], bins=70, density=True, alpha=0.5, label='MALA', color='teal')
        
        ax.set_title(r"$\nu$ = {}, $\gamma$ = {}".format(nu, gamma_step), fontsize=12)
        ax.set_xlim(-6, 6)
        ax.grid(True, alpha=0.3)
        if i == 0 and j == 0:
            ax.legend()

plt.tight_layout()
plt.show()
