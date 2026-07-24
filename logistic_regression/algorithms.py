import os
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from tqdm import tqdm
import numpy as np


##############################################################
## UTILS
##############################################################

def sig(x):
 return 1/(1 + np.exp(-x))

def gradient_proximal_logistic_reg(x, l, f): 
    s = 1/(1+np.exp(- np.matmul(f, x)))
    
    return np.matmul((l-s).transpose(), f).transpose()

# À VOIR - extrait de mon code 
def log_p(th, x, y_l, y_f):
    f_T_x = np.dot(y_f, x)
    log_lik = np.sum(y_l*f_T_x - np.log(1+np.exp(f_T_x)))
    log_prior = np.sum((x-th) ** 2) / 5.0
    return log_lik - log_prior

def log_p_laplace(th, x, design_matrix, labels, b=1.0):
    """
    Computes log p(theta, x, y) = log p(y | x) + log p(x | theta)
    """
    logits = np.matmul(design_matrix, x)
    log_lik = np.sum(labels * logits - np.log(1 + np.exp(logits)))
    
    # Log-prior (Laplace distribution centered at theta)
    log_prior = np.sum(-np.abs(x - th) / b)
    
    return log_lik + log_prior

##############################################################
## SOUL ALGORITHM VARIANTS
##############################################################

# SOUL with Metropolis-Hastings.
def soul_mh(log_p, th0, x0_M, y_l, y_f, T, M, B, D, delta_step, proposal_std, b=1.0):
  """
  SOUL where the latent sampling is done via Metropolis-Hastings (MH) instead of ULA.

  Parameters:
  - log_p: Function returning log p(th, X, y_l, y_f). Returns a scalar float.
  - th0: Initial parameters
  - x0_M: Initial latent variables from the previous step
  - y_l, y_f: Observed data
  - T: Number of outer optimization steps
  - M: Number of MH steps
  - B: Burn-in steps
  - D: 
  - delta_step: Step size for theta update
  - proposal_std: Standard deviation of the Gaussian random walk proposal (replaces gamma_step)
  - b: Scale parameter of the Laplace prior (default 1.0)
  """
  th = np.copy(th0)
  x_t = np.copy(x0_M[:, 0:1]).reshape(D, 1)

  x_values = np.array(x0_M)
  th_list = [np.copy(th)]
  al=0

  for t in range(1, T + 1):

    # Metropolis-Hastings step
    for m in range(1, M + 1):
      z = np.random.normal(0.0, 1.0, x_t.shape)
      x_prop = x_t + proposal_std*z

      # The proposal is symmetri ie. q(x_t | x_prop) = q(x_prop | x_t), so the proposal ratio cancels out.
      log_alpha = min(0, log_p(th, x_prop, y_l, y_f, b) - log_p(th, x_t, y_l, y_f, b))

      # Accept or reject
      if np.log(np.random.uniform(0.0, 1.0)) < log_alpha:
          x_t = x_prop # Accept proposal
      # else: x_t remains the same

      # Store the current position (accpted or not)
      x_values = np.append(x_values, np.copy(x_t), axis=1)

    burnin_x_samples = x_values[:, -(M-B):] # Shape (D, M-B)

    # Compute average gradient with respect to theta over the kept samples
    avg_grad_th = np.zeros_like(th)
    for idx in range(M - B):
      x_m_burnin = burnin_x_samples[:, idx:idx+1] # Shape (D, 1)
      # Use the provided theta gradient function
      avg_grad_th += ((x_m_burnin - th).sum(0) / 5)

    # Update theta
    th = th + delta_step * (avg_grad_th / (M - B))
    th_list.append(np.copy(th))

  return th_list, x_values

# SOUL with Metropolis-Hastings with a decaying learning rate 
def soul_mh_decay(log_p, th0, x0_M, y_l, y_f, T, M, B, D, delta_step, proposal_std, b=1.0, gamma=1):
  """
  SOUL where the latent sampling is done via Metropolis-Hastings (MH) instead of ULA.

  Parameters:
  - log_p: Function returning log p(th, X, y_l, y_f). Returns a scalar float.
  - th0: Initial parameters
  - x0_M: Initial latent variables from the previous step
  - y_l, y_f: Observed data
  - T: Number of outer optimization steps
  - M: Number of MH steps
  - B: Burn-in steps
  - D: 
  - delta_step: Step size for theta update
  - proposal_std: Standard deviation of the Gaussian random walk proposal (replaces gamma_step)
  - b: Scale parameter of the Laplace prior (default 1.0)
  """
  th = np.copy(th0)
  x_t = np.copy(x0_M[:, 0:1]).reshape(D, 1)

  x_values = np.array(x0_M)
  th_list = [np.copy(th)]
  al=0

  for t in range(1, T + 1):
    current_delta = delta_step * (gamma**t)

    # Metropolis-Hastings step
    for m in range(1, M + 1):
      z = np.random.normal(0.0, 1.0, x_t.shape)
      x_prop = x_t + proposal_std*z

      # The proposal is symmetri ie. q(x_t | x_prop) = q(x_prop | x_t), so the proposal ratio cancels out.
      log_alpha = min(0, log_p(th, x_prop, y_l, y_f, b) - log_p(th, x_t, y_l, y_f, b))

      # Accept or reject
      if np.log(np.random.uniform(0.0, 1.0)) < log_alpha:
          x_t = x_prop # Accept proposal
      # else: x_t remains the same

      # Store the current position (accpted or not)
      x_values = np.append(x_values, np.copy(x_t), axis=1)

    burnin_x_samples = x_values[:, -(M-B):] # Shape (D, M-B)

    # Compute average gradient with respect to theta over the kept samples
    avg_grad_th = np.zeros_like(th)
    for idx in range(M - B):
      x_m_burnin = burnin_x_samples[:, idx:idx+1] # Shape (D, 1)
      # Use the provided theta gradient function
      avg_grad_th += ((x_m_burnin - th).sum(0) / 5)

    # Update theta
    th = th + delta_step * (avg_grad_th / (M - B))
    th_list.append(np.copy(th))

  return th_list, x_values

#@title SOUL with PAIES algorithm and the stretch move
def soul_stretch(log_p, th0, x0_N, y_l, y_f, T, M, B, delta_step, a=2.0):
    """
    Stochastic Optimisation via Affine-Invariant Ensemble SOUL - stretch move.

    Parameters:
    - log_p: Function returning log p(th, X, y_l, y_f). Shape of X is (D, 1).
    - th0: Initial parameters
    - x0_N: Initial latent variables ensemble (S), shape (D, N) where N is number of walkers
    - y_l, y_f: Observed data
    - T: Number of outer optimization steps
    - M: Target chain length (number of ensemble update steps)
    - B: Burn-in steps
    - delta_step: Step size for theta update
    - a: Stretch move scale parameter (typically 2.0)
    """
    th = np.copy(th0)
    D, N = x0_N.shape
    half = N // 2

    # S is our active ensemble of N walkers, shape (D, N)
    S = np.copy(x0_N)

    # Collection C to store all states across the chain length M
    C = np.expand_dims(np.copy(S), axis=2)
    th_list = [th0]

    for t in range(1, T + 1):
        # Affine-Invariant Ensemble steps
        for m in range(1, M + 1):
            # Randomly shuffle and split S into two halves S1 and S2
            indices = np.random.permutation(N)
            S1_idx, S2_idx = indices[:half], indices[half:]

            # Update each half using the complementary half
            for ens_idx, comp_idx in [(S1_idx, S2_idx), (S2_idx, S1_idx)]:
                for i in ens_idx:
                    X_i = S[:, i:i+1] # shape D, 1

                    # Select a random walker X_j from the complementary ensemble
                    j = np.random.choice(comp_idx)
                    X_j = S[:, j:j+1]

                    # Propose new position using the stretch move rule R
                    u_z = np.random.uniform(0.0, 1.0)
                    z = (a + (1/a) - 2)*(u_z**2) + 2*u_z*(1-(1/a)) + (1/a)
                    X_i_new = X_j + z * (X_i - X_j)

                    log_alpha = (D - 1) * np.log(z) + log_p(th, X_i_new, y_l, y_f) - log_p(th, X_i, y_l, y_f)

                    # Accept or reject
                    if np.log(np.random.uniform(0.0, 1.0)) < log_alpha:
                        S[:, i:i+1] = X_i_new

            # Save new position (accepted or not)
            C = np.concatenate((C, np.expand_dims(np.copy(S), axis=2)), axis=2)

        burnin_ensemble_samples = C #[:, :, -(M - B):]  # Shape (D, N, M - B)
        # no burnin for a test

        # Reshape to treat all walker positions (post-burnin) as individual particles: shape (D, N * (M - B))
        flat_samples = burnin_ensemble_samples.reshape(D, -1)
        num_samples = flat_samples.shape[1]

        # Compute new theta using the averaged gradient
        avg_grad_th = np.zeros_like(th)
        for idx in range(num_samples):
            x_m_burnin = flat_samples[:, idx:idx+1]  # Shape (D, 1)
            avg_grad_th += ((x_m_burnin - th).sum(0) / 5)

        th = th + delta_step * avg_grad_th / num_samples
        th_list.append(np.copy(th))

    return th_list, C


##############################################################
### MOREAU-YOSIDA LANGEVIN ALGORITHMS AND PROXIMAL MAPS
##############################################################

def proximal_map_laplace_approx(theta, particles, gamma):
    """
    Compute the proximal mapping approximately for a Laplace prior.
    """

    input_proximal_x = particles 

    input_proximal_theta = theta
    
    x_prox = input_proximal_theta + (input_proximal_x - np.sign(input_proximal_x - input_proximal_theta) * gamma - input_proximal_theta) * (np.abs(input_proximal_x-input_proximal_theta) >= gamma)
    theta_prox = input_proximal_theta + np.sign(x_prox - input_proximal_theta).sum(axis = 0) * gamma
    
    proximal_output_x =  x_prox
    proximal_output_theta = theta_prox 
    
    return np.expand_dims(proximal_output_theta, axis=0), proximal_output_x


def proximal_map_laplace_iterative(theta, particles, gamma):
    """
    Compute the proximal mapping iteratively for a Laplace prior.
    """

    input_proximal_x = particles 

    input_proximal_theta = theta

    # Initialize input for the fixed point iteration method.
    x_prox = input_proximal_x 
    theta_prox = input_proximal_theta
    for _ in range(40):
        x_prox = input_proximal_x - np.sign(x_prox - theta_prox) * gamma
        theta_prox = input_proximal_theta + np.sign(x_prox - theta_prox).sum(axis = 0) * gamma

    return np.expand_dims(theta_prox, axis=0), x_prox



def mypipla(th, X, design_matrix, data, proximal_map = proximal_map_laplace_approx, N = 100, K = 4000, gamma = 0.001, h = 0.001, progress_bar=True):
    """
    Run the Moreau-Yosida Interacting Particle Langevin Algorithm for a given proximal mapping.
    """

    for k in (tqdm(range(K), disable=not progress_bar)):

        Xk = X[:, -N:]

        proximal_output_theta_expand, proximal_output_particles = proximal_map(th[k], Xk, gamma = gamma)  
        
        proximal_output_theta = proximal_output_theta_expand.mean(axis = 1)
        
        Xkp1 =  Xk * (1-h/gamma) + h * gradient_proximal_logistic_reg(Xk, data, design_matrix) + h * proximal_output_particles/gamma + np.sqrt(2*h) * np.random.normal(0, 1, Xk.shape)
        thkp1 = th[k] * (1-h/gamma) + h * proximal_output_theta/gamma + np.sqrt(2 * h/N) * np.random.normal(0, 1, 1)
        
        X = np.append(X, Xkp1, axis=1) # Store updated cloud.
        th = np.append(th, thkp1)  # Update theta.

    return th, X


def mypgd(th, X, design_matrix, data, proximal_map = proximal_map_laplace_approx, N = 100, K = 4000, gamma = 0.001, h = 0.001, progress_bar=True):
    """
    Run the Moreau-Yosida Particle Gradient Descent for a given proximal mapping.
    """

    for k in (tqdm(range(K), disable=not progress_bar)):

        Xk = X[:, -N:]

        proximal_output_theta_expand, proximal_output_particles = proximal_map(th[k], Xk, gamma = gamma)  
        
        proximal_output_theta = proximal_output_theta_expand.mean(axis = 1)
        
        Xkp1 =  Xk * (1-h/gamma) + h * gradient_proximal_logistic_reg(Xk, data, design_matrix) + h * proximal_output_particles/gamma + np.sqrt(2*h) * np.random.normal(0, 1, Xk.shape)
        thkp1 = th[k] * (1-h/gamma) + h * proximal_output_theta/gamma 
        
        X = np.append(X, Xkp1, axis=1) # Store updated cloud.
        th = np.append(th, thkp1)  # Update theta.

    return th, X
