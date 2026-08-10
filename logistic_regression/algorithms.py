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
  - D: Number of dimensions of latent space
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
      x_m_burnin = burnin_x_samples[:, idx:idx+1] 
      avg_grad_th += np.sign(x_m_burnin - th).mean(axis=0) / b # Adapted to Laplace prior

    th = th + delta_step * (avg_grad_th / (M - B))
    th_list.append(np.copy(th))

  return th_list, x_values

# SOUL with Metropolis-Hastings with a decaying learning rate: gamma
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
  - D: Number of dimensions of latent space
  - delta_step: Step size for theta update
  - proposal_std: Standard deviation of the Gaussian random walk proposal (replaces gamma_step)
  - b: Scale parameter of the Laplace prior (default 1.0)
  - gamma: Decay factor applied per outer step (gamma <= 1); current_delta = delta_step * gamma**t
  """
  th = np.copy(th0)
  x_t = np.copy(x0_M[:, 0:1]).reshape(D, 1)

  x_values = np.array(x0_M)
  th_list = [np.copy(th)]
  al=0

  for t in range(1, T + 1):
    current_delta = delta_step * (gamma**t)

    for m in range(1, M + 1):
      z = np.random.normal(0.0, 1.0, x_t.shape)
      x_prop = x_t + proposal_std*z

      log_alpha = min(0, log_p(th, x_prop, y_l, y_f, b) - log_p(th, x_t, y_l, y_f, b))

      if np.log(np.random.uniform(0.0, 1.0)) < log_alpha:
          x_t = x_prop # Accept proposal

      x_values = np.append(x_values, np.copy(x_t), axis=1)

    burnin_x_samples = x_values[:, -(M-B):] 

    avg_grad_th = np.zeros_like(th)
    for idx in range(M - B):
      x_m_burnin = burnin_x_samples[:, idx:idx+1]
      avg_grad_th += np.sign(x_m_burnin - th).mean(axis=0) / b 

    th = th + delta_step * (avg_grad_th / (M - B))
    th_list.append(np.copy(th))

  return th_list, x_values

# SOUL MHSS
def soul_mh_ss(log_p, th0, x0_M, y_l, y_f, T, M, B, delta_step, proposal_std, lower_bounds, upper_bounds, b=1):
    """
    SOUL where the latent sampling is done via Metropolis-Hastings with Standardised Scaling (MH SS).

    Parameters:
    - log_p: Function returning log p(th, X, y_l, y_f). Returns a scalar float.
    - th0: Initial parameters
    - x0_M: Initial latent variables from the previous step
    - y_l, y_f: Observed data
    - T: Number of outer optimization steps
    - M: Number of MH steps
    - B: Burn-in steps
    - delta_step: Step size for theta update
    - proposal_std: Standard deviation of the Gaussian random walk proposal in the [0, 1] space (typically 0.05 to 0.2)
    - lower_bounds: numpy array of shape (D, 1) containing the lower prior limits for X
    - upper_bounds: numpy array of shape (D, 1) containing the upper prior limits for X
    """
    th = np.copy(th0)
    D = x0_M.shape[0] # Can be given as a parameter or computed directly inside.
    x_t = np.copy(x0_M[:, 0:1]).reshape(D, 1)

    bounds_range = upper_bounds - lower_bounds

    x_values = np.array(x0_M)
    th_list = [np.copy(th)]

    n_accept = 0
    n_total = 0

    for t in range(1, T + 1):

        for m in range(1, M + 1):
            # Transform current position to [0, 1] standardised space
            x_t_std = (x_t - lower_bounds) / bounds_range

            z = np.random.normal(0.0, 1.0, x_t.shape)
            x_prop_std = x_t_std + proposal_std * z

            # proposals must stay within [0, 1]
            if np.any(x_prop_std < 0.0) or np.any(x_prop_std > 1.0):
                # Out of bounds proposal: we reject it and stay at x_t
                x_values = np.append(x_values, np.copy(x_t), axis=1)
                continue

            x_prop = x_prop_std * bounds_range + lower_bounds

            # Since the proposal is symmetric in the standardised space and we reject
            # out-of-bounds proposals, the proposal ratio q(x_t | x_prop) / q(x_prop | x_t) remains 1.
            log_alpha = min(0, log_p(th, x_prop, y_l, y_f) - log_p(th, x_t, y_l, y_f))

            # Accept or reject
            if np.log(np.random.uniform(0.0, 1.0)) < log_alpha:
                x_t = x_prop
                n_accept += 1
            n_total+=1

            # Store the current position
            x_values = np.append(x_values, np.copy(x_t), axis=1)

            burnin_x_samples = x_values[:, -(M - B):]


        avg_grad_th = np.zeros_like(th)
        for idx in range(M - B):
            x_m_burnin = burnin_x_samples[:, idx:idx + 1]
            avg_grad_th += np.sign(x_m_burnin - th).mean(axis=0) / b   # mean over D - Laplace score

        th = th + delta_step * (avg_grad_th / (M - B))
        th_list.append(np.copy(th))

        # Uncomment the following to debug or retune if needed.
        #if t%13==0:
            #print("t=", t, "and theta is ", th, "mean(x_t)=", x_t.mean(), "accept rate=", n_accept/n_total)

    return th_list, x_values

# SOUL MHSS with decaying learning rate
def soul_mh_ss_decay(log_p, th0, x0_M, y_l, y_f, T, M, B, delta_step, proposal_std,
                      lower_bounds, upper_bounds, b=1, gamma=1):
    """
    SOUL where the latent sampling is done via Metropolis-Hastings with Standardised Scaling (MH SS),
    with a decaying learning rate for the theta update: current_delta = delta_step * gamma**t.

    Parameters:
    - log_p: Function returning log p(th, X, y_l, y_f). Returns a scalar float.
    - th0: Initial parameters
    - x0_M: Initial latent variables from the previous step
    - y_l, y_f: Observed data
    - T: Number of outer optimization steps
    - M: Number of MH steps
    - B: Burn-in steps
    - delta_step: Initial step size for theta update
    - proposal_std: Standard deviation of the Gaussian random walk proposal in the [0, 1] space (typically 0.05 to 0.2)
    - lower_bounds: numpy array of shape (D, 1) containing the lower prior limits for X
    - upper_bounds: numpy array of shape (D, 1) containing the upper prior limits for X
    - b: Scale parameter of the Laplace prior (default 1)
    - gamma: Decay factor applied per outer step (gamma <= 1); current_delta = delta_step * gamma**t
    """
    th = np.copy(th0)
    D = x0_M.shape[0]
    x_t = np.copy(x0_M[:, 0:1]).reshape(D, 1)

    bounds_range = upper_bounds - lower_bounds

    x_values = np.array(x0_M)
    th_list = [np.copy(th)]

    n_accept = 0
    n_total = 0

    for t in range(1, T + 1):
        # Decaying learning rate for this outer step
        current_delta = delta_step * (gamma ** t)

        for m in range(1, M + 1):
            # Transform current position to [0, 1] standardised space
            x_t_std = (x_t - lower_bounds) / bounds_range

            z = np.random.normal(0.0, 1.0, x_t.shape)
            x_prop_std = x_t_std + proposal_std * z

            # proposals must stay within [0, 1]
            if np.any(x_prop_std < 0.0) or np.any(x_prop_std > 1.0):
                # Out of bounds proposal: we reject it and stay at x_t
                x_values = np.append(x_values, np.copy(x_t), axis=1)
                continue

            x_prop = x_prop_std * bounds_range + lower_bounds

            # Since the proposal is symmetric in the standardised space and we reject
            # out-of-bounds proposals, the proposal ratio q(x_t | x_prop) / q(x_prop | x_t) remains 1.
            log_alpha = min(0, log_p(th, x_prop, y_l, y_f) - log_p(th, x_t, y_l, y_f))

            # Accept or reject
            if np.log(np.random.uniform(0.0, 1.0)) < log_alpha:
                x_t = x_prop
                n_accept += 1
            n_total += 1

            # Store the current position
            x_values = np.append(x_values, np.copy(x_t), axis=1)

            burnin_x_samples = x_values[:, -(M - B):]

        avg_grad_th = np.zeros_like(th)
        for idx in range(M - B):
            x_m_burnin = burnin_x_samples[:, idx:idx + 1]
            avg_grad_th += np.sign(x_m_burnin - th).mean(axis=0) / b   # mean over D, correct Laplace score

        # Update theta using the decayed step size
        th = th + current_delta * (avg_grad_th / (M - B))
        th_list.append(np.copy(th))

        # Uncomment the following to debug or tune parameters if needed
        #if t % 13 == 0:
            #print("t=", t, "current_delta=", current_delta, "and theta is ", th, "mean(x_t)=", x_t.mean(), "accept rate=", n_accept / n_total)

    return th_list, x_values

# SOUL with PAIES algorithm and the stretch move - with a decaying learning rate
def soul_stretch_fast_decay(log_p, th0, x0_N, y_l, y_f, T, M, B, delta_step, a=2.0, b=1, gamma =1):
    """
    SOUL where the latent sampling is done via PAIES algorithm and the stretch move,
    with a decaying learning rate for the theta update: current_delta = delta_step * gamma**t.

    Parameters:
    - log_p: Function returning log p(th, X, y_l, y_f). Returns a scalar float.
    - th0: Initial parameters
    - x0_N: 
    - y_l, y_f: Observed data
    - T: Number of outer optimization steps
    - M: Number of PAIES steps
    - B: Burn-in steps
    - delta_step: Initial step size for theta update
    - a:
    - b: Scale parameter of the Laplace prior (default 1)
    - gamma: Decay factor applied per outer step (gamma <= 1); current_delta = delta_step * gamma**t (default 1 - no decaying learning rate)
    """
    th = np.copy(th0)
    D, N = x0_N.shape
    half = N // 2

    S = np.copy(x0_N)

    # Pre-allocate memory for C to avoid dynamic concatenation overhead
    total_steps = T * M
    C = np.zeros((D, N, total_steps + 1))
    C[:, :, 0] = S

    th_list = [th0]
    global_step = 1

    for t in range(1, T + 1):
        current_delta = delta_step * (gamma ** t)
        for m in range(1, M + 1):
            indices = np.random.permutation(N)
            S1_idx, S2_idx = indices[:half], indices[half:]

            for ens_idx, comp_idx in [(S1_idx, S2_idx), (S2_idx, S1_idx)]:
                X_ens = S[:, ens_idx]  # Shape (D, half)

                # Pick random complement walkers for each walker in the active set
                rand_comp_indices = np.random.choice(comp_idx, size=half)
                X_comp = S[:, rand_comp_indices]  # Shape (D, half)

                # Sample z for all active walkers simultaneously
                u_z = np.random.uniform(0.0, 1.0, size=(1, half))
                z = (a + (1.0/a) - 2.0)*(u_z**2) + 2.0*u_z*(1.0 - (1.0/a)) + (1.0/a)

                # Vectorized proposal
                X_new = X_comp + z * (X_ens - X_comp)

                # Compute log_p for current and proposed positions across active walkers
                log_p_cur = np.array([log_p(th, X_ens[:, i:i+1], y_l, y_f) for i in range(half)])
                log_p_new = np.array([log_p(th, X_new[:, i:i+1], y_l, y_f) for i in range(half)])

                log_alpha = (D - 1) * np.log(z.reshape(half)) + log_p_new - log_p_cur

                # Vectorized accept/reject step
                accept_mask = np.log(np.random.uniform(0.0, 1.0, size=half)) < log_alpha
                X_ens[:, accept_mask] = X_new[:, accept_mask]

                # Update main ensemble state
                S[:, ens_idx] = X_ens

            # Store current state directly into pre-allocated array
            C[:, :, global_step] = S
            global_step += 1

        # Flatten samples for theta update
        #flat_samples = C[:, :, :global_step].reshape(D, -1)
        start = (t - 1) * M + 1          # first index of this iteration's block
        current = C[:, :, start + B : global_step]   # drop B burn-in steps
        flat_samples = current.reshape(D, -1)

        # Vectorized average theta gradient computation
        avg_grad_th = np.mean(np.sign(flat_samples - th)) / b

        th = th + current_delta * avg_grad_th
        th_list.append(np.copy(th))

        # Uncomment the following to debug or tune parameters if necessary
        if t%17==0:
            print("t=", t, "and theta = ", th)

    return th_list, C



def soul_walk_fast_decay(log_p, th0, x0_N, y_l, y_f, T, M, B, delta_step, b=1.0, gamma = 1):
    """
    Stochastic Optimisation via Affine-Invariant Ensemble SOUL - walk move (optimized),
    with a decaying learning rate for the theta update: current_delta = delta_step * gamma**t.

    Parameters:
    - log_p: Function returning log p(th, X, y_l, y_f). Returns a scalar float.
    - th0: Initial parameters
    - x0_N: 
    - y_l, y_f: Observed data
    - T: Number of outer optimization steps
    - M: Number of inner PAIES steps
    - B: Burn-in steps
    - delta_step: Initial step size for theta update
    - b: Scale parameter of the Laplace prior (default 1)
    - gamma: Decay factor applied per outer step (gamma <= 1); current_delta = delta_step * gamma**t (default 1 - no decaying learning rate)
    """
    th = np.copy(th0)
    D, N = x0_N.shape
    half = N // 2

    S = np.copy(x0_N)

    total_steps = T * M
    C = np.zeros((D, N, total_steps + 1))
    C[:, :, 0] = S

    th_list = [th0]
    global_step = 1

    for t in range(1, T + 1):
        current_delta = delta_step * (gamma ** t)
        for m in range(1, M + 1):
            indices = np.random.permutation(N)
            S1_idx, S2_idx = indices[:half], indices[half:]

            for ens_idx, comp_idx in [(S1_idx, S2_idx), (S2_idx, S1_idx)]:
                S_comp = S[:, comp_idx]
                Cov = np.cov(S_comp, ddof=0)
                if D == 1:
                    Cov = np.array([[Cov]])

                X_ens = S[:, ens_idx] 

                W = np.random.multivariate_normal(np.zeros(D), Cov, size=half).T
                X_prop = X_ens + W

                log_p_cur = np.array([log_p(th, X_ens[:, i:i+1], y_l, y_f) for i in range(half)])
                log_p_prop = np.array([log_p(th, X_prop[:, i:i+1], y_l, y_f) for i in range(half)])

                log_alpha = log_p_prop - log_p_cur

                accept_mask = np.log(np.random.uniform(0.0, 1.0, size=half)) < log_alpha
                X_ens[:, accept_mask] = X_prop[:, accept_mask]

                S[:, ens_idx] = X_ens

            C[:, :, global_step] = S
            global_step += 1

        #flat_samples = C[:, :, :global_step].reshape(D, -1)
        start = (t - 1) * M + 1   
        current = C[:, :, start + B : global_step]   # Takes ONLY current iteration's samples
        flat_samples = current.reshape(D, -1)

        #avg_grad_th = np.mean((flat_samples - th).sum(axis=0) / 5)
        avg_grad_th = np.mean(np.sign(flat_samples - th)) / b

        th = th + current_delta * avg_grad_th
        th_list.append(np.copy(th))

        if t%17==0:
            print("t=", t, "and theta = ", th)

    return th_list, C


def soul_walk_fast_decay_v2(log_p, th0, x0_N, y_l, y_f, T, M, B, delta_step, b=1.0, gamma=1.0, s=3):
    """
    Stochastic Optimisation via Affine-Invariant Ensemble SOUL - Walk Move.
    Uses s random complement walkers per active walker to construct proposals.
    """
    th = np.copy(th0)
    D, N = x0_N.shape
    half = N // 2

    S = np.copy(x0_N)

    total_steps = T * M
    C = np.zeros((D, N, total_steps + 1))
    C[:, :, 0] = S

    th_list = [th0]
    global_step = 1

    for t in range(1, T + 1):
        current_delta = delta_step * (gamma ** t)

        for m in range(1, M + 1):
            indices = np.random.permutation(N)
            S1_idx, S2_idx = indices[:half], indices[half:]

            for ens_idx, comp_idx in [(S1_idx, S2_idx), (S2_idx, S1_idx)]:
                X_ens = S[:, ens_idx]  # Shape (D, half)

                # Pick s random distinct walkers from complement set for EACH active walker
                # Shape of rand_comp_idx: (half, s)
                rand_comp_idx = np.array([np.random.choice(comp_idx, size=s, replace=False) for _ in range(half)])
                
                # Extract complement samples: shape (D, half, s)
                X_comp_s = S[:, rand_comp_idx].transpose(0, 1, 2)  
                
                # Mean of chosen s walkers for each active walker: shape (D, half, 1)
                x_bar = np.mean(X_comp_s, axis=2, keepdims=True)

                # Standard normal vector z: shape (s, half)
                z = np.random.randn(s, half)

                # Vectorized Walk proposal: W = sum_k z_k * (X_k - x_bar)
                # Shape of (X_comp_s - x_bar): (D, half, s)
                W = np.einsum('k j, d j k -> d j', z, X_comp_s - x_bar)

                X_prop = X_ens + W

                # Compute log_p for current and proposed positions
                log_p_cur = np.array([log_p(th, X_ens[:, i:i+1], y_l, y_f) for i in range(half)])
                log_p_prop = np.array([log_p(th, X_prop[:, i:i+1], y_l, y_f) for i in range(half)])

                log_alpha = log_p_prop - log_p_cur

                # Accept / Reject
                accept_mask = np.log(np.random.uniform(0.0, 1.0, size=half)) < log_alpha
                X_ens[:, accept_mask] = X_prop[:, accept_mask]

                S[:, ens_idx] = X_ens

            C[:, :, global_step] = S
            global_step += 1

        # Slicing current iteration's non-burned-in samples
        start = (t - 1) * M + 1
        current = C[:, :, start + B : global_step]
        flat_samples = current.reshape(D, -1)

        # Average theta gradient computation
        avg_grad_th = np.mean(np.sign(flat_samples - th)) / b

        th = th + current_delta * avg_grad_th
        th_list.append(np.copy(th))

        if t % 50 == 0:
            print(f"t={t}, theta={th}")

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
