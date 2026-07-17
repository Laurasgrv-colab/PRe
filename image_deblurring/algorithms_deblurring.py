import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

###########################
## AUXILIARY FUNCTIONS
###########################

def generate_blur_matrix(n1, n2, patch_size=10):
    """Generates a blurring matrix H that uniformly averages over a patch_size x patch_size neighborhood."""
    size = n1 * n2
    H = sp.lil_matrix((size, size)) # This is a structure for constructing sparse matrices incrementally.
    pad = patch_size // 2
    
    for i in range(n1):
        for j in range(n2):
            row_idx = i * n2 + j
            weights = []
            indices = []
            
            for di in range(-pad, pad + 1):
                for dj in range(-pad, pad + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n1 and 0 <= nj < n2:
                        col_idx = ni * n2 + nj
                        indices.append(col_idx)
                        weights.append(1)
            
            weights = np.array(weights) / np.sum(weights)
            H[row_idx, indices] = weights
    
    return H.tocsr()

def discrete_gradient(n1, n2):
    """Constructs the discrete gradient operator for total variation."""
    size = n1 * n2
    Dx = sp.lil_matrix((size, size))
    Dy = sp.lil_matrix((size, size))
    
    for i in range(n1):
        for j in range(n2):
            idx = i * n2 + j
            if i < n1 - 1:
                Dx[idx, idx] = -1
                Dx[idx, idx + n2] = 1
            if j < n2 - 1:
                Dy[idx, idx] = -1
                Dy[idx, idx + 1] = 1
    
    return Dx.tocsr(), Dy.tocsr()

def total_variation(x, Dx, Dy):
    """Computes the total variation prior."""
    x_flat = x.reshape(-1, x.shape[2])
    grad_x = Dx @ x_flat
    grad_y = Dy @ x_flat
    return np.sum(np.sqrt(grad_x**2 + grad_y**2), axis=0)


###########################
## ALGORITHMS
###########################

import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

def soul_mh(theta, w_init, H, y, sigma, lambdaaa, Dx, Dy, original, 
            K=1000, M=10, B=2, delta_step=0.01, proposal_std=0.01):
    """
    Samples from the posterior using SOUL where the latent particles 
    are updated via Metropolis-Hastings (MH) (using discrete TV operations).

    Parameters:
    -----------
    theta : float
        Initial value of the hyperparameter log-regularization parameter.
    w_init : numpy.ndarray
        Initial latent image particle tensor of shape (height, width, N),
        where N is the number of particles.
    H : scipy.sparse matrix or numpy.ndarray
        Forward blur/degradation operator matrix of shape (data_dim, pixel_dim).
    y : numpy.ndarray
        Observed noisy/degraded image (can be 2D matrix or flattened).
    sigma : float
        Standard deviation of the additive white Gaussian observation noise.
    lambdaaa : float
        Base regularization parameter scaling the Total Variation penalty.
    Dx : scipy.sparse.csr_matrix
        Discrete horizontal gradient operator matrix for TV computation.
    Dy : scipy.sparse.csr_matrix
        Discrete vertical gradient operator matrix for TV computation.
    original : numpy.ndarray
        Ground truth reference image of shape (height, width) used to compute metrics.
    K : int, optional
        Number of outer iterations (optimization/sampling steps). Default is 1000.
    M : int, optional
        Number of Metropolis-Hastings inner proposal loops per outer iteration. Default is 10.
    B : int, optional
        Burn-in steps for latent samples (retained for backward compatibility). Default is 2.
    delta_step : float, optional
        Step size (learning rate) for the theta parameter gradient update. Default is 0.01.
    proposal_std : float, optional
        Standard deviation of the Gaussian random-walk proposal for MH steps. Default is 0.01.

    Returns:
    --------
    w : numpy.ndarray
        Final sampled latent image particles of shape (height, width, N).
    nmse_values : numpy.ndarray
        Normalized Mean Squared Error track over K iterations, averaged across particles.
    theta_values : numpy.ndarray
        Evolution history of the theta parameter across all K iterations.
    mse_values : numpy.ndarray
        Mean Squared Error track over K iterations, averaged across particles.
    ssim_values : numpy.ndarray
        Structural Similarity Index Measure track over K iterations, evaluated on the first particle.
    """
    w = w_init.copy()
    height, width, N = w.shape
    Dw = w[:, :, 0].size  # Dimension of a single particle (number of pixels)
    
    # Initialize tracking arrays
    nmse_values = np.zeros(K)
    mse_values = np.zeros(K)
    ssim_values = np.zeros(K)
    theta_values = np.zeros(K) 
    
    y_flat = y.flatten()[:, None]
    
    def log_posterior(img_tensor, th):
        """Computes log-posterior up to an additive constant for all particles."""
        flat_tensor = img_tensor.reshape(-1, N)
        
        # Gaussian Likelihood: -1/(2 * sigma^2) * ||Hw - y||^2
        residual = H @ flat_tensor - y_flat
        likelihood = -0.5 * np.sum(residual**2, axis=0) / (sigma**2)
        
        # TV Prior: -exp(theta) * lambdaaa * TV(w)
        tv_penalty = total_variation(img_tensor, Dx, Dy)
        prior = -np.exp(th) * lambdaaa * tv_penalty
        
        return likelihood + prior

    for k in tqdm(range(K)):
        for m in range(M):
            w_proposal = w + np.random.normal(0, proposal_std, w.shape)
            
            # Evaluate log-posterior for current and proposed states
            log_p_current = log_posterior(w, theta)
            log_p_proposal = log_posterior(w_proposal, theta)
            
            # Acceptance step
            log_alpha = log_p_proposal - log_p_current
            accept = np.log(np.random.uniform(0, 1, N)) < log_alpha
            
            # Apply update only to the accepted particles
            w[:, :, accept] = w_proposal[:, :, accept]
            
        # Theta Update 
        avg_tv = np.mean(total_variation(w, Dx, Dy))        
        grad_theta = -np.exp(theta) * lambdaaa * (avg_tv / Dw) + 1.0
        
        theta = theta + delta_step * grad_theta
        theta_values[k] = theta

        # Compute Performance Metrics
        original_reshape = original[:, :, None]
        err_aux = np.sum((w - original_reshape)**2, axis=(0, 1)) / np.sum(original**2)
        
        nmse_values[k] = np.mean(err_aux)
        mse_values[k] = np.mean((w - original_reshape)**2)
        ssim_values[k] = ssim(original, w[:, :, 0], data_range=w[:, :, 0].max() - w[:, :, 0].min())

    return w, nmse_values, theta_values, mse_values, ssim_values