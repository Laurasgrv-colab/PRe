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
