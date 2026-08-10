import numpy as np

def log_p_laplace_batched(th, X, design_matrix, labels, b=1.0):
    """
    Batched calculation: X has shape (D, half).
    Returns shape (half,).
    """
    # Force th to 2D column vector if given as 1D array to avoid broadcast bugs
    if th.ndim == 1:
        th = th[:, None]
        
    logits = design_matrix @ X                           # (n_obs, D) @ (D, half) -> (n_obs, half)
    log_lik = np.sum(
        labels[:, None] * logits - np.logaddexp(0.0, logits),
        axis=0
    )                                                    # (half,)

    log_prior = -np.sum(np.abs(X - th), axis=0) / b       # (half,)

    return log_lik + log_prior                           # (half,)


def soul_stretch_fast_decay_v2(log_p, th0, x0_N, y_l, y_f, T, M, B, delta_step, a=2.0, b=1, gamma=1.0):
    th = np.copy(th0)
    D, N = x0_N.shape
    half = N // 2
    S = np.copy(x0_N)

    th_list = [np.copy(th)]

    for t in range(1, T + 1):
        current_delta = delta_step * (gamma ** t)

        # Iteration sample buffer
        C_iter = np.zeros((D, N, M + 1))
        C_iter[:, :, 0] = S

        for m in range(1, M + 1):
            indices = np.random.permutation(N)
            S1_idx, S2_idx = indices[:half], indices[half:]

            for ens_idx, comp_idx in [(S1_idx, S2_idx), (S2_idx, S1_idx)]:
                X_ens = S[:, ens_idx]
                rand_comp_indices = np.random.choice(comp_idx, size=half)
                X_comp = S[:, rand_comp_indices]

                # Sample scaling factor z
                u_z = np.random.uniform(0.0, 1.0, size=(1, half))
                z = (a + (1.0 / a) - 2.0) * (u_z ** 2) + 2.0 * u_z * (1.0 - (1.0 / a)) + (1.0 / a)
                
                # Stretch move proposal
                X_new = X_comp + z * (X_ens - X_comp)

                # Fully batched evaluation
                log_p_cur = log_p(th, X_ens, y_l, y_f)   # shape (half,)
                log_p_new = log_p(th, X_new, y_l, y_f)   # shape (half,)

                log_alpha = (D - 1) * np.log(z.reshape(half)) + log_p_new - log_p_cur
                accept_mask = np.log(np.random.uniform(0.0, 1.0, size=half)) < log_alpha
                
                X_ens[:, accept_mask] = X_new[:, accept_mask]
                S[:, ens_idx] = X_ens

            C_iter[:, :, m] = S

        # Extract non-burn-in samples for gradient calculation
        flat_samples = C_iter[:, :, B:].reshape(D, -1)
        
        # Keep shape consistency for th vector update across arbitrary input dimensions
        th_col = th[:, None] if th.ndim == 1 else th
        avg_grad_th = np.mean(np.sign(flat_samples - th_col), axis=1) / b
        
        if th.ndim == 1:
            th = th + current_delta * avg_grad_th
        else:
            th = th + current_delta * avg_grad_th[:, None]

        th_list.append(np.copy(th))

    return th_list