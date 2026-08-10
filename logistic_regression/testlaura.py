import os
import matplotlib.pyplot as plt
import numpy as np

# Prevent NumPy OpenMP/BLAS thread thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from joblib import Parallel, delayed

# --- 1. Pure Function Definition (Explicit Arguments) ---
def log_p_laplace(th, x, design_matrix, labels, b=1.0):
    logits = np.matmul(design_matrix, x)
    log_lik = np.sum(labels * logits - np.log(1 + np.exp(logits)))
    log_prior = np.sum(-np.abs(x - th) / b)
    return log_lik + log_prior


def soul_stretch_fast_decay(
    log_p,
    th0,
    x0_N,
    y_l,
    y_f,
    T,
    M,
    B,
    delta_step,
    a=2.0,
    b=1,
    gamma=1,
    verbose=False,
):
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
        current_delta = delta_step * (gamma**t)
        for m in range(1, M + 1):
            indices = np.random.permutation(N)
            S1_idx, S2_idx = indices[:half], indices[half:]

            for ens_idx, comp_idx in [(S1_idx, S2_idx), (S2_idx, S1_idx)]:
                X_ens = S[:, ens_idx]

                rand_comp_indices = np.random.choice(comp_idx, size=half)
                X_comp = S[:, rand_comp_indices]

                u_z = np.random.uniform(0.0, 1.0, size=(1, half))
                z = (
                    (a + (1.0 / a) - 2.0) * (u_z**2)
                    + 2.0 * u_z * (1.0 - (1.0 / a))
                    + (1.0 / a)
                )

                X_new = X_comp + z * (X_ens - X_comp)

                log_p_cur = np.array(
                    [log_p(th, X_ens[:, i : i + 1], y_l, y_f) for i in range(half)]
                )
                log_p_new = np.array(
                    [log_p(th, X_new[:, i : i + 1], y_l, y_f) for i in range(half)]
                )

                log_alpha = (
                    (D - 1) * np.log(z.reshape(half)) + log_p_new - log_p_cur
                )

                accept_mask = (
                    np.log(np.random.uniform(0.0, 1.0, size=half)) < log_alpha
                )
                X_ens[:, accept_mask] = X_new[:, accept_mask]

                S[:, ens_idx] = X_ens

            C[:, :, global_step] = S
            global_step += 1

        start = (t - 1) * M + 1
        current = C[:, :, start + B : global_step]
        flat_samples = current.reshape(D, -1)

        avg_grad_th = np.mean(np.sign(flat_samples - th)) / b

        th = th + current_delta * avg_grad_th

        # SILENCED DEFAULT: Prevents stdout deadlock in multiprocessing!
        if verbose and (t % 17 == 0):
            print("t=", t, "and theta = ", th)

        th_list.append(np.copy(th))

    return th_list, C


# --- 2. Worker Wrapper ---
def run_single_soul_stretch_decay(run_idx, design_matrix, labels, T, M, B, N, delta_step, b_scale):
    theta0_val = np.random.randint(-15, 10)
    th0 = np.array([[float(theta0_val)]])
    D = design_matrix.shape[1]
    X0_M = np.random.normal(loc=theta0_val, scale=1.0, size=(D, N))

    th_list, x_values = soul_stretch_fast_decay(
        log_p=log_p_laplace,
        th0=th0,
        x0_N=X0_M,
        y_l=design_matrix,
        y_f=labels,
        T=T,
        M=M,
        B=B,
        delta_step=delta_step,
        b=b_scale,
        gamma=0.99995,
        verbose=False,  # Set to False to keep Jupyter locks clean
    )
    th_trajectory = np.array([t[0, 0] for t in th_list])
    return run_idx, theta0_val, th_trajectory, x_values