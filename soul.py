def soul(theta_0, delta, gamma, K, B, X0_m, T):
    """ 
    theta_0: initial parameter
    delta >0: step size
    gamma >0: Langevin step size
    K: number of Langevin steps
    B: burn-in steps 
    X0_m
    T: total number of steps
    """
    for t in range(1, T):
        X0_t = X_m[t-1]
        for m in range(1, M):

    return 

import numpy as np

def soul_algorithm(theta_0, X_0_M, grad_log_p_x, grad_log_p_theta, y, T, M, B, delta, gamma):
    """
    theta_0 : Initial parameter vector
    X_0_M : Initial state/sample X_0^(M)
    grad_log_p_x : Function computing the gradient w.r.t x: f(theta, x, y) -> ndarray
    grad_log_p_theta : Function computing the gradient w.r.t theta: f(theta, x, y) -> ndarray
    y : The observed data/target
    T : Number of total  steps
    M : Number of Langevin steps per iteration
    B : Number of burn-in steps
    delta : Optimization step size (> 0)
    gamma : Langevin step size (> 0)
    """
    # Initialisation
    theta = np.copy(theta_0)
    X_curr = np.copy(X_0_M)
    
    noise = np.sqrt(2 * gamma)

    theta_samples = [theta]
    
    for t in range(1, T):
        X_samples = []
        
        for m in range(1, M):
            Z_k = np.random.normal(size=X_curr.shape)
            
            grad_x = grad_log_p_x(theta, X_curr, y)
            X_curr = X_curr + gamma * grad_x + noise * Z_k
            
            if m > B: # burnin (can be done later too)
                X_samples.append(np.copy(X_curr))
                
        # np.zeros_like() same shape and data type as the array passed to it        
        # sum_grad_theta = np.zeros_like(theta, dtype=float) 

        n = len(theta)
        sum_grad_theta = np.zeros(n)
        for X_m in X_samples:
            sum_grad_theta += grad_log_p_theta(theta, X_m, y)
            
        theta = theta + delta * sum_grad_theta / (M - B)
        theta_samples.append8(theta)

    return X_curr, theta_samples