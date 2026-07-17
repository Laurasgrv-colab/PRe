import numpy as np
import matplotlib.pyplot as plt

# banana target log-density
def log_banana(x, y):
    return(-((x**2)/10.0) - ((y**4)/10.0)- 2.0 * (y - x**2)**2)

# gradient of the log-density for MALA
def grad_log_banana(x,y):
    p1 = -x/5.0 + 8.0*x*(y-x**2)
    p2 = -(2.0*y**3)/5.0 - 4.0*(y-x**2)
    return np.array([p1, p2])

# --- Parameters ---
n = 50000          # iter nb
burnin = 10000    
sigma_q = 0.4      # step size for Random Walk
gamma = 0.05     # step size for MALA


samples_RW = np.zeros((2, n))
samples_Langevin = np.zeros((2, n)) # MALA

samples_RW[:, 0] = np.array([0.0, 0.0])
samples_Langevin[:, 0] = np.array([0.0, 0.0])


# 1. Random Walk Metropolis-Hastings (RWMH)
def r(x1, y1, x2, y2):
    logp2_ = log_banana(x2, y2)
    logp1_ = log_banana(x1, y1)
    return(logp2_ - logp1_)

for i in range(1, n):
    current = samples_RW[:, i-1]

    new = current + np.random.normal(0, sigma_q, size=2)
    # new = np.random.normal(current, sigma_q, size=2)

    # in MH, alpha = min(1, (p(x')*q(x|x')) / (p(x)*q(x'|x)) )
    # but symmetric --> q(x|x') == q(x'|x)
    # so alpha = p(new)/p(current)

    log_alpha = r(current[0], current[1], new[0], new[1])
    u = np.random.uniform(0,1)
    if np.log(u) < log_alpha:
        samples_RW[:, i] = new
    else:
        samples_RW[:, i] = current

# 2. MALA

for i in range(1,n):
    current = samples_Langevin[:, i-1]
    grad_curr = grad_log_banana(current[0], current[1])

    new = current + gamma*grad_curr + np.sqrt(2*gamma)*np.random.normal(0, 1, size = 2)
    # new = ... + np.sqrt(2 * tau) * np.random.normal(0, 1, size=2)
    grad_new = grad_log_banana(new[0], new[1])


    # MALA is not symmetric !!
    mu_curr = new + gamma*grad_new
    mu_new = current + gamma*grad_curr
    log_q_new_given_curr = -np.sum((new - mu_new)**2) / (4*gamma)
    log_q_curr_given_new = -np.sum((current - mu_curr)**2) / (4*gamma) 

    log_r = (log_banana(new[0], new[1]) + log_q_curr_given_new) - (log_banana(current[0], current[1]) + log_q_new_given_curr)
    
    u = np.random.uniform(0,1)
    if np.log(u) < log_r:
        samples_Langevin[:, i] = new
    else:
        samples_Langevin[:, i] = current



######### Given plot instructions
x_bb = np.linspace(-4, 4, 100)
y_bb = np.linspace(-2, 6, 100)
X_bb, Y_bb = np.meshgrid(x_bb, y_bb)
Z_bb = np.exp(log_banana(X_bb, Y_bb)) # your banana function

plt.subplot(1, 3, 1)
plt.contourf(X_bb, Y_bb, Z_bb, 100, cmap='RdBu')
plt.title('True Banana Density')

plt.subplot(1, 3, 2)
plt.hist2d(samples_RW[0, burnin:n], samples_RW[1, burnin:n], 100, cmap='RdBu', 
           range=[[-4, 4], [-2, 6]], density=True)
plt.title('Random Walk Metropolis')

plt.subplot(1, 3, 3)
plt.hist2d(samples_Langevin[0, burnin:n], samples_Langevin[1, burnin:n], 100, cmap='RdBu', 
            range=[[-4, 4], [-2, 6]], density=True)
plt.title('Metropolis Adjusted Langevin Algorithm')
plt.show()