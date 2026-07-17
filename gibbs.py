import numpy as np

# def p(k, n): 

def gibbs(n, x0):
    m = len(x0)
    samples = np.zeros(m, n)
    for i in range(1,n):
        x_n = samples[i-1]
        for k in range(m):
            x_n[k] = p(k, x_n) # update x_n each round 
        samples[i] = x_n 
    return samples

def gibbs_random(n, x0):
    m = len(x0)
    samples = np.zeros(m, n)
    for i in range(1,n):
        x_n = samples[i-1]
        set = [k for k in range(m)]
        np.random.shuffle(set)
        for k in set:
            x_n[k] = p(k, x_n) # update x_n each round 
        samples[i] = x_n 
    return samples

# need to define p and q
def gibbs_mh(z0, s0, y, n):
    samples_z = np.zeros(n)
    samples_s = np.zeros(n)
    for k in range(1, n):
        z_k = samples_z[k-1]
        s_k = samples_s[k-1]

        z_new = q(z_k)
        r_z = p(z_new, s_k, y)*q(z_k) / (p(z_k, s_k, y)*q(z_new)) # the q might simplifye if symmetric
        u = np.random.uniform(0,1)
        if u<r_z:
            samples_z[k] = z_new
        else:
            samples_z[k] = z_k

        # now for s
        s_new = q(s_k)
        r_s = p(s_new, z_k, y)*q(s_k) / (p(s_k, z_k, y)*q(s_new)) # the q might simplifye if symmetric
        u = np.random.uniform(0,1)
        if u<r_s:
            samples_s[k] = s_new
        else:
            samples_s[k] = s_k

    return samples_z, samples_s