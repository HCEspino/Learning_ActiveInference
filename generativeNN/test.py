import torch

def kl_divergence_gaussian(mu_p, log_var_p, mu_q, log_var_q):
    var_p = torch.exp(log_var_p)
    var_q = torch.exp(log_var_q)

    k = mu_p.size(0)  # Dimensionality of the distribution

    term1 = torch.sum(var_q / var_p)
    term2 = torch.sum((mu_q - mu_p).pow(2) / var_p)
    term3 = k
    term4 = torch.sum(log_var_q - log_var_p)

    kl = 0.5 * (term1 + term2 - term3 + term4)
    return kl
'''
# Example usage:
mu_p = torch.tensor([1.0, 2.0, 3.0])
log_var_p = torch.tensor([0.1, 0.2, 0.3])
mu_q = torch.tensor([10.0, 2.0, 3.0])
log_var_q = torch.tensor([0.15, 0.25, 0.35])

kl = kl_divergence_gaussian(mu_p, log_var_p, mu_q, log_var_q)
print("KL Divergence:", kl.item())
'''

import numpy as np

def log_likelihood(observation, mean, variance):
    nll = -0.5 * (np.log(2 * np.pi * variance) + ((observation - mean) ** 2) / variance)
    return nll

obs = 11.0
mean = 11.0
variance = 0.01

print(log_likelihood(obs, mean, variance))