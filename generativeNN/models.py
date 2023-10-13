import torch
import torch.nn as nn
import torch.optim as optim

class TransitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=8):
        super(TransitionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x
    
class PosteriorModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=8):
        super(PosteriorModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x
    
class LikelihoodModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2):
        super(LikelihoodModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size


        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x

def sample(mean, log_variance):
    # Sample ε from a standard normal distribution
    epsilon = torch.randn_like(mean)
    
    # Reparameterization: s = μ + ε * σ, where σ = exp(0.5 * log_variance)
    std_deviation = torch.exp(0.5 * log_variance)
    sampled_value = mean + epsilon * std_deviation
    
    return sampled_value

def kl_divergence_gaussian(mu_p, log_var_p, mu_q, log_var_q):
    var_p = torch.exp(log_var_p) ^ 2
    var_q = torch.exp(log_var_q)

    k = mu_p.size(0)  # Dimensionality of the distribution

    term1 = torch.sum(var_q / var_p)
    term2 = torch.sum((mu_q - mu_p).pow(2) / var_p)
    term3 = k
    term4 = torch.sum(log_var_q - log_var_p)

    kl = 0.5 * (term1 + term2 - term3 + term4)
    return kl

def log_likelihood(observation, mean, log_variance):
    variance = torch.exp(log_variance)  # Convert log variance to variance
    log_lik = -0.5 * (torch.log(2 * torch.tensor([torch.pi], dtype=variance.dtype) * variance) + (observation - mean).pow(2) / variance)
    return log_lik

def freeEnergy(p_phi_mean, p_phi_logvar, p_theta_mean, p_theta_logvar, p_eps_mean, p_eps_logvar, obs):
    '''
    Calculate the free energy
    p_phi_mean, p_phi_var: the mean and variance of the prior distribution over the parameters of the transition model
    p_theta_mean, p_theta_var: the mean and variance of the prior distribution over the parameters of the posterior model
    p_eps_mean, p_eps_var: the mean and variance of the prior distribution over the parameters of the likelihood model
    '''
    kl_term = kl_divergence_gaussian(p_phi_mean, p_phi_logvar, p_theta_mean, p_theta_logvar)
    log_likelihood_term = log_likelihood(obs, p_eps_mean, p_eps_logvar)

    return kl_term - log_likelihood_term
    