import torch
import torch.nn as nn
import torch.optim as optim
import math

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

def sample(mean, log_std):
    std_dev = torch.exp(log_std)
    epsilon = torch.randn_like(mean)
    return mean + epsilon * std_dev


def kl_divergence_gaussian(p_mean, p_logstd, q_mean, q_logstd):

    p_std = torch.exp(p_logstd)
    q_std = torch.exp(q_logstd)

    print(p_std, q_std)

    return 0

def log_likelihood(obs, mean, log_std):
    std = torch.exp(log_std)
    log_likelihood = -0.5 * torch.log(2*torch.tensor(torch.pi)) - 0.5 * torch.log(std**2) - ((1 / (2*std**2))*(obs - mean)**2)
    return log_likelihood

def freeEnergy(theta_mean, theta_logstd, phi_mean, phi_logstd, eps_mean, eps_logstd, obs):
    '''
    Calculate the free energy
        p_phi_mean, p_phi_var: the mean and variance of the prior distribution over the parameters of the transition model
        p_theta_mean, p_theta_var: the mean and variance of the prior distribution over the parameters of the posterior model
        p_eps_mean, p_eps_var: the mean and variance of the prior distribution over the parameters of the likelihood model
    '''
    kl_term = kl_divergence_gaussian(phi_mean, phi_logstd, theta_mean, theta_logstd)
    log_likelihood_term = log_likelihood(obs, eps_mean, eps_logstd)
    return kl_term - log_likelihood_term
    