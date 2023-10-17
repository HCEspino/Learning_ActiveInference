import torch
import torch.nn as nn
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import Tensor

class GenerativeModel(nn.Module):
    def __init__(self, state_size, action_space, observation_space, hidden_size=20):
        super(GenerativeModel, self).__init__()
        self.state_size = state_size
        self.action_space = action_space
        self.observation_space = observation_space

        self.transition_model = nn.Sequential(
            nn.Linear(state_size + action_space, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        self.transition_mean = nn.Linear(hidden_size, state_size)
        self.transition_var = nn.Linear(hidden_size, state_size)

        self.posterior_model = nn.Sequential(
            nn.Linear(state_size + action_space + observation_space, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        self.posterior_mean = nn.Linear(hidden_size, state_size)
        self.posterior_var = nn.Linear(hidden_size, state_size)

        self.likelihood_model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        self.likelihood_mean = nn.Linear(hidden_size, observation_space)
        self.likelihood_var = nn.Linear(hidden_size, observation_space)

    def encode(self, state: Tensor, action: Tensor, observation: Tensor) -> List[Tensor]:
        x_tr = torch.cat([state, action], dim=0)
        x_tr = self.transition_model(x_tr)
        x_tr = torch.flatten(x_tr, start_dim=0)
        mu_tr = self.transition_mean(x_tr)
        logvar_tr = self.transition_var(x_tr)

        x_po = torch.cat([state, action, observation], dim=0)
        x_po = self.posterior_model(x_po)
        x_po = torch.flatten(x_po, start_dim=0)
        mu_po = self.posterior_mean(x_po)
        logvar_po = self.posterior_var(x_po)

        return [mu_tr, logvar_tr, mu_po, logvar_po]
    
    def decode(self, state: Tensor) -> List[Tensor]:
        x = self.likelihood_model(state)
        x = torch.flatten(x, start_dim=0)
        mu = self.likelihood_mean(x)
        logvar = self.likelihood_var(x)
        return [mu, logvar]
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, state: Tensor, action: Tensor, obs: Tensor, **kwargs) -> List[Tensor]:
        '''
        Forward pass through the model
            s_tr: mu and logvar of observation reconstruction from transition model
            s_po: mu and logvar of observation reconstruction from posterior model
            mu_tr: mu of transition state encoding
            logvar_tr: logvar of transition state encoding
            mu_po: mu of posterior state encoding
            logvar_po: logvar of posterior state encoding
        '''
        #Get encoding
        mu_tr, logvar_tr, mu_po, logvar_po = self.encode(state, action, obs)
        s_tr = self.reparameterize(mu_tr, logvar_tr)
        s_po = self.reparameterize(mu_po, logvar_po)    

        return [self.decode(s_tr), self.decode(s_po), mu_tr, logvar_tr, mu_po, logvar_po]
    
    def loss_function(self, *args, **kwargs) -> dict:
        #mu/logvar of transition and posterior reconstruction (from decoding)
        tr_obs_mu = args[0][0]
        tr_obs_logvar = args[0][1]
        po_obs_mu = args[1][0]
        po_obs_logvar = args[1][1]

        #mu/logvar of transition and posterior encoding
        mu_tr, logvar_tr, mu_po, logvar_po = args[2:6]

        #observation
        obs = args[6]

        KLD = 0.5 * (logvar_tr - logvar_po + (torch.exp(logvar_po) + (mu_po - mu_tr).pow(2)) / torch.exp(logvar_tr) - 1)
        KLD = KLD.sum()

        NLL = 0.5 * (2.0 * po_obs_logvar + ((obs - po_obs_mu)**2.0 / (torch.exp(2.0 * po_obs_logvar))))

        loss = torch.tensor(0.0, requires_grad=True)
        loss = KLD + NLL

        return {'loss': loss, 'KLD': KLD, 'NLL': NLL}