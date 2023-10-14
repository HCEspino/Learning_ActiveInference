import gymnasium as gym
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.kl import kl_divergence
from models import TransitionModel, PosteriorModel, LikelihoodModel

def randomAgent(last_action):
   #10 percent chance of turning left or right
    if np.random.rand() < 0.1:
        #Pick 0 or 2
        return int(np.random.choice([0, 2]))
    else:
        return last_action 


state_space = 4
observation_space = 1
action_space = 3

#Load models
transition_model = torch.load("transition_model.pt")
posterior_model = torch.load("posterior_model.pt")
likelihood_model = torch.load("likelihood_model.pt")

ground_truth = []
posterior = []
prior = []
state = torch.zeros(1, state_space)

#Evaluate ability to predict from state
env = gym.make('MountainCar-v0')
env._max_episode_steps = 200
observation, info = env.reset()
last_action = 1

for _ in range(200):
    action = randomAgent(last_action)
    last_action = action
    observation, reward, terminated, truncated, info = env.step(action)

    action = F.one_hot(torch.tensor(action), num_classes=3).view(1, 3)
    observation = torch.tensor(observation).view(1, 2)
    observation = observation[:, 0].unsqueeze(1)

    #Sample from prior
    transition_output = transition_model(torch.cat([state, action], dim=1))
    transition_output = MultivariateNormal(transition_output[:, :state_space], torch.diag(torch.exp(transition_output[:, state_space:][0])))
    prior_sample = transition_output.sample()

    #Sample from posterior
    posterior_output = posterior_model(torch.cat([state, action, observation], dim=1))
    posterior_output = MultivariateNormal(posterior_output[:, :state_space], torch.diag(torch.exp(posterior_output[:, state_space:][0])))
    posterior_sample = posterior_output.sample()

    #Use states to predict observation
    prior_obs = likelihood_model(prior_sample)
    prior_obs = Normal(prior_obs[:, 0], torch.exp(prior_obs[:, 1]))
    prior_obs = prior_obs.sample()

    posterior_obs = likelihood_model(posterior_sample)
    posterior_obs = Normal(posterior_obs[:, 0], torch.exp(posterior_obs[:, 1]))
    posterior_obs = posterior_obs.sample()

    ground_truth.append(observation.item())
    posterior.append(posterior_obs.item())
    prior.append(prior_obs.item())

    state = posterior_sample

    if terminated or truncated:
        observation, info = env.reset()
env.close()


#Plot
plt.figure()
plt.ylim(min(ground_truth) - 0.1, max(ground_truth) + 0.1)
plt.plot(ground_truth, label="Ground Truth")
plt.plot(posterior, label="Posterior")
plt.plot(prior, label="Prior")
plt.legend()
plt.show()