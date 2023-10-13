import gymnasium as gym
import torch.optim as optim
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from models import TransitionModel, PosteriorModel, LikelihoodModel, freeEnergy, sample

#State space is of size 4
#Observation space is only a single number (position)
#Action space is turn left, right, or do nothing
state_space = 4
observation_space = 1
action_space = 3

#Define the models
transition_model = TransitionModel(state_space + action_space, 20, 8)
posterior_model = PosteriorModel(state_space + action_space + observation_space, 20, 8)
likelihood_model = LikelihoodModel(state_space, 20, 2)
optimizer = optim.Adam(list(transition_model.parameters()) + list(posterior_model.parameters()) + list(likelihood_model.parameters()), lr=0.01)

#Collect data from random policy
print("Collecting data...")
env = gym.make('MountainCar-v0')
observation, info = env.reset()
eps = 320
data = []
for i in range(eps):
    ep = []
    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        one_hot_action = F.one_hot(torch.tensor(action), num_classes=3).view(1, 3)
        observation = torch.tensor(observation).view(1, 2)
        ep.append((one_hot_action, observation))

        if terminated or truncated:
            observation, info = env.reset()
    data.append(ep)
    observation, info = env.reset()
env.close()

print("Training...")

transition_model.train()
posterior_model.train()
likelihood_model.train()

batch_size = 32
epochs = 1

#Train the models
for j in range(epochs):
    print("======EPOCH {}======".format(j))

    #Shuffle data using numpy
    np.random.shuffle(data)
    optimizer.zero_grad()

    for i in range(len(data)):

        state = torch.zeros(1, state_space)
        loss = torch.Tensor(1, 1).fill_(0.0)

        for action, obs in data[i]:
            obs = obs[:, 0].unsqueeze(1)
            transition_output = transition_model(torch.cat([state, action], dim=1))
            posterior_output = posterior_model(torch.cat([state, action, obs], dim=1))
            likelihood_output = likelihood_model(state)

            obs_sample = sample(likelihood_output[:, 0], likelihood_output[:, 1])
            state_sample = sample(posterior_output[:, :4], posterior_output[:, 4:]) * torch.randn(1, 4)

            loss += freeEnergy(transition_output[:, :4], transition_output[:, 4:], posterior_output[:, :4], posterior_output[:, 4:], likelihood_output[:, 0], likelihood_output[:, 1], obs)
            state = state_sample

        if i % batch_size == 0 and i != 0:
            optimizer.zero_grad()
            loss = loss / batch_size
            loss.backward()
            optimizer.step()
            print(loss.item())
            loss = torch.Tensor(1, 1).fill_(0.0)

print("Evaluation...")

#Evaluate ability to predict from state
env = gym.make('MountainCar-v0')
env._max_episode_steps = 1000
observation, info = env.reset()

ground_truth = []
posterior = []
prior = []
state = torch.zeros(1, state_space)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    action = F.one_hot(torch.tensor(action), num_classes=3).view(1, 3)
    observation = torch.tensor(observation).view(1, 2)
    observation = observation[:, 0].unsqueeze(1)

    transition_output = transition_model(torch.cat([state, action], dim=1))
    posterior_output = posterior_model(torch.cat([state, action, obs], dim=1))

    prior_sample = sample(transition_output[:, :4], transition_output[:, 4:]) * torch.randn(1, 4)
    posterior_sample = sample(posterior_output[:, :4], posterior_output[:, 4:]) * torch.randn(1, 4)

    prior_obs = likelihood_model(prior_sample)
    prior_obs = sample(prior_obs[:, 0], prior_obs[:, 1])
    posterior_obs = likelihood_model(posterior_sample)
    posterior_obs = sample(posterior_obs[:, 0], posterior_obs[:, 1])

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