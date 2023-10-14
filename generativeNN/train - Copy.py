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


#if torch.cuda.is_available():
#    device = torch.device("cuda")
#else:
#    device = torch.device("cpu")
device = torch.device("cpu")
print("Using device: {}".format(device))


#State space is of size 4
#Observation space is only a single number (position)
#Action space is turn left, right, or do nothing
state_space = 4
observation_space = 1
action_space = 3

#Define the models
transition_model = TransitionModel(state_space + action_space, 20, 8).to(device)
posterior_model = PosteriorModel(state_space + action_space + observation_space, 20, 8).to(device)
likelihood_model = LikelihoodModel(state_space, 20, 2).to(device)
optimizer = optim.Adam(list(transition_model.parameters()) + list(posterior_model.parameters()) + list(likelihood_model.parameters()), lr=0.001)

#Collect data from random policy
print("Collecting data...")
env = gym.make('MountainCar-v0')
observation, info = env.reset()
eps = 640
data = []
last_action = 1
for i in range(eps):
    ep = []
    for _ in range(100):
        action = randomAgent(last_action)
        last_action = action

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
epochs = 10

bk = False

losses = []

#Train the models
for j in range(epochs):
    print("======EPOCH {}======".format(j + 1))

    #Shuffle data using numpy
    np.random.shuffle(data)
    optimizer.zero_grad()
    loss = torch.Tensor(1, 1).fill_(0.0).to(device)

    for i in range(len(data)):

        state = torch.zeros(1, state_space).to(device)

        #Pairs are (at-1, ot)
        for action, obs in data[i]:
            obs = obs[:, 0].unsqueeze(1)
            action = action.to(device)
            obs = obs.to(device)

            #Get Prior P(st | st-1, at-1)
            transition_output = transition_model(torch.cat([state, action], dim=1).to(device))
            transition_output = MultivariateNormal(transition_output[:, :state_space], torch.diag(torch.exp(transition_output[:, state_space:][0])))

            #and posterior Q(st | st-1, at-1, ot)
            posterior_output = posterior_model(torch.cat([state, action, obs], dim=1).to(device))
            posterior_output = MultivariateNormal(posterior_output[:, :state_space], torch.diag(torch.exp(posterior_output[:, state_space:][0])))
            state_sample = posterior_output.sample()

            #Sample prior and use to get likelihood of observation
            likelihood_output = likelihood_model(state_sample)
            likelihood_output = Normal(likelihood_output[:, 0], torch.exp(likelihood_output[:, 1]))

            #Calculate free energy
            loss += (kl_divergence(posterior_output, transition_output) - likelihood_output.log_prob(obs))

            #Update state
            state = state_sample


        if i % batch_size == 0 and i != 0:
            #loss = loss / batch_size
            optimizer.step()
            print(f"EPOCH {j+1}/{epochs} | {i // batch_size}/{len(data) // batch_size} | {loss.item()}")
            optimizer.zero_grad()
            losses.append(loss.item())
            loss = torch.Tensor(1, 1).fill_(0.0).to(device)

print("Evaluation...")

#Evaluate ability to predict from state
env = gym.make('MountainCar-v0')
env._max_episode_steps = 1000
observation, info = env.reset()

ground_truth = []
posterior = []
prior = []
state = torch.zeros(1, state_space)
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

torch.save(transition_model, "transition_model.pt")
torch.save(posterior_model, "posterior_model.pt")
torch.save(likelihood_model, "likelihood_model.pt")

#Plot
plt.figure()
plt.ylim(min(ground_truth) - 0.1, max(ground_truth) + 0.1)
plt.plot(ground_truth, label="Ground Truth")
plt.plot(posterior, label="Posterior")
plt.plot(prior, label="Prior")
plt.legend()
plt.savefig("results.png")
plt.show()

#Plot losses
plt.figure()
plt.plot(losses)
plt.title("Loss")
plt.savefig("losses.png")
plt.show()