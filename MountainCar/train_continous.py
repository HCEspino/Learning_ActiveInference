import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from models import GenerativeModel
from torch.optim.lr_scheduler import StepLR
import signal
import sys

PATH = "graphs/"

def randomAgent(last_action):
   #10 percent chance of turning left or right
    if np.random.rand() < 0.1:
        #Pick 0 or 2S
        return int(np.random.choice([0, 2]))
    else:
        return last_action 

def handler(signum, frame):
    print("Force quit")
    env.close()
    torch.save(model, "model.pt")

    sys.exit(1)

state_space = 4
action_space = 2
observation_space = 1
batch_size = 1
epochs = 5
env = gym.make('MountainCar-v0')


model = GenerativeModel(state_space, action_space, observation_space)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
batchnum = 1
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)

while(True):
    #Collect a batch of data
    observation, info = env.reset()
    eps = 32
    data = []
    last_action = int(np.random.choice([0, 2]))
    for i in range(eps):
        ep = []
        for _ in range(100):
            action = randomAgent(last_action)
            last_action = action

            observation, reward, terminated, truncated, info = env.step(action)
            if action == 0:
                one_hot_action = torch.tensor([1.0, 0.0]).view(2)
            else:
                one_hot_action = torch.tensor([0.0, 1.0]).view(2)

            one_hot_action.requires_grad = True
            observation = torch.tensor([observation[0]], requires_grad=True)
            ep.append((one_hot_action, observation))

            if terminated or truncated:
                observation, info = env.reset()
        data.append(ep)
        observation, info = env.reset()

    #Train on batch
    batch_loss = torch.tensor([0.0])
    batch_kl = 0.0
    batch_nll = 0.0
    optimizer.zero_grad()
    for i in range(len(data)):
        state = torch.zeros(state_space)
        #Per step in episode
        for action, observation in data[i]:

            s_tr, s_po, mu_tr, logvar_tr, mu_po, logvar_po = model.forward(state, action, observation)
            loss = model.loss_function(s_tr, s_po, mu_tr, logvar_tr, mu_po, logvar_po, observation)

            batch_loss += loss['loss']

            state = model.reparameterize(mu_po, logvar_po)

            batch_kl += loss['KLD'].item()
            batch_nll += loss['NLL'].item()

            #Detatch hidden states
            state.detach_()

    #Backpropagate
    batch_loss /= eps
    batch_loss.backward()
    optimizer.step()

    print(f"Batch {batchnum} | Loss: {batch_loss.item()}")

    if batchnum % 100 == 0:
        torch.save(model, f"chkpt{batchnum}.pt")
        print(f"Model saved at {batchnum} batches")

    batchnum += 1
