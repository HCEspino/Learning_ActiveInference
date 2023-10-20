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
        return float(np.random.choice([-1.0, 1.0]))
    else:
        return last_action 

def handler(signum, frame):
    print("Force quit. Saving model...")
    env.close()
    torch.save(model, "model.pt")

    sys.exit(1)

state_space = 4
action_space = 1
observation_space = 1
env = gym.make('MountainCarContinuous-v0')


model = GenerativeModel(state_space, action_space, observation_space, neg_slope=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
batchnum = 1
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)

while(True):
    #Collect a batch of data
    observation, info = env.reset()
    eps = 32
    data = []
    last_action = float(np.random.choice([-1.0, 1,0]))
    env = gym.make('MountainCarContinuous-v0')
    env.reset()
    for i in range(eps):
        ep = []
        for _ in range(100):
            action = randomAgent(last_action)
            last_action = action

            observation, reward, terminated, truncated, info = env.step([action])
            tensor_action = torch.tensor([last_action]).view(1)
            tensor_action.requires_grad = True
            observation = torch.tensor([observation[0]], requires_grad=True)
            ep.append((tensor_action, observation))

            if terminated or truncated:
                observation, info = env.reset()
        data.append(ep)
        observation, info = env.reset()
    env.close()

    #Train on individual episodes
    batch_loss = torch.tensor([0.0])
    for i in range(len(data)):
        state = torch.zeros(state_space)
        episode_loss = torch.tensor([0.0])
        #Per step in episode
        for action, observation in data[i]:

            s_tr, s_po, mu_tr, var_tr, mu_po, var_po = model.forward(state, action, observation)
            loss = model.loss_function(s_tr, s_po, mu_tr, var_tr, mu_po, var_po, observation)

            batch_loss += loss['loss']
            #episode_loss += loss['loss']

            state = model.reparameterize(mu_po, var_po)

            #Detatch hidden states
            state.detach_()

        #episode_loss.backward()
        #optimizer.step()
        #optimizer.zero_grad()
        #batch_loss += episode_loss

    batch_loss /= eps
    print(f"Batch {batchnum} | Loss: {batch_loss.item()}")
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batchnum % 100 == 0:
        torch.save(model, f"chkpt{batchnum}.pt")
        print(f"Model saved at {batchnum} batches")

    batchnum += 1
