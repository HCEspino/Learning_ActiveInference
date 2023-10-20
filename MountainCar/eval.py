import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

def randomAgent(last_action):
   #10 percent chance of turning left or right
    if np.random.rand() < 0.1:
        #Pick 0 or 2S
        return float(np.random.choice([-1.0, 1.0]))
    else:
        return last_action 

model = torch.load("bestmodel1/chkpt3500.pt")
state_space = 4
action_space = 1
observation_space = 1

print("Evaluation...")

#Evaluate ability to predict from state
env = gym.make('MountainCarContinuous-v0')
env._max_episode_steps = 200
observation, info = env.reset()
last_action = float(np.random.choice([-1.0, 1,0]))
ground_truth = []
posterior = []
prior = []
state = torch.zeros(state_space)
for _ in range(200):
    action = randomAgent(last_action)
    last_action = action
    observation, reward, terminated, truncated, info = env.step([action])

    action = torch.tensor([last_action]).view(1)

    observation = torch.tensor([observation[0]])

    s_tr, s_po, mu_tr, logvar_tr, mu_po, logvar_po = model.forward(state, action, observation)

    transition_gen = model.reparameterize(s_tr[0], s_tr[1])
    posterior_gen = model.reparameterize(s_po[0], s_po[1])

    ground_truth.append(observation.item())
    posterior.append(posterior_gen.item())
    prior.append(transition_gen.item())

    state = model.reparameterize(mu_po, logvar_po)

    if terminated or truncated:
        observation, info = env.reset()
        break
env.close()

#Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(ground_truth, label="Ground Truth")
plt.plot(posterior, label="Posterior")
plt.plot(prior, label="Prior")
plt.legend()
plt.title("Predictions")
plt.savefig("predictions.png")