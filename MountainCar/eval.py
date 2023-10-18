import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from train import randomAgent, state_space

model = torch.load("C:\\Users\\Harrison\\Documents\\GitHub\\Learning_ActiveInference\\MountainCar\\t1\\model.pt")


print("Evaluation...")

#Evaluate ability to predict from state
env = gym.make('MountainCar-v0')
env._max_episode_steps = 200
observation, info = env.reset()
last_action = 1
ground_truth = []
posterior = []
prior = []
state = torch.zeros(state_space)
for _ in range(100):
    action = randomAgent(last_action)
    last_action = action
    observation, reward, terminated, truncated, info = env.step(action)

    if action == 0:
        action = torch.tensor([1.0, 0.0]).view(2)
    else:
        action = torch.tensor([0.0, 1.0]).view(2)

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