import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from models import GenerativeModel
from torch.optim.lr_scheduler import StepLR

PATH = "graphs/"

def randomAgent(last_action):
   #10 percent chance of turning left or right
    if np.random.rand() < 0.1:
        #Pick 0 or 2S
        return int(np.random.choice([0, 2]))
    else:
        return last_action 

state_space = 4
action_space = 2
observation_space = 1
batch_size = 1
epochs = 5

if __name__ == "__main__":

    model = GenerativeModel(state_space, action_space, observation_space)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    #Collect data from random policy
    print("Collecting data...")
    env = gym.make('MountainCar-v0')
    observation, info = env.reset()
    eps = 12800 * 2
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
    env.close()

    losses = []
    kl = []
    nll = []
    print("Training...")


    for i in range(epochs):
        #Per batch in data
        for num, i in enumerate(range(0, len(data), batch_size)):

            batch_loss = torch.tensor([0.0])
            batch_kl = 0.0
            batch_nll = 0.0

            #Zero gradients
            optimizer.zero_grad()

            #Per episode in batch
            for idx in range(i, i + batch_size):

                state = torch.zeros(state_space)

                #Per step in episode
                for action, observation in ep:

                    s_tr, s_po, mu_tr, logvar_tr, mu_po, logvar_po = model.forward(state, action, observation)
                    loss = model.loss_function(s_tr, s_po, mu_tr, logvar_tr, mu_po, logvar_po, observation)

                    batch_loss += loss['loss']

                    state = model.reparameterize(mu_po, logvar_po)

                    batch_kl += loss['KLD'].item()
                    batch_nll += loss['NLL'].item()

                    #Detatch hidden states
                    state.detach_()

            #Backpropagate
            batch_loss /= batch_size
            batch_loss.backward()
            optimizer.step()

            losses.append(batch_loss.item())
            kl.append(batch_kl / batch_size)
            nll.append(batch_nll / batch_size)
                
            if num % 100 == 0:
                print(f"Batch: {num} of {len(data) // batch_size} | Loss: {batch_loss.item()}")

        scheduler.step()
            

    print("Evaluation...")

    #Evaluate ability to predict from state
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 200
    observation, info = env.reset()

    ground_truth = []
    posterior = []
    prior = []
    state = torch.zeros(state_space)
    for _ in range(200):
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

    #Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Loss")
    plt.savefig("losses.png")

    plt.figure(figsize=(10, 5))
    plt.plot(kl)
    plt.title("KL Divergence")
    plt.savefig("kl.png")

    plt.figure(figsize=(10, 5))
    plt.plot(nll)
    plt.title("Negative Log Likelihood")
    plt.savefig("nll.png")

    #Plot predictions
    plt.figure(figsize=(10, 5))
    plt.plot(ground_truth, label="Ground Truth")
    plt.plot(posterior, label="Posterior")
    plt.plot(prior, label="Prior")
    plt.legend()
    plt.title("Predictions")
    plt.savefig("predictions.png")

    torch.save(model, "model.pt")
