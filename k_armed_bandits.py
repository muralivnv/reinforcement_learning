#%%
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
#%% setup number of actions and reward function for each action
n_actions   = 4

# (mean, variance)
reward_dist = [(5.6, 3.2), (4.3, 1.3), (1.0, 0.15), (6.0, 4.5)]

#%% sample average method to estimate action value function
n_runs               = 1
n_iter               = 2000
estim_action_rewards = [0.0]*n_actions
action_counter       = [1]*n_actions
estim_reward_hist    = [[] for i in range(n_actions)]
total_reward         = [0.0]*n_iter
for i in range(n_runs):
  for j in range(1, n_iter):

    # take the action that has the maximum reward at this instant
    action = np.argmax(estim_action_rewards)
    action_counter[action] += 1
    reward = reward_dist[action][0] + np.random.randn(1)[0]*(reward_dist[action][1])
    estim_action_rewards[action] += (reward - estim_action_rewards[action])/(action_counter[action])
    estim_reward_hist[action].append(estim_action_rewards[action])
    total_reward[j] = total_reward[j-1] + reward

plt.figure(figsize=(12,7))
for i in range(n_actions):
  plt.plot(estim_reward_hist[i], '-', linewidth=1)
plt.grid(True)
plt.xlabel("Iterations", fontsize=12)
plt.legend([i for i in range(n_actions)])
plt.title("Action estimated value function", fontsize=14)
plt.show()

#%% epsilon-greedy method
epsilon = 0.1
n_runs               = 1
n_iter               = 2000
estim_action_rewards = [0.0]*n_actions
action_counter       = [1]*n_actions
estim_reward_hist    = [[] for i in range(n_actions)]
total_reward         = [0.0]*n_iter
for i in range(n_runs):
  for j in range(1, n_iter):

    # roll a dice and see which one to pick
    dice = np.random.uniform(0.0, 1.0)
    if (dice < (1.0 - epsilon)):
      action = np.argmax(estim_action_rewards)
    else:
      action_best = np.argmax(estim_action_rewards)
      action = int(np.random.uniform(0.0, 1.0)*n_actions)
      if (action == action_best):
        if (action > 0):
          action -= 1
        else:
          action += 1

    reward = reward_dist[action][0] + np.random.randn(1)[0]*(reward_dist[action][1])
    action_counter[action] += 1
    estim_action_rewards[action] += (reward - estim_action_rewards[action])/(action_counter[action])
    estim_reward_hist[action].append(estim_action_rewards[action])

    total_reward[j] = total_reward[j-1] + reward

plt.figure(figsize=(12,7))
for i in range(n_actions):
  plt.plot(estim_reward_hist[i], '-', linewidth=1)
plt.grid(True)
plt.xlabel("Iterations", fontsize=12)
plt.legend([i for i in range(n_actions)])
plt.title("Action estimated value function", fontsize=14)
plt.show()

#%% upper-confidence bound action selection
# this takes into account action value uncertainties
c = 15.0
n_runs               = 1
n_iter               = 2000
estim_action_rewards = [0.0]*n_actions
action_counter       = [1]*n_actions
estim_reward_hist    = [[] for i in range(n_actions)]

for i in range(n_runs):
  for j in range(1, n_iter):
    rewards_w_uncertainty = [0.0]*n_actions
    for action in range(n_actions):
      rewards_w_uncertainty[action] = estim_action_rewards[action]
      rewards_w_uncertainty[action] += c*np.sqrt(np.log(j)/action_counter[action])

    best_action = np.argmax(rewards_w_uncertainty)

    reward = reward_dist[best_action][0] + np.random.randn(1)[0]*(reward_dist[best_action][1])
    action_counter[best_action] += 1
    estim_action_rewards[best_action] += (reward - estim_action_rewards[best_action])/(action_counter[best_action])
    estim_reward_hist[best_action].append(estim_action_rewards[best_action])

plt.figure(figsize=(12,7))
for i in range(n_actions):
  plt.plot(estim_reward_hist[i], '-', linewidth=1)
plt.grid(True)
plt.xlabel("Iterations", fontsize=12)
plt.legend([i for i in range(n_actions)])
plt.title("Action estimated value function", fontsize=14)
plt.show()

#%% gradient bandit
