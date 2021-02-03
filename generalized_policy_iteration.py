#%%
import numpy as np
from queue import PriorityQueue

import matplotlib.pyplot as plt
import seaborn as sns

#%% Custom type 
class StateReward:
  state:int    = 0
  reward:float = 0.0
  prob:float   = 0.0
  def __init__(self, s:int, r:float, p:float):
    self.state  = s
    self.reward = r
    self.prob   = p

#%% Setup grid and allowable actions
UP                = 0
LEFT              = 1
DOWN              = 2
RIGHT             = 3
grid_shape        = (5,  5)
goal_state_reward = [(3, 3), 0]

actions     = [UP, LEFT, DOWN, RIGHT]
grid        = np.arange(grid_shape[0]*grid_shape[1]).reshape(grid_shape)

default_policy = [0.25, 0.25, 0.25, 0.25]                            # default policy, every action is equally likely to be picked
gamma          = 1.0                                                 # discount factor. As this is an episodic task, no discount is necessary
policy         = np.ndarray(grid_shape, dtype=object)                # policy to map states to actions
state_value    = np.zeros(grid.size, dtype=np.float32)               # state-value function 

def get_trans_prob(old_state, new_state):
  if (old_state == new_state):
    return 1.0
  else:
    return 1.0

def get_reward(old_state, new_state, default_reward=-1.0):
  if (old_state == new_state):
    return 2*default_reward
  else:
    return default_reward

#%% Initialize grid with info of states they will transition into with certain action and rewards they will receive
augmented_grid = grid
augmented_grid = np.vstack((augmented_grid[0, :],                augmented_grid))
augmented_grid = np.vstack((augmented_grid,                      augmented_grid[-1, :]))
augmented_grid = np.hstack((augmented_grid[:, 0][:, np.newaxis], augmented_grid))
augmented_grid = np.hstack((augmented_grid,                      augmented_grid[:, -1][:, np.newaxis]))

transition_matrix = np.ndarray(grid.shape, dtype=object)
for i in range(1, augmented_grid.shape[0]-1):
  for j in range(1, augmented_grid.shape[1]-1):
    transition_matrix[i-1, j-1]              = [StateReward(0, -1.0, 1.0) for _ in actions]
    transition_matrix[i-1, j-1][UP].state    = augmented_grid[i-1, j]
    transition_matrix[i-1, j-1][UP].prob     = get_trans_prob(grid[i-1, j-1], augmented_grid[i-1, j])
    transition_matrix[i-1, j-1][UP].reward   = get_reward(grid[i-1, j-1], augmented_grid[i-1, j])

    transition_matrix[i-1, j-1][LEFT].state  = augmented_grid[i,   j-1]
    transition_matrix[i-1, j-1][LEFT].prob   = get_trans_prob(grid[i-1, j-1], augmented_grid[i, j-1])
    transition_matrix[i-1, j-1][LEFT].reward = get_reward(grid[i-1, j-1], augmented_grid[i, j-1])

    transition_matrix[i-1, j-1][DOWN].state  = augmented_grid[i+1, j]
    transition_matrix[i-1, j-1][DOWN].prob   = get_trans_prob(grid[i-1, j-1], augmented_grid[i+1, j])
    transition_matrix[i-1, j-1][DOWN].reward = get_reward(grid[i-1, j-1], augmented_grid[i+1, j])

    transition_matrix[i-1, j-1][RIGHT].state  = augmented_grid[i,   j+1]
    transition_matrix[i-1, j-1][RIGHT].prob   = get_trans_prob(grid[i-1, j-1], augmented_grid[i, j+1])
    transition_matrix[i-1, j-1][RIGHT].reward = get_reward(grid[i-1, j-1], augmented_grid[i, j+1])

# add goal state info
goal_row = goal_state_reward[0][0]
goal_col = goal_state_reward[0][1]
transition_matrix[goal_row-1, goal_col][DOWN].reward    = goal_state_reward[1]
transition_matrix[goal_row,   goal_col-1][RIGHT].reward = goal_state_reward[1]
transition_matrix[goal_row+1, goal_col][UP].reward      = goal_state_reward[1]
transition_matrix[goal_row,   goal_col+1][LEFT].reward  = goal_state_reward[1]

transition_matrix[goal_row, goal_col] = [StateReward(grid[goal_row, goal_col], 0.0, 0.0) for _ in actions]

# Initialize policy to default
for i in range(grid.shape[0]):
  for j in range(grid.shape[1]):
    policy[i, j] = default_policy
#%% Do Policy iteration
# Policy iteration is to use richardson iteration to estimate state-value as k->inf
# Once the state-value converges, perform policy improvement

policy_converged     = False
max_state_value_iter = 500
max_policy_iter      = 50

policy_iter = 0
while(not policy_converged and (policy_iter < max_policy_iter)):

  # perform iterative policy evaluation until convergence using current policy
  state_value_converged = False
  state_value_iter = 0
  state_value_diff = np.full(state_value.size, 1000.0)
  while(not state_value_converged):

    for i in range(grid.shape[0]):
      for j in range(grid.shape[1]):
        if (np.abs(state_value_diff[grid[i, j]]) < 1e-5 and state_value_iter > 5):
          continue
        temp = 0.0
        for a in actions:
          trans_state  = transition_matrix[i, j][a].state
          reward       = transition_matrix[i, j][a].reward
          state_prob   = transition_matrix[i, j][a].prob
          action_prob  = policy[i, j][a]
          temp        += action_prob*state_prob*(reward + gamma*state_value[trans_state])
        
        this_state = grid[i, j]
        state_value_diff[this_state] = temp - state_value[this_state]
        state_value[this_state] = temp

    n_converged_states = np.where(np.abs(state_value_diff) < 1e-5)[0]
    if ((n_converged_states.size == state_value.size) or (state_value_iter > max_state_value_iter)):
      state_value_converged = True
    state_value_iter += 1
  
  # perform policy improvement using the converged state-value function
  policy_stabilized = True
  for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
      old_policy   = policy[i, j]

      policy[i, j]     = [0.0]*len(actions) 
      best_action_list = []
      v_s = PriorityQueue(maxsize=len(actions))
      for a in actions:
        trans_state  = transition_matrix[i, j][a].state
        reward       = transition_matrix[i, j][a].reward
        state_prob   = transition_matrix[i, j][a].prob
        # negative as priority is min priority queue and we want max priority 
        v_s.put( (-state_prob*(reward + gamma*state_value[trans_state]), a) )
      
      best_action_list.append(v_s.get())
      while(not v_s.empty()):
        temp = v_s.get()
        if (np.abs(temp[0] - best_action_list[0][0]) < 1e-3):
          best_action_list.append(temp)

      action_prob = 1.0/len(best_action_list)
      for value, action in best_action_list:
        policy[i, j][action] = action_prob
      
      for old_prob, new_prob in zip(old_policy, policy[i, j]):
        if np.abs(old_prob - new_prob) > 0.1:
          policy_stabilized = False

  if (policy_stabilized):
    policy_converged = True
  
  policy_iter += 1
#%%
print("Policy from policy iteration method")
action_text = ['u', 'l', 'd', 'r']
for i in range(grid.shape[0]):
  for j in range(grid.shape[0]):
    best_policy_str = "("
    for a in actions:
      if (policy[i, j][a] > 0.0):
        best_policy_str = best_policy_str + f"{action_text[a]}, "
    
    best_policy_str = best_policy_str + ")"
    print(best_policy_str, " | ", end="")
  
  print("\n")
#%% Value iteration

max_state_value_iter = 500
state_value_converged = False
state_value_iter = 0
state_value      = np.zeros(grid.size, dtype=np.float32)

# Initialize policy to default
for i in range(grid.shape[0]):
  for j in range(grid.shape[1]):
    policy[i, j] = default_policy

# perform value iteration and evaluate current policy
# value iteration essentially performs both policy improvement and evaluation because of the max function used
while((not state_value_converged) and (state_value_iter < max_state_value_iter)):
  delta = 0
  for i in range(0, grid.shape[0]):
    for j in range(0, grid.shape[1]):
      this_iter_value = -1000.0
      for a in actions:
        trans_state  = transition_matrix[i, j][a].state
        reward       = transition_matrix[i, j][a].reward
        state_prob   = transition_matrix[i, j][a].prob
        action_prob  = policy[i, j][a]
        temp        = action_prob*state_prob*(reward + gamma*state_value[trans_state])
        if (temp > this_iter_value):
          this_iter_value = temp
        
      delta = max(np.abs(state_value[grid[i, j]]-this_iter_value), delta)
      state_value[grid[i,j]] = this_iter_value
  
  if (delta < 1e-3):
    state_value_converged = True
  state_value_iter+=1

# once the state value function is converged, pick the best optimal policy by being greedy
for i in range(grid.shape[0]):
  for j in range(grid.shape[1]):

    policy[i, j]     = [0.0]*len(actions) 
    best_action_list = []
    v_s = PriorityQueue(maxsize=len(actions))
    for a in actions:
      trans_state  = transition_matrix[i, j][a].state
      reward       = transition_matrix[i, j][a].reward
      state_prob   = transition_matrix[i, j][a].prob
      # negative as priority is min priority queue and we want max priority 
      v_s.put( (-state_prob*(reward + gamma*state_value[trans_state]), a) )
    
    best_action_list.append(v_s.get())
    while(not v_s.empty()):
      temp = v_s.get()
      if (np.abs(temp[0] - best_action_list[0][0]) < 1e-3):
        best_action_list.append(temp)

    action_prob = 1.0/len(best_action_list)
    for value, action in best_action_list:
      policy[i, j][action] = action_prob

#%% print optimal policy from value iteration
print("Policy from value iteration method")
action_text = ['u', 'l', 'd', 'r']
for i in range(grid.shape[0]):
  for j in range(grid.shape[0]):
    best_policy_str = "("
    for a in actions:
      if (policy[i, j][a] > 0.0):
        best_policy_str = best_policy_str + f"{action_text[a]}, "
    
    best_policy_str = best_policy_str + ")"
    print(best_policy_str, " | ", end="")
  
  print("\n")
# %%
