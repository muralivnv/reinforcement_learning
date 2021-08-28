# Fully Observable Upper Confidence Tree / Fully Observable Monte-Carlo Tree Search with Upper Confidence Bound
#%%
import numpy as np
from scipy import interpolate

import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# custom libs
import config as cfg
import object_maintenance as obj_mntn
import mcts as mcts

#%% setup environment objects initial condition

# define object initial state
# objects = [ obj_mntn.ObjectState(x=50.0, y=5.0, vx=0.0, vy=-0.5), 
#             obj_mntn.ObjectState(x=51.0, y=5.5, vx=0.0, vy=-0.45),
#             obj_mntn.ObjectState(x=52.0, y=5.2, vx=0.0, vy=-0.35)
#           ]
objects = [ obj_mntn.ObjectState(x=50.0, y=-50.0, vx=0.0, vy=3.0),
            obj_mntn.ObjectState(x=53.0, y=-52.0, vx=0.0, vy=3.0),
            obj_mntn.ObjectState(x=60.0, y=60.0, vx=0.0, vy=-3.0),
            obj_mntn.ObjectState(x=62.0, y=60.0, vx=0.0, vy=-3.0),
            obj_mntn.ObjectState(x=20.0, y=-5.0, vx=0.5, vy=0.5),
            obj_mntn.ObjectState(x=20.3, y=-5.0, vx=0.5, vy=0.5),
            obj_mntn.ObjectState(x=20.5, y=-5.0, vx=0.5, vy=0.5)
          ]
ego = obj_mntn.ObjectState(x=0.0, y=0.0, vx=5.0, vy=0.0)

dt = 0.25 #sec

policy_action_to_ego_action_map = {}
policy_action_to_ego_action_map[cfg.ACTION_GO_LEFT]     = (4.0, -0.5,)
policy_action_to_ego_action_map[cfg.ACTION_GO_RIGHT]    = (4.0, 0.5,)
policy_action_to_ego_action_map[cfg.ACTION_GO_STRAIGHT] = (5.0, 0.0,)
policy_action_to_ego_action_map[cfg.ACTION_STOP]        = (0.0, 0.0,)

#%% helper functions to simulate environment
reward_func = interpolate.interp1d([cfg.SMALLEST_TTC_ALLOWED, cfg.SMALLEST_TTC_ALLOWED+1], [-10.0, 5.0], fill_value=(-10.0, 5.0,), bounds_error=False)
def reward(state:mcts.State, action:int)->float:
  policy_state = state.policy_state
  r = 0.0
  for i in range(0, policy_state.shape[0]):
    r += reward_func(policy_state[i])

  # penalize for not moving towards goal
  if ((action == cfg.ACTION_GO_LEFT) or (action == cfg.ACTION_GO_RIGHT)):
    r -= 10.0
  elif (action == cfg.ACTION_STOP):
    r -= 5.0
  else: #reward for moving towards goal
    r += 5.0

  return r

def get_required_actions(action:int)->(float, float):
  if (action in policy_action_to_ego_action_map):
    return policy_action_to_ego_action_map[action]
  else:
    print(f"[Warning] unknown action-{action} in function get_required_actions")
    return (0.0, 0.0,)

def env_to_policy_state(s:mcts.State)->np.ndarray:
  policy_state = np.zeros(len(s.objects_state), dtype=int)
  for i, obj in enumerate(s.objects_state):
    ttc = obj_mntn.calc_ttc(s.ego_state, obj)
    policy_state[i] = int(np.floor(ttc))
  return policy_state

def predict_obj(obj:obj_mntn.ObjectState)->obj_mntn.ObjectState:
  dt_sq = dt**2

  obj.x += obj.vx*dt + 0.5*obj.ax*dt_sq
  obj.y += obj.vy*dt + 0.5*obj.ay*dt_sq

  obj.vx += obj.ax*dt
  obj.vy += obj.ay*dt

  return obj

def predict_env(s:mcts.State)->(obj_mntn.ObjectState, list):
  # predict ego
  s.ego_state = predict_obj(s.ego_state)

  # predict env objects
  for obj in s.objects_state:
    obj = predict_obj(obj)
  return s.ego_state, s.objects_state

# given current state and action this function will apply the action and will return next state and reward
def simulator(s:mcts.State, a:int)->(mcts.State, float):
  s_next = mcts.State()
  s_next.ego_state     = copy.deepcopy(s.ego_state)
  s_next.objects_state = copy.deepcopy(s.objects_state)
  s_next.policy_state  = copy.deepcopy(s.policy_state)

  s_next.ego_state.vx, s_next.ego_state.vy = get_required_actions(a)
  s_next.ego_state, s_next.objects_state = predict_env(s_next)
  s_next.policy_state = env_to_policy_state(s_next)
  r = reward(s_next, a)

  return s_next, r

#%% debug variables
ego_x_hist = []
ego_y_hist = []
ego_action_hist = []
action_Q_hist = []

objects_x_hist = [[] for _ in range(len(objects))]
objects_y_hist = [[] for _ in range(len(objects))]

def store_env_hist(s:mcts.State, a:int, Q:float)->None:
  ego_x_hist.append(s.ego_state.x)
  ego_y_hist.append(s.ego_state.y)
  ego_action_hist.append(a)
  action_Q_hist.append(Q)

  for i, obj in enumerate(s.objects_state):
    objects_x_hist[i].append(obj.x)
    objects_y_hist[i].append(obj.y)

#%% main
s = mcts.State()
s.ego_state     = copy.deepcopy(ego)
s.objects_state = copy.deepcopy(objects)
s.policy_state  = env_to_policy_state(s)
mcts.mcts_initialize(s, simulator)

max_iter = 300
i = 0
while (s.ego_state.x < 80.0) and (i < max_iter):
  action, Q = mcts.action_search(mcts.global_tree, s)
  s_next, _ = simulator(s, action)
  mcts.prune_tree(mcts.global_tree, s, action, s_next)
  s = s_next
  store_env_hist(s, action, Q)

  i += 1


#%% plot env history

# plot actions taken by ego
plt.figure(figsize=(8,4))
plt.subplot(2,1,1)
plt.plot(ego_action_hist, 'b-', linewidth=1.5)
plt.ylabel("Action", fontsize=12)
plt.title("Ego actions\n0-left,1-right,2-straight,3-stop", fontsize=14)
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(action_Q_hist, 'r-', linewidth=1.5)
plt.xlabel("cycle", fontsize=12)
plt.ylabel("Q", fontsize=12)
plt.grid(True)
plt.show(block=False)

# plot ego and environment objects trajectory
fig = plt.figure(figsize=(8,4))
object_cmap = plt.cm.get_cmap('hsv', len(objects)+1)
plt.xlim(-50.0, 50.0)
plt.ylim(-10.0, 100.0)
plt.xlabel("Y (m)", fontsize=12)
plt.ylabel("X (m)", fontsize=12)
plt.grid(True)
ax = plt.gca()

ego_path, = ax.plot([], [], '-o', color="black", markersize=0.5, linewidth=1.5, label="Ego")
objects_path = []
for i in range(0, len(objects)):
  temp, = ax.plot([], [], '-x', color=object_cmap(i+1) ,markersize=0.5, linewidth=1.5, label=f"#{i}")
  objects_path.append(temp)

def anim_init():
  global ego_path, objects_path

  ego_path.set_data([], [])
  for obj_path in objects_path:
    obj_path.set_data([], [])
  
  return (ego_path, ) + tuple(objects_path)

def anim_run(frame):
  global ego_path, objects_path
  ego_path.set_data(ego_y_hist[0:frame], ego_x_hist[0:frame])
  for i, obj_path in enumerate(objects_path):
    obj_path.set_data(objects_y_hist[i][0:frame], objects_x_hist[i][0:frame])

  return (ego_path, ) + tuple(objects_path)

anim = FuncAnimation(fig, anim_run, init_func=anim_init,
                    frames=200, interval=60, blit=True)

plt.legend(loc="upper right")
plt.show()


# %%
