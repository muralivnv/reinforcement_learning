#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
actor_loss_hist = np.load("actor_loss_hist.npy")
critic_loss_hist = np.load("critic_loss_hist.npy")
actor_wgrad_norm_hist = np.load("actor_wgrad_norm_hist.npy")
critic_wgrad_norm_hist = np.load("critic_wgrad_norm_hist.npy")

#%%
def exponential_smoothing(x:np.ndarray, smoothing_factor=0.9) -> np.ndarray:
  x_smoothened = np.zeros_like(x)
  x_smoothened[0] = x[0]
  for i in range(1, x.shape[0]):
    x_smoothened[i] = smoothing_factor*x_smoothened[i-1] + (1.0 - smoothing_factor)*x[i]
  
  return x_smoothened

#%%
actor_loss_smoothened        = exponential_smoothing(actor_loss_hist, 0.95)
critic_loss_smoothened       = exponential_smoothing(critic_loss_hist, 0.95)
actor_wgrad_norm_smoothened  = exponential_smoothing(actor_wgrad_norm_hist, 0.95)
critic_wgrad_norm_smoothened = exponential_smoothing(critic_wgrad_norm_hist, 0.95)
#%%
cycle_start = 25000
cycle_end   = -1

# Actor
plt.figure(figsize=(12,7))
plt.subplot(2,1,1)
plt.plot(actor_loss_smoothened[cycle_start:cycle_end], 'rx', markersize=0.5)
plt.ylabel("Loss", fontsize=12)
plt.title("Actor", fontsize=14)
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(actor_wgrad_norm_smoothened[cycle_start:cycle_end], 'rx', markersize=0.5)
plt.ylabel("Wgrad Norm", fontsize=12)
plt.title("Critic", fontsize=14)
plt.grid(True)
plt.show()

# Critic
plt.figure(figsize=(12,7))
plt.subplot(2,1,1)
plt.plot(critic_loss_smoothened[cycle_start:cycle_end], 'rx', markersize=0.5)
plt.ylabel("Loss", fontsize=12)
plt.title("Critic", fontsize=14)
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(critic_wgrad_norm_smoothened[cycle_start:cycle_end], 'rx', markersize=0.5)
plt.ylabel("Wgrad Norm", fontsize=12)
plt.title("Critic", fontsize=14)
plt.grid(True)
plt.show()

#%% Loss gradient analysis
delta_cycle = 100
actor_loss_grad  = (actor_loss_smoothened[delta_cycle:-1:delta_cycle]-actor_loss_smoothened[0:-delta_cycle:delta_cycle])
critic_loss_grad = (critic_loss_smoothened[delta_cycle:-1:delta_cycle]-critic_loss_smoothened[0:-delta_cycle:delta_cycle])

#%%
plt.figure(figsize=(12,7))
plt.subplot(2,1,1)
plt.plot(actor_loss_grad, 'rx', markersize=1.0)
plt.ylabel("actor loss grad", fontsize=12)
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(critic_loss_grad, 'rx', markersize=1.0)
plt.ylabel("critic loss grad", fontsize=12)
plt.grid(True)
plt.show()
# %%
