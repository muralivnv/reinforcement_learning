#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
replay_buffer = np.load("replay_buffer.npy")
#%% current sample generation

# state current
plt.figure(figsize=(12,7))
plt.plot(replay_buffer[:, 0], replay_buffer[: ,1], "ro", markersize=1, alpha=0.6)
plt.xlabel("S1", fontsize=12)
plt.ylabel("S2", fontsize=12)
plt.title("[S1, S2]", fontsize=14)
plt.show()
#%%
# action current
plt.figure(figsize=(12,7))
plt.plot(replay_buffer[:, 2], replay_buffer[: ,3], "ro", markersize=1, alpha=0.6)
plt.xlabel("A1", fontsize=12)
plt.ylabel("A2", fontsize=12)
plt.title("[A1, A2]", fontsize=14)
plt.show()
#%%
# reward
plt.figure(figsize=(12,7))
plt.plot(replay_buffer[: ,4], "ro", markersize=1, alpha=0.6)
plt.ylabel("R", fontsize=12)
plt.title("Reward", fontsize=14)
plt.show()
#%%
plt.figure(figsize=(12,7))
plt.plot(replay_buffer[: ,5], replay_buffer[:, 6], "ro", markersize=1, alpha=0.6)
plt.ylabel("R", fontsize=12)
plt.title("Reward", fontsize=14)
plt.show()


#%% generating proper samples
state_low = 0.0
state_high = 1.0
action_low = 0.0
action_high = 1.0

n_samples_to_generate = 2000

s1_generated = np.random.uniform(low=state_low, high=state_high, size=n_samples_to_generate)
s2_generated = np.random.uniform(low=state_low, high=state_high, size=n_samples_to_generate)

a1_generated = np.random.uniform(low=action_low, high=action_high, size=n_samples_to_generate)
a2_generated = np.random.uniform(low=action_low, high=action_high, size=n_samples_to_generate)

# state current
plt.figure(figsize=(12,7))
plt.plot(s1_generated, s2_generated, "ro", markersize=1, alpha=0.6)
plt.xlabel("S1", fontsize=12)
plt.ylabel("S2", fontsize=12)
plt.title("[S1, S2] -- Generated", fontsize=14)
plt.show()

# action current
plt.figure(figsize=(12,7))
plt.plot(a1_generated, a2_generated, "ro", markersize=1, alpha=0.6)
plt.xlabel("A1", fontsize=12)
plt.ylabel("A2", fontsize=12)
plt.title("[A1, A2] -- Generated", fontsize=14)
plt.show()

# %%
