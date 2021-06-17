#%%
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#%%
critic_weight_grad_hist = np.load("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/critic_weight_grad_hist.npy")
critic_bias_grad_hist   = np.load("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/critic_bias_grad_hist.npy")
critic_weight_hist      = np.load("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/critic_weight_hist.npy")
critic_bias_hist        = np.load("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/critic_bias_hist.npy")
critic_loss_hist        = np.load("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/critic_loss_hist.npy")

actor_weight_grad_hist  = np.load("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/actor_weight_grad_hist.npy")
actor_bias_grad_hist    = np.load("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/actor_bias_grad_hist.npy")
actor_weight_hist       = np.load("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/actor_weight_hist.npy")
actor_bias_hist         = np.load("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/actor_bias_hist.npy")
actor_loss_hist         = np.load("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/actor_loss_hist.npy")

#%%
cycles_neg_loss   = np.where(actor_loss_hist < 0.0)[0]
actor_wgrad_norm  = np.linalg.norm(actor_weight_grad_hist[cycles_neg_loss, :], axis=0)
critic_wgrad_norm = np.linalg.norm(critic_weight_grad_hist[cycles_neg_loss, :], axis=0)
actor_bgrad_norm  = np.linalg.norm(actor_bias_grad_hist[cycles_neg_loss, :], axis=0)
critic_bgrad_norm = np.linalg.norm(critic_bias_grad_hist[cycles_neg_loss, :], axis=0)

#%%
actor_layer_config  = [2, 8, 12, 20, 25, 2]
critic_layer_config = [4, 10, 14, 21, 27, 1]

#%% analyze actor network

# plt.figure(figsize=(12,4))
# plt.plot(actor_loss_hist, 'r', linewidth=0.8)
# plt.title("actor_loss")
# plt.grid(True)
# plt.show()

# layer_colors = plt.cm.get_cmap('hsv', len(actor_layer_config))

# plt.figure(figsize=(12,17))
# weights_end = 0
# for layer in range(1, len(actor_layer_config)):
#   plt.subplot(len(actor_layer_config)-1, 1, layer)
#   weights_start = weights_end
#   weights_end   = weights_start + actor_layer_config[layer]*actor_layer_config[layer-1]
#   for i in range(weights_start, weights_end):
#     plt.plot(actor_weight_grad_hist[:, i], color=layer_colors(layer), linewidth=0.8)
#   plt.title(f"Actor/Weights/Hidden layer {layer}")
#   plt.grid(True)
# plt.show()

# plt.figure(figsize=(12,17))
# bias_end = 0
# for layer in range(1, len(actor_layer_config)):
#   plt.subplot(len(actor_layer_config)-1, 1, layer)
#   bias_start = bias_end
#   bias_end   = bias_start + actor_layer_config[layer]
#   for i in range(bias_start, bias_end):
#     plt.plot(actor_bias_grad_hist[:, i], color=layer_colors(layer), linewidth=0.8)
#   plt.title(f"Actor/Bias/Hidden layer {layer}")
#   plt.grid(True)
# plt.show()

#%% analyze critic network

# plt.figure(figsize=(12,4))
# plt.plot(critic_loss_hist, 'r', linewidth=0.8)
# plt.title("critic_loss")
# plt.grid(True)
# plt.show()

# layer_colors = plt.cm.get_cmap('hsv', len(critic_layer_config))

# ## weight gradient
# plt.figure(figsize=(12,17))
# weights_end = 0
# for layer in range(1, len(critic_layer_config)):
#   plt.subplot(len(critic_layer_config)-1, 1, layer)
#   weights_start = weights_end
#   weights_end   = weights_start + critic_layer_config[layer]*critic_layer_config[layer-1]
#   for i in range(weights_start, weights_end):
#     plt.plot(critic_weight_grad_hist[:, i], color=layer_colors(layer), linewidth=0.8)
#   plt.title(f"Critic/Weights-grad/Hidden layer {layer}")
#   plt.grid(True)
# plt.show()

# ## weights 
# plt.figure(figsize=(12,17))
# weights_end = 0
# for layer in range(1, len(critic_layer_config)):
#   plt.subplot(len(critic_layer_config)-1, 1, layer)
#   weights_start = weights_end
#   weights_end   = weights_start + critic_layer_config[layer]*critic_layer_config[layer-1]
#   for i in range(weights_start, weights_end):
#     plt.plot(critic_weight_hist[:, i], color=layer_colors(layer), linewidth=0.8)
#   plt.title(f"Critic/Weights/Hidden layer {layer}")
#   plt.grid(True)
# plt.show()

# ## bias gradient
# plt.figure(figsize=(12,17))
# bias_end = 0
# for layer in range(1, len(critic_layer_config)):
#   plt.subplot(len(critic_layer_config)-1, 1, layer)
#   bias_start = bias_end
#   bias_end   = bias_start + critic_layer_config[layer]
#   for i in range(bias_start, bias_end):
#     plt.plot(critic_bias_grad_hist[:, i], color=layer_colors(layer), linewidth=0.8)
#   plt.title(f"Critic/Bias-grad/Hidden layer {layer}")
#   plt.grid(True)
# plt.show()

# ## bias
# plt.figure(figsize=(12,17))
# bias_end = 0
# for layer in range(1, len(critic_layer_config)):
#   plt.subplot(len(critic_layer_config)-1, 1, layer)
#   bias_start = bias_end
#   bias_end   = bias_start + critic_layer_config[layer]
#   for i in range(bias_start, bias_end):
#     plt.plot(critic_bias_hist[:, i], color=layer_colors(layer), linewidth=0.8)
#   plt.title(f"Critic/Bias/Hidden layer {layer}")
#   plt.grid(True)
# plt.show()

# %% Gradient analysis revision-2 -- Actor

plt.figure(figsize=(12,4))
plt.plot(actor_loss_hist, 'r', linewidth=0.8)
plt.title("actor_loss")
plt.grid(True)
plt.show()

layer_colors = plt.cm.get_cmap('hsv', len(actor_layer_config))

plt.figure(figsize=(12,17))
weights_end = 0
for layer in range(1, len(actor_layer_config)):
  plt.subplot(len(actor_layer_config)-1, 1, layer)
  weights_start = weights_end
  weights_end   = weights_start + actor_layer_config[layer]*actor_layer_config[layer-1]
  this_layer_grad  = actor_weight_grad_hist[:, weights_start:weights_end]
  grad_mean        = np.mean(this_layer_grad, axis=1)
  grad_stddev      = np.std(this_layer_grad, axis=1)
  plt.plot(grad_mean,   '-',  color=layer_colors(layer), linewidth=0.8)
  # plt.plot(grad_stddev, '-.', color=layer_colors(layer), linewidth=0.8)
  plt.legend(["Mean", "StdDev"])
  plt.title(f"Actor/Weight-grad/Hidden layer {layer}")
  plt.grid(True)
plt.show()

# weight
plt.figure(figsize=(12,17))
weights_end = 0
for layer in range(1, len(actor_layer_config)):
  plt.subplot(len(actor_layer_config)-1, 1, layer)
  weights_start = weights_end
  weights_end   = weights_start + actor_layer_config[layer]*actor_layer_config[layer-1]
  this_layer_param  = actor_weight_hist[:, weights_start:weights_end]
  param_mean        = np.mean(this_layer_param, axis=1)
  param_stddev      = np.std(this_layer_param, axis=1)
  plt.plot(param_mean,   '-',  color=layer_colors(layer), linewidth=0.8)
  # plt.plot(param_stddev, '-.', color=layer_colors(layer), linewidth=0.8)
  plt.legend(["Mean", "StdDev"])
  plt.title(f"Actor/Weight/Hidden layer {layer}")
  plt.grid(True)
plt.show()

plt.figure(figsize=(12,17))
bias_end = 0
for layer in range(1, len(actor_layer_config)):
  plt.subplot(len(actor_layer_config)-1, 1, layer)
  bias_start = bias_end
  bias_end   = bias_start + actor_layer_config[layer]
  this_layer_grad = actor_bias_grad_hist[:, bias_start:bias_end]
  grad_mean       = np.mean(this_layer_grad, axis=1)
  grad_stddev     = np.std(this_layer_grad, axis=1)
  plt.plot(grad_mean,   color=layer_colors(layer), linewidth=0.8)
  # plt.plot(grad_stddev, color=layer_colors(layer), linewidth=0.8)
  plt.legend(["Mean", "StdDev"])
  plt.title(f"Actor/Bias-grad/Hidden layer {layer}")
  plt.grid(True)
plt.show()

# bias
plt.figure(figsize=(12,17))
bias_end = 0
for layer in range(1, len(actor_layer_config)):
  plt.subplot(len(actor_layer_config)-1, 1, layer)
  bias_start = bias_end
  bias_end   = bias_start + actor_layer_config[layer]
  this_layer_param = actor_bias_hist[:, bias_start:bias_end]
  param_mean       = np.mean(this_layer_param, axis=1)
  param_stddev     = np.std(this_layer_param, axis=1)
  plt.plot(param_mean,   color=layer_colors(layer), linewidth=0.8)
  # plt.plot(param_stddev, color=layer_colors(layer), linewidth=0.8)
  plt.legend(["Mean", "StdDev"])
  plt.title(f"Actor/Bias/Hidden layer {layer}")
  plt.grid(True)
plt.show()

# %% Gradient analysis revision-2 -- Critic

plt.figure(figsize=(12,4))
plt.plot(critic_loss_hist, 'r', linewidth=0.8)
plt.title("critic_loss")
plt.grid(True)
plt.show()

layer_colors = plt.cm.get_cmap('hsv', len(critic_layer_config))

# weight grad
plt.figure(figsize=(12,17))
weights_end = 0
for layer in range(1, len(critic_layer_config)):
  plt.subplot(len(critic_layer_config)-1, 1, layer)
  weights_start = weights_end
  weights_end   = weights_start + critic_layer_config[layer]*critic_layer_config[layer-1]
  this_layer_grad  = critic_weight_grad_hist[:, weights_start:weights_end]
  grad_mean        = np.mean(this_layer_grad, axis=1)
  grad_stddev      = np.std(this_layer_grad, axis=1)
  plt.plot(grad_mean,   '-',  color=layer_colors(layer), linewidth=0.8)
  # plt.plot(grad_stddev, '-.', color=layer_colors(layer), linewidth=0.8)
  plt.legend(["Mean", "StdDev"])
  plt.title(f"Critic/Weight-grad/Hidden layer {layer}")
  plt.grid(True)
plt.show()

# weight
plt.figure(figsize=(12,17))
weights_end = 0
for layer in range(1, len(critic_layer_config)):
  plt.subplot(len(critic_layer_config)-1, 1, layer)
  weights_start = weights_end
  weights_end   = weights_start + critic_layer_config[layer]*critic_layer_config[layer-1]
  this_layer_param  = critic_weight_hist[:, weights_start:weights_end]
  param_mean        = np.mean(this_layer_param, axis=1)
  param_stddev      = np.std(this_layer_param, axis=1)
  plt.plot(param_mean,   '-',  color=layer_colors(layer), linewidth=0.8)
  # plt.plot(param_stddev, '-.', color=layer_colors(layer), linewidth=0.8)
  plt.legend(["Mean", "StdDev"])
  plt.title(f"Critic/Weight/Hidden layer {layer}")
  plt.grid(True)
plt.show()

# bias grad
plt.figure(figsize=(12,17))
bias_end = 0
for layer in range(1, len(critic_layer_config)):
  plt.subplot(len(critic_layer_config)-1, 1, layer)
  bias_start = bias_end
  bias_end   = bias_start + critic_layer_config[layer]
  this_layer_grad = critic_bias_grad_hist[:, bias_start:bias_end]
  grad_mean       = np.mean(this_layer_grad, axis=1)
  grad_stddev     = np.std(this_layer_grad, axis=1)
  plt.plot(grad_mean,   color=layer_colors(layer), linewidth=0.8)
  # plt.plot(grad_stddev, color=layer_colors(layer), linewidth=0.8)
  plt.legend(["Mean", "StdDev"])
  plt.title(f"Critic/Bias-grad/Hidden layer {layer}")
  plt.grid(True)
plt.show()

# bias
plt.figure(figsize=(12,17))
bias_end = 0
for layer in range(1, len(critic_layer_config)):
  plt.subplot(len(critic_layer_config)-1, 1, layer)
  bias_start = bias_end
  bias_end   = bias_start + critic_layer_config[layer]
  this_layer_param = critic_bias_hist[:, bias_start:bias_end]
  param_mean       = np.mean(this_layer_param, axis=1)
  param_stddev     = np.std(this_layer_param, axis=1)
  plt.plot(param_mean,   color=layer_colors(layer), linewidth=0.8)
  # plt.plot(param_stddev, color=layer_colors(layer), linewidth=0.8)
  plt.legend(["Mean", "StdDev"])
  plt.title(f"Critic/Bias/Hidden layer {layer}")
  plt.grid(True)
plt.show()

# %%
nums = [-4300,-1364,-814,-1232,-902,-990,288,108,60,144,96,312,264,192,24,156,228,204,300,348,360,252,372,276,36,48,12,84,240,324,180,132,336,168,72,216,120,384,396,-1932,-5152,-3772,-2576,-5888,-2064,-4472,2457,91,6370,910,2002,3276,1183,728,2912,4004,5642,1092,3731,2548,3549,4550,4095,546,3003,5005,1274,1547,6006,5551,4823,1820,4732,4277,3458,2821,4914,-266,-608,-304,-950,-38,-1178,-798,-1064,-760,-912,-190,-1330,-570,-646,-684,-228,-1292,-342,-456,-1216,-1254,-494,-1026,-532,-114,-1140,-380,-1102,-418,-722,-988,-836,-76,-874,-152,5733,3185,273,3640,-1998,-1184,-304,-76,-190,-114,-38,-380,-342,-228,-266,-152,-1258,-888,-444,-2886,-2812,-1406,-3552,-3922,-2590,-1628,-4736,-2516,-3034,-2294,-666,-1554,-962,-4292,-3256,-4440,-1850,-1702,-2442,-4514,-4884,-2220,-4588,-2960,-3626,-4366,-3700,-5106,-3996,-4958,-3848,-1036,-296,-4218,-4144,-3330,-2664,-2146,-3182,-2072,-5180,-814,-740,-4810,3185,910,1105,1950,4030,2860,2145,4550,650,260,4355,1170,585,3835,4160,4095,2470,2990,1560,975,2080,4420,3575,3510,390,-1025,-650,-850,-1425,-200,-700,-1250,-925,-125,-1725,-550,-1600,-1175,-600,-1475,-400,-150,-450,-1200,-1300,-575,-1125,-175,-1700,-675,-225,-725,-300,-1450,-250,-1325,-1525,-375,1200,2352,1152,1248,48,2880,480,880,1166,66,88,44,308,814,440,506,990,528,704,1034,396,374,1144,132,792,1430,1320,1386,1254,220,638,484,1452,1276,1518,330,924,1540,1122,198,1364,660,264,858,726,1496,1408,1100,286,154,1342,902,770,352,946,176,1188,572,22,748,594,836,1298,418,110,242,1078,682,1012,616,1474,1210,1056,1232,968,462,550,315,119,63,385,266,91,84,280,133,483,203,-3111,-1632,-3366,-2805,-3417,-51,-1428,-2295,-1122,-102,-1020,-2448,-1887,-1224,-1683,-2958,-2040,-3009,-2907,183,549,-2108,-340,-2380,-4556,-1088]
l = [235,133,248,248,276,31,91,44,96,220,46,42,61,295,276,248,228,247,264,16,176,115,238,131,68,43,205,204,92,81,46,253,190,160,198,162,335,263,37,209,240,158,82,269,307,211,175,117,76,197,131,256,219,87,118,22,1,173,158,27,142,57,73,2,186,198,252,139,46,54,208,118,73,28,109,16,256,49,165,97,139,32,177,106,213,202,57,237,177,16,266,78,277,120,83,24,185,119,148,65,69,118,140,38,186,146,282,279,188,16,148,178,10,99,67,191,217,15,212,27,104,10,266,53,13,210,196,56,74,177,140,266,118,233,81,328,187,95,21,89,15,195,44,147,239,25,78,103,97,179,216,39,48,20,256,109,273,217,156,214,220,175,226,241,235,262,223,14,97,260,154,259,46,317,167,14,37,312,159,173,255,142,60,107,0,256,113,93,223,107,91,225,281,146,128,7,134,220,7,71,243,199,111,94,269,84,118,84,27,25,170,162,98,241,233,269,213,183,255,170,278,217,113,291,248,252,47,133,51,34,220,27,150,12,101,232,223,72,107,91,116,9,135,257,6,230,94,118,229,115,155,214,162,201,53,240,225,192,234,22,232,240,165,78,130,295,107,117,71,88,15,198,244,240,63,57,118,201,65,164,122,224,34,61,309,66,268,229,139,225,264,248,259,130,179,242,2,76,58,102,2,213,39,190,40,191,84,170,266,273,111,235,50,156,264,3,161,191,123,52,159,182,221,236,244,43,96,173,245,149,266,159,45,267,61,41,162,242,212,195,142,229,226,220,182,220,13,311,126,87,107,26,274,223,121,271,115,94,61,107,167,252,75,21,226,85,213,170,162,53,73,156,49,230,118,263,188,266,236,247,257,333,38,163,84,159,106,88,179,37,152,110,247,205,175,12,110,272,104,11,191,295,129,110,326,263,51,141,103,67,75,125,109,203,162,18,74,265,27,223,77,133,259,1,89,292,96,143,220,116,126,109]
r = [304,202,317,317,311,60,160,113,165,289,115,111,130,298,345,263,297,313,333,52,245,184,256,200,138,112,274,273,161,150,59,264,211,229,237,231,338,332,106,266,308,227,151,338,324,280,244,186,145,211,200,276,288,156,187,91,70,242,227,96,211,126,142,13,255,267,321,179,69,123,254,127,142,85,178,85,325,118,234,166,188,101,233,175,226,271,126,270,246,85,335,147,304,189,152,93,254,188,217,134,138,187,209,73,255,167,341,311,257,85,217,247,79,168,69,260,283,84,281,96,173,79,335,122,82,279,265,125,128,235,201,335,187,302,150,341,225,164,61,158,84,264,113,216,308,94,147,172,166,248,285,108,77,45,325,172,290,236,225,251,289,244,295,310,293,331,292,83,153,283,223,294,115,342,181,45,106,341,228,242,262,211,129,176,69,265,168,162,246,176,93,294,323,215,178,76,136,225,76,140,286,268,138,163,325,104,141,153,96,94,239,231,142,310,302,338,282,234,324,239,347,286,182,313,250,321,116,152,120,36,289,52,162,81,170,268,246,141,176,160,185,78,204,326,38,299,163,131,298,127,224,283,231,270,122,309,294,261,303,91,266,309,234,147,199,329,176,164,112,157,84,248,313,309,132,126,187,270,134,233,191,276,57,98,346,135,337,298,208,294,333,317,328,159,196,311,45,145,127,171,71,261,108,259,55,260,93,220,335,342,180,279,107,225,333,72,230,230,154,65,228,251,290,305,313,112,144,242,248,218,335,202,114,336,130,110,164,311,281,264,197,298,295,289,251,289,82,341,195,156,176,38,316,262,190,340,184,163,130,138,236,321,144,90,295,154,282,217,231,122,142,225,118,299,187,290,257,335,305,316,303,342,107,193,150,228,175,154,221,106,166,119,287,274,244,81,179,337,173,80,260,327,198,179,341,299,120,150,172,136,144,149,178,272,167,87,77,334,96,292,111,135,328,70,158,337,165,212,289,185,188,158]
#%%
idx = 180
sub_arr = np.sort(nums[l[idx]:r[idx]+1])
print(sub_arr)
print(np.sum(sub_arr - sub_arr[0]))
# %%
