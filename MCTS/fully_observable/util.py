import numpy as np
import config as cfg
import object_maintenance as obj_mntn

def object_to_policy_state(ego_state: obj_mntn.ObjectState, env_objs: list)->np.ndarray:
  if (any(env_objs)):
    out_arr = np.zeros(len(env_objs), dtype=int)
    for i, obj in enumerate(env_objs):
      ttc = obj_mntn.calc_ttc(ego_state, obj) # output will be float
      out_arr[i] = np.floor(ttc)
    return out_arr
  else:
    return np.array([])


