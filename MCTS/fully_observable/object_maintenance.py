import numpy as np
import config as cfg

# custom classes to store states and handle transition
class ObjectState:
  x:float = 0.0
  y:float = 0.0
  vx:float = 0.0
  vy:float = 0.0
  ax:float = 0.0
  ay:float = 0.0
    
  def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, ax=0.0, ay=0.0):
    self.x = x
    self.y = y
    self.vx = vx
    self.vy = vy
    self.ax = ax
    self.ay = ay

# function to predict state for time span of dt
def state_predict(state:ObjectState, dt:float)->ObjectState:
  new_state = ObjectState()
  new_state.x = state.x + state.vx*dt + state.ax*dt*dt
  new_state.y = state.y + state.vy*dt + state.ay*dt*dt
  new_state.vx = state.vx + state.ax*dt
  new_state.vy = state.vy + state.ay*dt
  return new_state

# function to check whether a given object is greater than smallest allowable TTC 
def is_within_allowabled_ttc(ego_state:ObjectState, env_obj:ObjectState)->bool:
  is_safe = True 
  
  # check whether the object is within ego path
  is_in_ego_path = False
  obj_x_rel  = env_obj.x - ego_state.x
  obj_y_rel  = env_obj.y - ego_state.y
  obj_vy_rel = env_obj.vy - ego_state.vy
  obj_vx_rel = env_obj.vx - ego_state.vx
  
  if (obj_x_rel > 0.0):
    if ((obj_y_rel > 0.0) and (obj_vy_rel < 0.0)) or ((obj_y_rel < 0.0) and (obj_vy_rel > 0.0)) :
      is_in_ego_path = True
  
  if (is_in_ego_path):
    r = np.sqrt(obj_x_rel**2 + obj_y_rel**2)
    v = np.sqrt(obj_vx_rel**2 + obj_vy_rel**2)
      
    ttc = r/v
    if (ttc < cfg.SMALLEST_TTC_ALLOWED):
      is_safe = False
  
  return is_safe

# function to calculate TTC 
def calc_ttc(ego_state:ObjectState, env_obj:ObjectState)->float:
  ttc = 1000.0 #sec

  # check whether the object is within ego path
  is_in_ego_path = False
  obj_x_rel  = env_obj.x - ego_state.x
  obj_y_rel  = env_obj.y - ego_state.y
  obj_vy_rel = env_obj.vy - ego_state.vy
  obj_vx_rel = env_obj.vx - ego_state.vx
  
  if (obj_x_rel > 0.0):
    if ((obj_y_rel > 0.0) and (obj_vy_rel < 0.0)) or ((obj_y_rel < 0.0) and (obj_vy_rel > 0.0)) :
      is_in_ego_path = True
  
  if (is_in_ego_path):
    r = np.sqrt(obj_x_rel**2 + obj_y_rel**2)
    v = np.sqrt(obj_vx_rel**2 + obj_vy_rel**2)
    ttc = r/v
  
  return ttc