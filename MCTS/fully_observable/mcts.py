import numpy as np
import config as cfg

# node handling
UNKNOWN = -1 # unknown node
ROOT    = 0  # root node

# tree expansion strategies
UCB_SELECTION = 0
EXPANSION     = 1

environment_simulator = None # set simulator callback function before starting MCTS
current_max_node_id   = 0
global_tree           = {}
state_to_node_id_map  = {}

class State:
  policy_state:np.ndarray = np.array([])
  ego_state = None
  objects_state = None

class Node:
  parent_id:int     = UNKNOWN
  children_id:list  = []
  v:float           = 0.0
  n_visit:int       = 0

  is_action_node:bool = False
  action:int          = cfg.ACTION_STOP

# function to update action-value (Q)
def update_action_value(tree:dict, node_id:int, r:float)->None:
  query_node = tree[node_id]
  current_v = query_node.v
  delta = (r - current_v)/query_node.n_visit
  query_node.v += delta

# function update state-value (V)
def update_state_value(tree:dict, node_id:int)->None:
  query_node = tree[node_id]

  total_v = 0.0
  for child in query_node.children_id:
    total_v += tree[child].v 
  
  query_node.v += (total_v - query_node.v)/query_node.n_visit

def add_action_to_tree(tree:dict, action:int, parent_id:int)->int:
  global current_max_node_id

  new_node = Node()
  new_node.parent_id = parent_id
  new_node.n_visit += 1
  new_node.is_action_node = True
  new_node.action = action
  new_node.children_id = []

  tree[current_max_node_id] = new_node
  tree[parent_id].children_id.append(current_max_node_id)
  current_max_node_id += 1

  return current_max_node_id - 1

def add_state_to_tree(tree:dict, s:State, parent_id:int)->int:
  global current_max_node_id
  new_node = Node()
  new_node.parent_id = parent_id
  new_node.n_visit += 1
  new_node.is_action_node = False
  new_node.children_id = []

  tree[current_max_node_id] = new_node
  tree[parent_id].children_id.append(current_max_node_id)
  add_state_to_map(s, current_max_node_id)
  current_max_node_id += 1

  return current_max_node_id - 1

# calculate UCB measure for a given node 
def upper_confidence_bound(tree:dict, node_id:int)->float:
  ucb = 0.0

  query_node = tree[node_id]
  parent_id = query_node.parent_id
  if (parent_id != UNKNOWN):
    parent_n_visit = tree[tree[node_id].parent_id].n_visit
    if (parent_n_visit > 0):
      ucb = query_node.v + cfg.UCB_C * np.sqrt(np.log(parent_n_visit)/ (query_node.n_visit + 1e-8) )
  return ucb

# select action based on UCB metric
def ucb_selection(tree: dict, node_id:int)->int:
  query_node = tree[node_id]
  out_action_node_id = UNKNOWN
  if (any(query_node.children_id)):
    ucb = [0.0]*len(query_node.children_id)

    for i, node_id_ in enumerate(query_node.children_id):
      ucb[i] = upper_confidence_bound(tree, node_id_)

    out_action_node_id = query_node.children_id[np.argmax(ucb)]
  
  return out_action_node_id

def expansion_selection(tree:dict, node_id:int)->int:
  random_action = np.random.choice([cfg.ACTION_GO_LEFT, cfg.ACTION_GO_RIGHT, cfg.ACTION_GO_STRAIGHT, cfg.ACTION_STOP], size=1)[0]
  
  # check if this action is already added to the tree
  action_node_id = UNKNOWN
  for child_id in tree[node_id].children_id:
    if tree[child_id].action == random_action:
      action_node_id = child_id
  
  # node doesn't exist, create that node and add it to the tree
  if (action_node_id == UNKNOWN):
    action_node_id = add_action_to_tree(tree, random_action, node_id)

  return action_node_id

def determine_search_strategy(tree:dict, iteration:int)->int:
  strategy = UCB_SELECTION
  root_node = tree[ROOT]

  if (not any(root_node.children_id)) or ((iteration%cfg.EXPANSION_UPDATE_N_CYCLE) == 0):
    strategy = EXPANSION
  
  return strategy

def select_simulation_next_action(tree:int, node_id:int)->(int, int):
  query_node = tree[node_id]
  if (not any(query_node.children_id)):
    action_node_id = expansion_selection(tree, node_id)
  else:
    action_node_id = ucb_selection(tree, node_id)
  
  return tree[action_node_id].action, action_node_id

def simulate(tree:dict, s:State, s_node_id:int, a:int, a_node_id:int, depth:int=0)->float:
  
  # if we are at the end of the planning horizon exit
  if (cfg.DISCOUNT_FACTOR**depth < 0.05):
    return 0.0
  
  s_next, r_immediate = environment_simulator(s, a)
  s_next_node_id = state_to_tree_node(s_next)

  if (s_next_node_id == UNKNOWN):
    s_next_node_id = add_state_to_tree(tree, s_next, a_node_id)
  
  next_action, action_node_id = select_simulation_next_action(tree, s_next_node_id)

  tree[s_next_node_id].n_visit += 1
  tree[action_node_id].n_visit += 1

  r = r_immediate + cfg.DISCOUNT_FACTOR*simulate(tree, s_next, s_next_node_id, next_action, action_node_id, depth+1)

  update_action_value(tree, a_node_id, r)
  update_state_value(tree, s_node_id)
  return r

def get_action_priorities(tree:dict, node_id:int)->(np.ndarray, np.ndarray):
  query_node = tree[node_id]

  action_priorities = np.ones(cfg.N_ACTIONS, dtype=float)*1000.0
  action_node_ids   = np.ones(cfg.N_ACTIONS, dtype=int)*UNKNOWN

  for child in query_node.children_id:
    action = tree[child].action
    action_priorities[action] = -upper_confidence_bound(tree, child)
    action_node_ids[action] = child

  sorted_order = np.argsort(action_priorities)
  return sorted_order, action_node_ids[sorted_order]

# to get next best action
def action_search(tree:dict, state:State)->int:
  root_node = tree[ROOT]  
  i = 0

  actions, actions_node = get_action_priorities(tree, ROOT)
  while(i < cfg.MAX_SEARCH_ITERATIONS):   
    action_node_id = actions_node[i%cfg.N_ACTIONS]
    if (action_node_id == UNKNOWN):
      action_node_id = add_action_to_tree(tree, actions[i], ROOT)
      actions_node[i] = action_node_id

    _ = simulate(tree, state, ROOT, tree[action_node_id].action, action_node_id, 0)
    update_state_value(tree, ROOT)
    i += 1

  # now get best action out of available children
  max_v = -1000.0
  best_action = -1
  for child in root_node.children_id:
    if (tree[child].v > max_v):
      max_v = tree[child].v
      best_action = child
  return tree[best_action].action, max_v

# Monte Carlo Tree Seach initial setup
def add_state_to_map(s:State, node_id:int)->None:
  global state_to_node_id_map

  s_str = np.array2string(s.policy_state, separator='')
  s_str = s_str[1:-1] # drop square brackets
  state_to_node_id_map[s_str] = node_id

def state_to_tree_node(s:State)->int:
  s_str = np.array2string(s.policy_state, separator='')
  s_str = s_str[1:-1] # drop square brackets

  node_id = UNKNOWN
  if (s_str in state_to_node_id_map):
    node_id = state_to_node_id_map[s_str]

  return node_id

def prune_tree(tree:dict, s:State, action:int, s_next:State)->None:
  global ROOT

  # get action node_id
  s_node_id = state_to_tree_node(s)
  s_node    = tree[s_node_id]
  a_node_id = UNKNOWN
  for child in s_node.children_id:
    if (tree[child].action == action):
      a_node_id = child
  
  s_next_node_id = state_to_tree_node(s_next)
  if (s_next_node_id == UNKNOWN):
    s_next_node_id = add_state_to_tree(tree, s_next, a_node_id)
  
  ROOT = s_next_node_id

def mcts_initialize(s0:State, simulator)->None:
  global global_tree, current_max_node_id, environment_simulator

  root_node = Node()
  root_node.parent_id = UNKNOWN
  root_node.n_visit += 1

  global_tree[ROOT] = root_node
  current_max_node_id += 1

  add_state_to_map(s0, ROOT)

  environment_simulator = simulator
