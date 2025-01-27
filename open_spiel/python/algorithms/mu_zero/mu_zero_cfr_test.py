

import time
import pyspiel
import numpy as np
import jax.numpy as jnp


from open_spiel.python.policy import TabularPolicy
from open_spiel.python.algorithms.best_response import BestResponsePolicy

from open_spiel.python.jax.cfr.jax_cfr import JaxCFR
from open_spiel.python.algorithms.mu_zero.mu_zero_cfr import MuZeroCFR, MuZeroCFRConstants


def convert_player_depth_to_jax(arr):
  return [[jnp.array(p) for p in d] for d in arr]

def convert_depth_to_jax(arr):
  return [jnp.array(d) for d in arr]

def construct_dl_muzero_cfr(states: list[pyspiel.State], player:int, init_reaches, cf_values):
  # init_reaches = [[1], [1]]
  depth_actions = []  # Is just a list of integers
  dict_map = {}
  depth_iset_map = []
  depth_iset_legal = [] 
  depth_history_action_utility = [] 
  depth_history_iset = [] 
  depth_history_actions = [] 
  depth_history_legal = []
  depth_history_next_history= []
  
  def init_layer(depth, legal_size):
    depth_actions.append(legal_size)
    depth_history_action_utility.append([])
    depth_history_next_history.append([])
    depth_iset_map.append([])
    depth_iset_legal.append([])
    depth_history_iset.append([])
    depth_history_actions.append([])
    depth_history_legal.append([])
    for pl in range(2):
      depth_iset_map[depth].append([])
      depth_iset_legal[depth].append([])
      depth_history_iset[depth].append([])
      depth_history_actions[depth].append([])
      # depth_history_legal[depth].append([])
  
  
  def _traverse_tree(state: pyspiel.State, depth: int = 0):
    if state.is_terminal():
      return
    if len(depth_actions) < depth + 1:
      init_layer(depth, len(state.legal_actions_mask(0)))
      
    for pl in range(2):
      iset = state.information_state_string(pl)
      iset_tensor = state.information_state_tensor(pl)
      if iset not in dict_map:
        dict_map[iset] = (depth, len(depth_iset_map[depth][pl]))
        depth_iset_map[depth][pl].append(iset_tensor)
        depth_iset_legal[depth][pl].append(state.legal_actions_mask(pl))
      depth_history_iset[depth][pl].append(dict_map[iset][1])
      depth_history_actions[depth][pl].append([i + dict_map[iset][1] * len(state.legal_actions_mask(pl)) for i in range(len(state.legal_actions_mask(pl)))])
     
     
    action_utility = np.zeros((len(state.legal_actions_mask(0)), len(state.legal_actions_mask(1))))
    legal = np.zeros_like(action_utility, dtype=bool)
    next_history = np.full_like(action_utility, -1, dtype=int)
    
    depth_history_action_utility[depth].append(action_utility)
    depth_history_legal[depth].append(legal)
    depth_history_next_history[depth].append(next_history)
        
    for a1 in state.legal_actions(0):
      for a2 in state.legal_actions(1):
        next_state = state.clone()
        next_state.apply_actions([a1, a2])
        legal[a1, a2] = True
        action_utility[a1, a2] = next_state.returns()[0]
        
        if not next_state.is_terminal():
          next_id = 0 if len(depth_history_action_utility) < depth + 2 else len(depth_history_action_utility[depth + 1])
          next_history[a1, a2] = next_id
          
        _traverse_tree(next_state, depth + 1)
  # TODO: If you change this to different game, then this condition has to be different
  if len(states) == 1:
    _traverse_tree(states[0], 0)
  else:
    depth = 0
    init_layer(depth, 2)
    for i, state in enumerate(states):
      for pl in range(2):
        gadget_iset = "Gadget: " + state.information_state_string(pl)
        iset_tensor = state.information_state_tensor(pl)
        legal_actions = [1, 0] if pl == player else [1, 1]
        if gadget_iset not in dict_map:
          dict_map[gadget_iset] = (depth, len(depth_iset_map[depth][pl]))
          depth_iset_map[depth][pl].append(iset_tensor)
          depth_iset_legal[depth][pl].append(legal_actions)
        depth_history_iset[depth][pl].append(dict_map[gadget_iset][1])
        depth_history_actions[depth][pl].append([i + dict_map[gadget_iset][1] * len(legal_actions) for i in range(len(legal_actions))])
      action_utility = np.array([[cf_values[i], 0.0], [0.0, 0.0]])
      
      if player == 0:
        legal = np.array([[1, 1], [0, 0]])
        next_history = np.array([[-1, i], [-1, -1]])
      elif player == 1:
        legal = np.array([[1, 0], [1, 0]])
        next_history = np.array([[-1, -1], [i, -1]])
      else:
        assert False, "Player should be either 0 or 1"
      depth_history_action_utility[depth].append(action_utility)
      depth_history_legal[depth].append(legal)
      depth_history_next_history[depth].append(next_history)
      _traverse_tree(state, 1)
  
  constants = MuZeroCFRConstants(
    max_depth = len(depth_history_actions),
    resolving_player = 0,
    init_reaches = jnp.array(init_reaches),
    depth_actions = depth_actions,
    depth_iset_map = convert_player_depth_to_jax(depth_iset_map),
    depth_iset_legal = convert_player_depth_to_jax(depth_iset_legal),
    
    depth_history_action_utility = convert_depth_to_jax(depth_history_action_utility),
    depth_history_iset = convert_depth_to_jax(depth_history_iset),
    depth_history_actions = convert_depth_to_jax(depth_history_actions),
    depth_history_legal = convert_depth_to_jax(depth_history_legal),
    depth_history_next_history = convert_depth_to_jax(depth_history_next_history)
  )
  return MuZeroCFR(constants), dict_map

def construct_muzero_cfr(game: pyspiel.Game):
  init_reaches = [[1], [1]]
  cf_values = [0]
  return construct_dl_muzero_cfr([game.new_initial_state()], 1, init_reaches, cf_values) 

def extract_muzero_policy(muzero_cfr: MuZeroCFR, dict_map: dict, sim_game: pyspiel.Game, seq_game: pyspiel.Game):
  
  policy = TabularPolicy(seq_game)
  
  averages = [[p / jnp.sum(p, -1, keepdims=True) for p in d] for d in muzero_cfr.averages]
  
  def _traverse_tree(sim_state: pyspiel.State, seq_state:pyspiel.State, depth: int = 0):
    if sim_state.is_terminal():
      assert seq_state.is_terminal()
      return
    
    p1_average = averages[depth][0][dict_map[sim_state.information_state_string(0)][1]]
    p2_average = averages[depth][1][dict_map[sim_state.information_state_string(1)][1]]
    
    
    assert seq_state.legal_actions() == sim_state.legal_actions(0)
    for a1 in seq_state.legal_actions():
      
      p1_pol = policy.policy_for_key(seq_state.information_state_string(0))
      for i in range(len(p1_pol)):
        p1_pol[i] = p1_average[i]
        
      new_seq_state = seq_state.clone()
      new_seq_state.apply_action(a1)
      assert new_seq_state.legal_actions() == sim_state.legal_actions(1)
      for a2 in new_seq_state.legal_actions():
        
        p2_pol = policy.policy_for_key(new_seq_state.information_state_string(1))
        for i in range(len(p2_pol)):
          p2_pol[i] = p2_average[i]
        
        new_new_seq_state = new_seq_state.clone()
        new_new_seq_state.apply_action(a2)
        new_sim_state = sim_state.clone()
        new_sim_state.apply_actions([a1, a2])
        _traverse_tree(new_sim_state, new_new_seq_state, depth + 1)
    
    pass
  
  _traverse_tree(sim_game.new_initial_state(), seq_game.new_initial_state())
  return policy


def compare_muzero_cfr_to_jax_cfr(cards: int):
  params = {"num_cards": cards, "imp_info": True, "points_order": "descending"}
  
  game = pyspiel.load_game_as_turn_based("goofspiel", params)
  sim_game = pyspiel.load_game("goofspiel", params)
  jax_cfr = JaxCFR(game)
  
  muzero_cfr, dict_map = construct_muzero_cfr(sim_game)
  iterations = 1
  inner_iters = 10000
  for i in range(iterations): 
    start = time.time()
    jax_cfr.multiple_steps(inner_iters)
    print("JAX CFR: ", time.time() - start)
    start = time.time()
    muzero_cfr.multiple_steps(inner_iters)  
    print("MuZero CFR: ", time.time() - start)
    
  cfr_pol = jax_cfr.average_policy()
  muzero_pol = extract_muzero_policy(muzero_cfr, dict_map, sim_game, game)
  
  
  cfr_br1 = BestResponsePolicy(jax_cfr.game, 1, cfr_pol)
  cfr_br2 = BestResponsePolicy(jax_cfr.game, 0, cfr_pol)

  muzero_br1 = BestResponsePolicy(jax_cfr.game, 1, muzero_pol)
  muzero_br2 = BestResponsePolicy(jax_cfr.game, 0, muzero_pol)

  print("CFR P1: ", cfr_br1.value(jax_cfr.game.new_initial_state()))
  print("MuZero P1: ", muzero_br1.value(jax_cfr.game.new_initial_state()))
  print("CFR P2: ", cfr_br2.value(jax_cfr.game.new_initial_state()))
  print("MuZero P2: ", muzero_br2.value(jax_cfr.game.new_initial_state()))
  


def find_imm_next_isets(states: list[pyspiel.State], player):
  
  isets = {}
  next_states = []
  for state in states:
    for a1 in state.legal_actions(0):
      for a2 in state.legal_actions(1):
        new_state = state.clone()
        new_state.apply_actions([a1, a2])
        if new_state.is_terminal():
          continue
        isets[new_state.information_state_string(player)] = new_state.information_state_tensor(player)
        next_states.append(new_state)
  return isets, next_states
  
  
def find_gadget_policy(states, player, reaches, cf_values, policy_dict):
  is_gadget = len(states) > 1
  depth = 1 if is_gadget else 0
  muzero_cfr, dict_map = construct_dl_muzero_cfr(states, player, reaches, cf_values) 
  muzero_cfr.multiple_steps(1000)
  # if is_gadget:
  #   muzero_cfr.multiple_steps(1000)  
    
  averages = [[jnp.where(jnp.sum(p, -1, keepdims=True) > 1e-10, p / jnp.sum(p, -1, keepdims=True), 1 / p.shape[-1])  for p in d] for d in muzero_cfr.averages]
  for state in states:
    iset_string = state.information_state_string(player)
    policy_dict[iset_string] = averages[depth][player][dict_map[iset_string][1]]
 
  next_isets, new_possible_states = find_imm_next_isets(states, player)
  for iset_string, iset_tensor in next_isets.items():
    if iset_string in policy_dict:
      continue
    next_histories = muzero_cfr.find_public_state_from_iset(np.array(iset_tensor), player, depth+1)
    next_states = [new_possible_states[h] for h in next_histories]
    next_reaches = muzero_cfr.find_reaches_from_average()[depth+1][:, next_histories]
    next_reaches = jnp.where(jnp.array([[player == 0], [player == 1]]), next_reaches, 1.0)  
    next_isets = muzero_cfr.constants.depth_history_iset[depth+1][:, next_histories]
    next_cf_values = muzero_cfr.cf_values[depth+1][1 - player][next_isets[1 - player]]
    find_gadget_policy(next_states, player, next_reaches, next_cf_values, policy_dict)
    
def extract_policy_from_dict(seq_game, sim_game, policy_dict):
  policy = TabularPolicy(seq_game)
   
  
  def _traverse_tree(sim_state: pyspiel.State, seq_state:pyspiel.State, depth: int = 0):
    if sim_state.is_terminal():
      assert seq_state.is_terminal()
      return
    
    cfr_p1_policy = policy_dict[sim_state.information_state_string(0)]
    cfr_p2_policy = policy_dict[sim_state.information_state_string(1)]
    
    assert seq_state.legal_actions() == sim_state.legal_actions(0)
    for a1 in seq_state.legal_actions():
      
      p1_pol = policy.policy_for_key(seq_state.information_state_string(0))
      for i in range(len(p1_pol)):
        p1_pol[i] = cfr_p1_policy[i]
        
      new_seq_state = seq_state.clone()
      new_seq_state.apply_action(a1)
      assert new_seq_state.legal_actions() == sim_state.legal_actions(1)
      for a2 in new_seq_state.legal_actions():
        
        p2_pol = policy.policy_for_key(new_seq_state.information_state_string(1))
        for i in range(len(p2_pol)):
          p2_pol[i] = cfr_p2_policy[i]
        
        new_new_seq_state = new_seq_state.clone()
        new_new_seq_state.apply_action(a2)
        new_sim_state = sim_state.clone()
        new_sim_state.apply_actions([a1, a2])
        _traverse_tree(new_sim_state, new_new_seq_state, depth + 1)
    
    pass
  
  _traverse_tree(sim_game.new_initial_state(), seq_game.new_initial_state())
  return policy
    
def compute_exploitability_gadget(cards):
  params = {"num_cards": cards, "imp_info": True, "points_order": "descending"}
  
  game = pyspiel.load_game_as_turn_based("goofspiel", params)
  sim_game = pyspiel.load_game("goofspiel", params)
  
  
  policy_dict = {}
  states = [sim_game.new_initial_state()]
  reaches = np.array([[1], [1]])
  cf_values = np.array([0])
  
  for pl in range(2):
    find_gadget_policy(states, pl, reaches, cf_values, policy_dict)
    
  print(policy_dict)
  cfr_pol = extract_policy_from_dict(game, sim_game, policy_dict)
  
  cfr_br1 = BestResponsePolicy(game, 1, cfr_pol)
  cfr_br2 = BestResponsePolicy(game, 0, cfr_pol)
  
  print("CFR P1: ", cfr_br1.value(game.new_initial_state()))
  print("CFR P2: ", cfr_br2.value(game.new_initial_state()))
  
   
    
  

if __name__ == "__main__":
  compute_exploitability_gadget(3)