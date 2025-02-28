
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game import JaxGame, JaxPolicy
from open_spiel.python.algorithms.mu_zero.mu_zero_cfr import MuZeroCFR, MuZeroCFRConstants
from open_spiel.python.algorithms.mu_zero.experiments.utils import stringify
from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import convert_depth_to_jax, convert_player_depth_to_jax

import jax
import jax.numpy as jnp
import numpy as np
from copy import deepcopy


# TODO: Wouldn't it be better to use np instead of jnp?
def create_random_policy(game: JaxGame, seed: int):
  key = jax.random.key(seed)
  state_key, init_key = jax.random.split(key)
  game_state, legals = game.initialize_structures(init_key)
  
  policy_dict = JaxPolicy()

  def _traverse_tree(game_state, legal_actions, key, depth=0):
    
    state, p1_iset, p2_iset, ps = game.get_info(game_state)
    p1_iset = np.array(p1_iset)
    p2_iset = np.array(p2_iset)
    # ps_str = stringify(ps)
    p1_iset_str = stringify(p1_iset)
    p2_iset_str = stringify(p2_iset)
    
    key, p1_pol_key, p2_pol_key = jax.random.split(key, 3)
    p1_pol = jax.random.uniform(p1_pol_key, (len(legal_actions[0]),)) * legal_actions[0]
    p2_pol = jax.random.uniform(p2_pol_key, (len(legal_actions[1]),)) * legal_actions[1]
    p1_pol = p1_pol / jnp.sum(p1_pol)
    p2_pol = p2_pol / jnp.sum(p2_pol)
     
    policy_dict[p1_iset_str] = p1_pol
    policy_dict[p2_iset_str] = p2_pol
    
    for a1i, a1 in enumerate(legal_actions[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legal_actions[1]):
        if a2 < 0.5:
          continue 
        
        key, next_key, action_key = jax.random.split(key, 3)
        new_game_state, new_terminal, new_rewards, new_legals = game.apply_action(
            game_state, action_key, depth, np.array([a1i, a2i]))
        
        if new_terminal:
          continue 
        
        _traverse_tree(new_game_state, new_legals, next_key, depth + 1)
        
  _traverse_tree(game_state, legals, state_key)
  return policy_dict
  

def exploitability_jax_game(game: JaxGame, policy: JaxPolicy) -> tuple[JaxPolicy, JaxPolicy, float, float]:
  '''
    Computes a best response against both players using given policy.
    Return is a tuple of two JaxPolicy objects, one for each player and a value of this best response against the input policy.
    The other player is always the original policy.
    The output format: BR against policy of player 0, BR against policy of player 1, Value of BR against policy of player 0, Value of BR against policy of player 1.
  '''
  key = jax.random.key(0)
  
  init_key, key = jax.random.split(key)
  
  
  init_state, init_legals = game.initialize_structures(init_key)

  iset_action_value = {}
  iset_action_legal = {}
  
  iset_map = []
  iset_legals = [] # [D, Pl, I, A]
  
  states = []
  isets = [] # [D, Pl, H(D)]
  behavior_policy = [] # [D, Pl, H(D), A]
  all_reaches = [] # [D, Pl, H(D)]
  actions = [] # [D, ]
  legals = [] # [D, H(D), A1, A2]
  continuations = [] # [D, H(D), A1, A2]
  rewards = [] # [D, H(D), A1, A2]
  terminals = [] # [D, H(D), A1, A2]
   
  def _construct_tree(game_state, legal_actions, key, reaches: tuple[float, float] = (1.0, 1.0), depth: int =0):
    
    # actions = legal_actions[0].shape[0]
    
    if len(isets) < depth + 1:
      isets.append([[], []])
      actions.append([[], []])
      behavior_policy.append([[], []])
      iset_map.append([[], []]) 
      all_reaches.append([[], []])
      iset_legals.append([[], []])
      legals.append([])
      continuations.append([])
      rewards.append([])
      terminals.append([])
    state_tensor, p1_iset, p2_iset, ps = game.get_info(game_state)
    p1_iset = np.array(p1_iset)
    p2_iset = np.array(p2_iset)
    
    # ps_str = stringify(ps)
    p1_iset_str = stringify(p1_iset)
    p2_iset_str = stringify(p2_iset) 
     
    if p1_iset_str not in iset_map[depth][0]:
      iset_map[depth][0].append(p1_iset_str)
      iset_legals[depth][0].append(np.array(legal_actions[0]))
    if p2_iset_str not in iset_map[depth][1]:
      iset_map[depth][1].append(p2_iset_str)
      iset_legals[depth][1].append(np.array(legal_actions[1]))
    
    isets[depth][0].append(iset_map[depth][0].index(p1_iset_str))
    isets[depth][1].append(iset_map[depth][1].index(p2_iset_str))
    b_pol1 = np.array(policy[p1_iset_str])
    b_pol2 = np.array(policy[p2_iset_str])
    assert abs(np.sum(b_pol1) - 1) < 1e-3
    assert abs(np.sum(b_pol2) - 1) < 1e-3
    behavior_policy[depth][0].append(np.array(policy[p1_iset_str]))
    behavior_policy[depth][1].append(np.array(policy[p2_iset_str]))
    all_reaches[depth][0].append(reaches[0])
    all_reaches[depth][1].append(reaches[1])
      
    state_legals = legal_actions[0][..., None] * legal_actions[1][None, ...]
    state_reward = np.zeros_like(state_legals)
    state_continuation = np.full_like(state_legals, -1, dtype=np.int32)
    state_terminals = np.zeros_like(state_legals)
    legals[depth].append(state_legals)
    rewards[depth].append(state_reward)
    continuations[depth].append(state_continuation)
    terminals[depth].append(state_terminals)
    
  
    if p1_iset_str not in iset_action_value:
      iset_action_value[p1_iset_str] = np.zeros(len(legal_actions[0]))
      iset_action_legal[p1_iset_str] = np.array(legal_actions[0])
    if p2_iset_str not in iset_action_value:
      iset_action_value[p2_iset_str] = np.zeros(len(legal_actions[1]))
      iset_action_legal[p2_iset_str] = np.array(legal_actions[1])
    
    p1_state_value, p2_state_value = 0.0, 0.0
    for a1i, a1 in enumerate(legal_actions[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legal_actions[1]):
        if a2 < 0.5:
          continue
         
        
        next_key, action_key = jax.random.split(key)
        new_game_state, new_terminal, new_rewards, new_legals = game.apply_action(
          game_state, action_key, depth, np.array([a1i, a2i]))
        
        state_reward[a1i, a2i] = new_rewards
         
        if new_terminal:
          continue
        
        state_terminals[a1i, a2i] = 1
        
        next_history_id = 0 if len(legals) <= depth+1 else len(legals[depth + 1])
        state_continuation[a1i, a2i] = next_history_id
        
        new_reaches = (reaches[0] * np.array(policy[p1_iset_str][a1i]), reaches[1] * np.array(policy[p2_iset_str][a2i]))
        _construct_tree(new_game_state, new_legals, next_key, new_reaches, depth + 1) 
   
  
  _construct_tree(init_state, init_legals, key) 
  
  def convert_to_numpy(x):
    return [np.array(d) for d in x]
  def convert_to_numpy_players(x):
    return [[np.array(pl) for pl in d] for d in x]
  
  
  br_policy_p1 = deepcopy(policy)
  br_policy_p2 = deepcopy(policy)
  
  isets = convert_to_numpy(isets)
  iset_legals = convert_to_numpy_players(iset_legals)
  behavior_policy = convert_to_numpy(behavior_policy)
  all_reaches = convert_to_numpy(all_reaches)
  # actions = convert_to_numpy_players(actions)
  legals = convert_to_numpy(legals)
  continuations = convert_to_numpy(continuations)
  
  rewards = convert_to_numpy(rewards)
  rewards = [np.stack((r, -r), 0) for r in rewards]
  terminals = convert_to_numpy(terminals)
  actions = [np.arange(init_legals.shape[-1])[None, None, ...] + d[..., None] * init_legals.shape[-1] for d in isets]
  
  state_value = np.zeros((2, 1 ))
  for d in range(len(isets) -1, -1, -1):
    
    p1_joint_action_value = rewards[d][0] + state_value[0][continuations[d]]
    p2_joint_action_value = rewards[d][1] + state_value[1][continuations[d]]
    
    p1_action_value = np.sum(p1_joint_action_value * behavior_policy[d][1][:, None, ...], -1)
    p2_action_value = np.sum(p2_joint_action_value * behavior_policy[d][0][..., None], -2)
    
    p1_action_cf_value = p1_action_value * all_reaches[d][1][..., None]
    p2_action_cf_value = p2_action_value * all_reaches[d][0][..., None]
     
    
    p1_iset_action_value = np.bincount(actions[d][0].flatten(), p1_action_cf_value.flatten()).reshape(-1, actions[d].shape[-1])
    p2_iset_action_value = np.bincount(actions[d][1].flatten(), p2_action_cf_value.flatten()).reshape(-1, actions[d].shape[-1])
    
    p1_iset_action_value_masked = np.where(iset_legals[d][0] == 1, p1_iset_action_value, np.min(p1_iset_action_value) - 1)
    p2_iset_action_value_masked = np.where(iset_legals[d][1] == 1, p2_iset_action_value, np.min(p2_iset_action_value) - 1) 
    p1_br_action = np.argmax(p1_iset_action_value_masked, -1)
    p2_br_action = np.argmax(p2_iset_action_value_masked, -1)
    
    p1_history_br = p1_br_action[isets[d][0]]
    p2_history_br = p2_br_action[isets[d][1]]
    
    
    p1_br_policy = np.eye(p1_iset_action_value.shape[-1])[p1_br_action]
    p2_br_policy = np.eye(p2_iset_action_value.shape[-1])[p2_br_action]
    
    for i, iset in enumerate(iset_map[d][0]):
      br_policy_p1[iset] = p1_br_policy[i]
    for i, iset in enumerate(iset_map[d][1]):
      br_policy_p2[iset] = p2_br_policy[i]
    
    p1_history_value = np.squeeze(np.take_along_axis(p1_action_value, p1_history_br[..., None], 1))
    p2_history_value = np.squeeze(np.take_along_axis(p2_action_value, p2_history_br[..., None], 1)) 
    
    state_value = np.stack((p1_history_value, p2_history_value), 0)  
    pass 
  return br_policy_p2, br_policy_p1, state_value[1], state_value[0]

def expected_value_jax_game(game: JaxGame, policy: JaxPolicy):
  key = jax.random.key(0)
  
  init_key, key = jax.random.split(key)
  
  init_state, legals = game.initialize_structures(init_key)
  state_values = {}
  def _traverse_tree(game_state, legal_actions, key, depth=0):
    state, p1_iset, p2_iset, ps = game.get_info(game_state)
    state = np.array(state)
    # ps = np.array(ps)
    p1_iset = np.array(p1_iset)
    p2_iset = np.array(p2_iset)
    
    # ps_str = stringify(ps)
    p1_iset_str = stringify(p1_iset)
    p2_iset_str = stringify(p2_iset) 
    state_str = stringify(state)
    
    p1_policy = policy[p1_iset_str]
    p2_policy = policy[p2_iset_str]
    
    curr_state_value = 0.0
    for a1i, a1 in enumerate(legal_actions[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legal_actions[1]):
        if a2 < 0.5:
          continue
        next_key, action_key = jax.random.split(key)
        new_game_state, new_terminal, new_rewards, new_legals = game.apply_action(
          game_state, action_key, depth, np.array([a1i, a2i]))
        
        curr_state_value += p1_policy[a1i] * p2_policy[a2i] * new_rewards
        if new_terminal:
          continue
          
        next_value = _traverse_tree(new_game_state, new_legals, next_key, depth + 1)
        curr_state_value += p1_policy[a1i] * p2_policy[a2i] * next_value
    state_values[state_str] = curr_state_value
    return curr_state_value
  _traverse_tree(init_state, legals, key)
  return state_values

 
def extract_policy_from_cfr(game: JaxGame, cfr: MuZeroCFR, custom_map: dict = {}) -> JaxPolicy:

  key = jax.random.key(0)
  state_key, init_key = jax.random.split(key)
  game_state, legals = game.initialize_structures(init_key)
  
  policy_dict = JaxPolicy()

  def _traverse_tree(game_state, legal_actions, key, depth=0):
    
    state, p1_iset, p2_iset, ps = game.get_info(game_state)
    # ps_str = stringify(ps)
    p1_iset = np.array(p1_iset)
    p2_iset = np.array(p2_iset)
    p1_iset_str = stringify(p1_iset)
    p2_iset_str = stringify(p2_iset)
    
    
    p1_iset = p1_iset if len(custom_map) == 0 else custom_map[p1_iset_str]
    p2_iset = p2_iset if len(custom_map) == 0 else custom_map[p2_iset_str]
    
    p1_pol = cfr.get_strategy(p1_iset, 0, depth)
    p2_pol = cfr.get_strategy(p2_iset, 1, depth)
  
    p1_pol = np.array(p1_pol) * legal_actions[0]
    p2_pol = np.array(p2_pol) * legal_actions[1]
  
    if np.sum(p1_pol) < 1e-8:
      p1_pol = np.array(legal_actions[0])
    if np.sum(p2_pol) < 1e-8: 
      p2_pol = np.array(legal_actions[1])
  
    p1_pol = p1_pol / np.sum(p1_pol)
    p2_pol = p2_pol / np.sum(p2_pol)
    
    
    policy_dict[p1_iset_str] = p1_pol
    policy_dict[p2_iset_str] = p2_pol
    
    for a1i, a1 in enumerate(legal_actions[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legal_actions[1]):
        if a2 < 0.5:
          continue
        next_key, action_key = jax.random.split(key)
        new_game_state, new_terminal, new_rewards, new_legals = game.apply_action(
            game_state, action_key, depth, np.array([a1i, a2i]))
        
        if new_terminal:
          continue 
        
        _traverse_tree(new_game_state, new_legals, next_key, depth + 1)
        
  _traverse_tree(game_state, legals, state_key)
  return policy_dict
 

# TODO: Change this to use JaxCFR instead of MuZeroCFR
def prepare_cfr_from_game(game: JaxGame, custom_map: dict = {}) -> MuZeroCFR:
  '''
    Prepares a CFR structure that may be used to solve a game.
    The custom_map is a dictionary that maps the information set string into a vector.
  '''
  
  key = jax.random.key(0)
  state_key, init_key = jax.random.split(key)
  game_state, legals = game.initialize_structures(init_key)
   
  
  depth_iset_map  = [] # ID -> Public State concatenate with similarity
  depth_iset_legal  = []

  depth_history_action_utility  = [] # Float[D, H(D), A1, A2]
  depth_history_iset  = [] # Int[D, Pl, H(D)]
  depth_history_actions  = [] # Int[D, Pl, H(D), A] Just indices
  depth_history_legal = [] # Bool[D, Pl, H(D), A] or [D, H(D), A1, A2]

  depth_history_next_history = [] # Int[D, H(D), A1, A2]


  def _traverse_tree(game_state, legal_actions, key, depth=0):
    
    actions = legal_actions[0].shape[0]
    
    state, p1_iset, p2_iset, ps = game.get_info(game_state)
    p1_iset = np.array(p1_iset)
    p2_iset = np.array(p2_iset)
    # ps_str = stringify(ps)
    p1_iset_str = stringify(p1_iset)
    p2_iset_str = stringify(p2_iset)
    
    
    p1_iset = p1_iset if len(custom_map) == 0 else custom_map[p1_iset_str]
    p2_iset = p2_iset if len(custom_map) == 0 else custom_map[p2_iset_str]
    
    
    
    if len(depth_iset_map) <= depth:
      depth_iset_map.append([[], []])
      depth_iset_legal.append([[], []])
      depth_history_action_utility.append([])
      depth_history_iset.append([[], []])
      depth_history_actions.append([[], []])
      depth_history_legal.append([])
      depth_history_next_history.append([])
      
      
    p1_iset_id = len(depth_iset_map[depth][0])
    p2_iset_id = len(depth_iset_map[depth][1])
    for p1_map_iset_id, p1_map_iset in enumerate(depth_iset_map[depth][0]):
      if np.all(p1_map_iset == p1_iset):
        p1_iset_id = p1_map_iset_id
        break
    for p2_map_iset_id, p2_map_iset in enumerate(depth_iset_map[depth][1]):
      if np.all(p2_map_iset == p2_iset):
        p2_iset_id = p2_map_iset_id
        break
      
    if p1_iset_id == len(depth_iset_map[depth][0]):
      depth_iset_map[depth][0].append(p1_iset)
      depth_iset_legal[depth][0].append(legal_actions[0])
    if p2_iset_id == len(depth_iset_map[depth][1]):
      depth_iset_map[depth][1].append(p2_iset)
      depth_iset_legal[depth][1].append(legal_actions[1])
      
    depth_history_iset[depth][0].append(p1_iset_id)
    depth_history_iset[depth][1].append(p2_iset_id)
    depth_history_actions[depth][0].append(np.arange(actions) + p1_iset_id * actions)
    depth_history_actions[depth][1].append(np.arange(actions) + p2_iset_id * actions)
    
    legal_actions_both = np.expand_dims(legal_actions[0], 1) * np.expand_dims(legal_actions[1], 0)
    
    depth_history_legal[depth].append(legal_actions_both)
    depth_history_action_utility[depth].append(np.zeros((actions, actions)))
    depth_history_next_history[depth].append(np.full((actions, actions), -1))
    
    
    for a1i, a1 in enumerate(legal_actions[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legal_actions[1]):
        if a2 < 0.5:
          continue
        next_key, action_key = jax.random.split(key)
        new_game_state, new_terminal, new_rewards, new_legals = game.apply_action(
            game_state, action_key, depth, np.array([a1i, a2i]))

        depth_history_action_utility[depth][-1][a1i, a2i] = new_rewards
        
        
        if new_terminal:
          continue 
        next_history_id = 0
        if len(depth_history_iset) > depth + 1:
          next_history_id = len(depth_history_iset[depth+1][0])
        depth_history_next_history[depth][-1][a1i, a2i] = next_history_id
        _traverse_tree(new_game_state, new_legals, next_key, depth + 1)

  
  
  _traverse_tree(game_state, legals, state_key)
  # print(depth_iset_map)
  constants = MuZeroCFRConstants(
    max_depth = len(depth_iset_map),
    resolving_player = 0,
    init_reaches = np.ones((2, 1)),
    depth_actions = [a[0][0].shape[0] for a in depth_history_actions],
    
    depth_iset_map = convert_player_depth_to_jax(depth_iset_map),
    depth_iset_legal = convert_player_depth_to_jax(depth_iset_legal),
    
    depth_history_action_utility = convert_depth_to_jax(depth_history_action_utility),
    depth_history_iset = convert_depth_to_jax(depth_history_iset),
    depth_history_actions = convert_depth_to_jax(depth_history_actions),
    depth_history_legal = convert_depth_to_jax(depth_history_legal),
    
    depth_history_next_history = convert_depth_to_jax(depth_history_next_history),

  )
  
  return  MuZeroCFR(constants)
   
  
def nash_equilibrium_jax_game(game: JaxGame, iterations: int=4000) -> tuple[MuZeroCFR, JaxPolicy, float]:
  
  cfr = prepare_cfr_from_game(game) 
  cfr.multiple_steps(iterations)
  dict_nash = extract_policy_from_cfr(game, cfr)
  
  return cfr, dict_nash, (cfr.cf_values[0][0] + cfr.cf_values[0][1]) / 2