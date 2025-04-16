import numpy as np
import jax
import jax.numpy as jnp

import pyspiel

from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import prepare_cfr_structure, find_next_root, check_iset_similarity, validate_terminal
from open_spiel.python.algorithms.mu_zero.experiments.utils import stringify
from open_spiel.python.algorithms.mu_zero.mu_zero_train import MuZeroTrain
from open_spiel.python.algorithms.mu_zero.mu_zero_cfr import MuZeroCFR, MuZeroCFRConstants
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game import JaxGame, JaxPolicy
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game_algorithms import prepare_cfr_from_game, extract_policy_from_cfr, nash_equilibrium_cluster_game, exploitability_jax_game


def find_imm_next_isets(game: JaxGame, infos, key, depth: int, player: int):
  isets = {}
  next_infos = {}
  # next_states = []
  for state, legals in infos:
    for a1i, a1 in enumerate(legals[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legals[1]):
        if a2 < 0.5:
          continue
        
        key, action_key = jax.random.split(key)
        game_state, _, _, new_legals = game.apply_action(state, action_key, depth, np.array([a1i, a2i])) 
        
        _, p1_iset, p2_iset, public_state = game.get_info(game_state)
         
        if player == 0:
          iset = p1_iset
        else:
          iset = p2_iset
        isets[stringify(iset)] = (public_state, iset)
        new_info = (game_state, new_legals)
        if stringify(iset) not in next_infos:
          next_infos[stringify(iset)] = []
        next_infos[stringify(iset)].append(new_info)
        # next_states.append(new_info)
  return isets, next_infos
   

def solve_game_each_infoset(model: MuZeroTrain, depth_limit: int, resolve_iterations: int) -> dict[str, list[float]]:
  '''Constructs the depth-limited game from each infoset in the game. It traverses the game tree in BFS fashion, so you have cf-values and reaches for each subgame.'''
  
  game = model.game
  policy = {}
  
   
  def _find_policy(cfr: MuZeroCFR, infos, key, player: int, depth: int = 0):
    print(depth)
    # TODO: THis is not good
    construct_gadget = cfr.constants.depth_iset_legal[0][0].shape[1] == 2
    cfr.multiple_steps(resolve_iterations)
    
    for state, _ in infos:
      _, p1_iset, p2_iset, public_state = game.get_info(state)
      iset = p1_iset if player == 0 else p2_iset
      abstracted_iset = model.get_abstraction(public_state, iset, player)
      policy[stringify(iset)] = cfr.get_strategy(abstracted_iset, player, int(construct_gadget)) 
      print(policy[stringify(iset)]) 
    if depth + 2 >= game.cards:
      return
    key, next_key = jax.random.split(key) 
    next_isets, next_possible_infos = find_imm_next_isets(game, infos, next_key, depth, player)
    for iset_string, (public_state, iset_tensor) in next_isets.items():
      if iset_string in policy:
        continue
      abstracted_iset = model.get_abstraction(public_state, iset_tensor, player)
      # public_state_histories = cfr.find_public_state_from_iset(abstracted_iset, player, 1 + construct_gadget)
      next_abstracted_isets, next_reaches, next_cf_values = find_next_root(cfr, 1 + construct_gadget, player, public_state, abstracted_iset)
      next_infos = next_possible_infos[iset_string]

      # print("Next states: ", len(next_states))
      # print("Next CF values: ", len(next_cf_values))
      next_cfr = prepare_cfr_structure(model, player, depth_limit, next_abstracted_isets, next_reaches, next_cf_values, len(next_cf_values) > 1)
      
      # TODO: Next states should be only those that are in the next iset. 
      _find_policy(next_cfr, next_infos, key, player, depth + 1)
      del next_cfr
       
  
  init_reaches = np.ones((2, 1))
  init_cf_values = np.zeros((1, ))
  
  #can be arbitrary seed, as this game does not have chance nodes
  key = jax.random.key(0)
  state_key, init_key = jax.random.split(key)
  game_state, legals = game.initialize_structures(init_key) 
  _, init_p1_iset, init_p2_iset, init_ps = game.get_info(game_state)
  init_info = (game_state, legals)
  
  p1_abstracted, p2_abstracted = model.get_both_abstraction(init_ps, init_p1_iset, init_p2_iset)
  abs1, abs2, probs1, probs2, _, _ = model.get_both_similarities_and_probs(init_ps, init_p1_iset, init_p2_iset)
  # print(probs1)
  # print(probs2)
  
  init_iset = np.expand_dims(np.stack([p1_abstracted, p2_abstracted], axis=0), 1)
  for pl in range(2):
    init_cfr = prepare_cfr_structure(model, pl, depth_limit, init_iset, init_reaches, init_cf_values, False)
    _find_policy(init_cfr, [init_info], state_key, pl)
    # _find_policy(init_cfr, [init_info], pl, init_reaches, init_cf_values)
  return JaxPolicy(policy)
        
    
def solve_game_full(model: MuZeroTrain, resolve_iterations: int = 1000) -> dict[str, list[float]]:
  '''Constructs the game from the dynamics and solves it in that vain.'''
  game = model.game
  init_reaches = np.ones((2, 1))
  init_cf_values = np.zeros((1, ))
  
  #can be arbitrary seed, as this game does not have chance nodes
  key = jax.random.key(0)
  state_key, init_key = jax.random.split(key)
  game_state, legals = game.initialize_structures(init_key) 
  _, init_p1_isets, init_p2_iset, init_ps = game.get_info(game_state) 
  
  p1_abstracted, p2_abstracted = model.get_both_abstraction(init_ps, init_p1_isets, init_p2_iset)

  init_iset = np.expand_dims(np.stack([p1_abstracted, p2_abstracted], axis=0), 1) 
  
  cfr = prepare_cfr_structure(model, 0, model.config.trajectory_max, init_iset, init_reaches, init_cf_values, False)  
  
  cfr.multiple_steps(resolve_iterations)  
   
  policy = JaxPolicy()
  
  
  # TODO: Is this okay? I think this should just go through the game tree within the CFR structures and take it from there.
  #   Right now it just maps the original game to the abstraction and uses that.
  def _traverse_game(game_state, key, legals, depth=0): 
    _, p1_iset, p2_iset, ps = game.get_info(game_state)
    p1_abstracted, p2_abstracted = model.get_both_abstraction(ps, p1_iset, p2_iset)
    p1_strategy = cfr.get_strategy(p1_abstracted, 0, depth)
    p2_strategy = cfr.get_strategy(p2_abstracted, 1, depth)
    policy[stringify(p1_iset)] = p1_strategy
    policy[stringify(p2_iset)] = p2_strategy
    for a1i, a1 in enumerate(legals[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legals[1]):
        if a2 < 0.5:
          continue
        next_key, action_key = jax.random.split(key)
        new_game_state, terminal, rewards, new_legals = game.apply_action(game_state, action_key, depth, np.array([a1i, a2i]))
        if terminal:
          continue
        _traverse_game(new_game_state, next_key, new_legals, depth + 1)
        
  _traverse_game(game_state, state_key, legals)
  
  return policy

def solve_game_full_no_dynamics(model: MuZeroTrain, resolve_iterations: int = 1000) -> JaxPolicy:
  
  '''Constructs the original game but use trained abstraction and then solves it.'''
  game = model.game 
  
  #can be arbitrary seed, as this game does not have chance nodes
  key = jax.random.key(0)
  state_key, init_key = jax.random.split(key)
  game_state, legals = game.initialize_structures(init_key) 
  
  cluster_map = {}
  def _traverse_game(game_state, key, legals, depth=0): 
    _, p1_iset, p2_iset, ps = game.get_info(game_state)
    p1_abstracted, p2_abstracted = model.get_both_abstraction(ps, p1_iset, p2_iset) 
    
    p1_iset = np.array(p1_iset)
    p2_iset = np.array(p2_iset)
    p1_abstracted = np.array(p1_abstracted)
    p2_abstracted = np.array(p2_abstracted)
     
    cluster_map[stringify(p1_iset)] = np.concatenate((ps, p1_abstracted))
    cluster_map[stringify(p2_iset)] = np.concatenate((ps, p2_abstracted))
    for a1i, a1 in enumerate(legals[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legals[1]):
        if a2 < 0.5:
          continue
        next_key, action_key = jax.random.split(key)
        new_game_state, terminal, rewards, new_legals = game.apply_action(game_state, action_key, depth, np.array([a1i, a2i]))
        if terminal:
          continue
        _traverse_game(new_game_state, next_key, new_legals, depth + 1)
        
  _traverse_game(game_state, state_key, legals)
  
  policy = nash_equilibrium_cluster_game(game, resolve_iterations, cluster_map) 
  
  return policy


def solve_game_full_trained_dynamics(model: MuZeroTrain, resolve_iterations: int = 1000) -> JaxPolicy:
  
  '''Constructs the original game but use trained abstraction and then solves it.'''
  game = model.game 
  
  #can be arbitrary seed, as this game does not have chance nodes
  key = jax.random.key(0)
  state_key, init_key = jax.random.split(key)
  game_state, legals = game.initialize_structures(init_key) 
  
  cluster_map = {}
  def _traverse_game(game_state, p1_abstracted, p2_abstracted, key, legals, depth=0): 
    _, p1_iset, p2_iset, ps = game.get_info(game_state)
    p1_iset = np.array(p1_iset)
    p2_iset = np.array(p2_iset)
     
    cluster_map[stringify(p1_iset)] = np.concatenate((ps, p1_abstracted))
    cluster_map[stringify(p2_iset)] = np.concatenate((ps, p2_abstracted))
    for a1i, a1 in enumerate(legals[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legals[1]):
        if a2 < 0.5:
          continue
        
        
        next_key, action_key = jax.random.split(key)
        new_game_state, terminal, rewards, new_legals = game.apply_action(game_state, action_key, depth, np.array([a1i, a2i]))
         
        next_p1_isets, next_p2_isets, next_utilities, next_terminal = model.get_next_state_from_abstraction(p1_abstracted, p2_abstracted, a1i, a2i) 
        
        if terminal:
          continue
        _traverse_game(new_game_state, next_p1_isets, next_p2_isets, next_key, new_legals, depth + 1)
        
        
  _, p1_iset, p2_iset, ps = game.get_info(game_state)
  p1_abstracted, p2_abstracted = model.get_both_abstraction(ps, p1_iset, p2_iset) 
  
  _traverse_game(game_state, p1_abstracted, p2_abstracted, state_key, legals)
  
  policy = nash_equilibrium_cluster_game(game, resolve_iterations, cluster_map) 
  
  return policy
 

def solve_game_replace_legals(model: MuZeroTrain, resolve_iterations: int = 1000, use_dynamics: bool = False) -> JaxPolicy:
  '''This only makes some legals illegal, the other way around is much more difficult, because you would have some actions that lead to terminal that gives 0 and that would probably break the game.'''
  game = model.game 
  
  #can be arbitrary seed, as this game does not have chance nodes
  key = jax.random.key(0)
  state_key, init_key = jax.random.split(key)
  game_state, legals = game.initialize_structures(init_key) 
  
  cfr = prepare_cfr_from_game(game)
  
  iset_legals = [[np.array(pl) for pl in depth] for depth in cfr.constants.depth_iset_legal]
  
  cluster_map = {}
  legal_actions = []
  
  legal_epsilon = 0.0003
  
  def _traverse_game(game_state, p1_abstracted, p2_abstracted, key, legals, depth=0): 
    
    if len(legal_actions) <= depth:
      legal_actions.append([])
    
     
    _, p1_iset, p2_iset, ps = game.get_info(game_state)
    
    if use_dynamics:
      p1_legal_logits, p2_legal_logits = model.get_both_legal_actions_from_abstraction(p1_abstracted, p2_abstracted)
    else:
      p1_legal_logits, p2_legal_logits = model.get_both_legal_actions(ps, p1_iset, p2_iset)
    
    p1_legal = jax.nn.sigmoid(p1_legal_logits) > legal_epsilon 
    p2_legal = jax.nn.sigmoid(p2_legal_logits) > legal_epsilon
     
    # assert np.all(iset_legals[depth][0][cfr.constants.depth_history_iset[depth][0][len(legal_actions[depth])]] >= p1_legal)
    # assert np.all(iset_legals[depth][1][cfr.constants.depth_history_iset[depth][1][len(legal_actions[depth])]] >= p2_legal)
    
    
    history_both_legals = p1_legal[..., None] * p2_legal[None, ...]  
    
    history_both_legals = np.where(history_both_legals + cfr.constants.depth_history_legal[depth][len(legal_actions[depth])] > 1.5, 1, 0)
    
    legal_actions[depth].append(history_both_legals)
    
  
    for a1i, a1 in enumerate(legals[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legals[1]):
        if a2 < 0.5:
          continue
        
        
        
        
        next_key, action_key = jax.random.split(key)
        new_game_state, terminal, rewards, new_legals = game.apply_action(game_state, action_key, depth, np.array([a1i, a2i]))
         
        next_p1_isets, next_p2_isets, next_utilities, next_terminal = model.get_next_state_from_abstraction(p1_abstracted, p2_abstracted, a1i, a2i) 
        
        if terminal:
          continue
        _traverse_game(new_game_state, next_p1_isets, next_p2_isets, next_key, new_legals, depth + 1)
  
  _, p1_iset, p2_iset, ps = game.get_info(game_state)
  p1_abstracted, p2_abstracted = model.get_both_abstraction(ps, p1_iset, p2_iset)
  _traverse_game(game_state, p1_abstracted, p2_abstracted, state_key, legals)
  
  legal_actions = [jnp.array(la) for la in legal_actions]
  
  
  constants = MuZeroCFRConstants(
    max_depth = cfr.constants.max_depth,
    resolving_player = cfr.constants.resolving_player,
    init_reaches = cfr.constants.init_reaches,
    depth_actions = cfr.constants.depth_actions,
    depth_iset_map = cfr.constants.depth_iset_map,
    depth_iset_legal = cfr.constants.depth_iset_legal,
    depth_history_action_utility = cfr.constants.depth_history_action_utility,
    depth_history_iset = cfr.constants.depth_history_iset,
    depth_history_actions = cfr.constants.depth_history_actions,
    depth_history_legal = legal_actions,
    depth_history_next_history = cfr.constants.depth_history_next_history
  )
  
  cfr = MuZeroCFR(constants)
  
  cfr.multiple_steps(resolve_iterations)
  policy = extract_policy_from_cfr(model.game, cfr)
  return policy

def extract_rnad_policy(model: MuZeroTrain) -> dict[str, list[float]]:
  
  game = model.game
  isets = []
  iset_legals = []
  iset_str = {}
  
  # TODO: Could we somehow create traverse_game as a lambda with some info parameter so we do not copy it into every function we neeed with slight changes
  def _traverse_game(game_state, key, legals, depth=0): 
    _, p1_iset, p2_iset, ps = game.get_info(game_state)
    if not stringify(p1_iset) in iset_str:
      iset_str[stringify(p1_iset)] = len(isets)
      isets.append(p1_iset)
      iset_legals.append(legals[0])
    if not stringify(p2_iset) in iset_str:
      iset_str[stringify(p2_iset)] = len(isets)
      isets.append(p2_iset)
      iset_legals.append(legals[1])
      
    for a1i, a1 in enumerate(legals[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legals[1]):
        if a2 < 0.5:
          continue
        next_key, action_key = jax.random.split(key)
        new_game_state, terminal, rewards, new_legals = game.apply_action(game_state, action_key, depth, np.array([a1i, a2i]))
        if terminal:
          continue
        _traverse_game(new_game_state, next_key, new_legals, depth + 1)
        
  key = jax.random.key(0)
  state_key, init_key = jax.random.split(key)
  game_state, legals = game.initialize_structures(init_key) 
  _traverse_game(game_state, state_key, legals)
  isets = np.array(isets)
  iset_legals = np.array(iset_legals)
  pi = model._jit_get_policy(model.network_parameters.rnad_params_target, isets, iset_legals)
  pi = np.array(pi)
  policy_dict = {}
  for str_i, iset_id in iset_str.items():
    policy_dict[str_i] = pi[iset_id]
  return policy_dict

def solve_game_abstraction_legals(model: MuZeroTrain, resolve_iterations: int = 1000, use_dynamics: bool = False) -> JaxPolicy:
  '''Constructs the game using real infosets mapped to abstractions, preserving legal actions from the original game.'''
  
  # TODO: I think this method won't work with use_dynamics = False, because it may happen that in _traverse_for_legals some abstracted infosets may not have any legal actions (so they won't be in the abstraction_legals)
  
  game = model.game
  
  # Initialize with arbitrary seed since game has no chance nodes
  key = jax.random.key(0)
  state_key, init_key = jax.random.split(key)
  game_state, legals = game.initialize_structures(init_key)
  
  # Maps abstracted infosets to their legal actions
  # abstraction_legals = {}
  depth_abstraction_legals = []
  depth_abstraction_legals_map =[]
  
  iset_threshold_similarity = 1e-4
  
  cluster_map = {}
  # TODO: This could probably be merged with the traverse_for_cfr
  def _traverse_for_legals(game_state, p1_abstracted, p2_abstracted, key, legals, depth=0): 
    
    _, p1_iset, p2_iset, ps = game.get_info(game_state)
    if not use_dynamics:
      p1_abstracted, p2_abstracted = model.get_both_abstraction(ps, p1_iset, p2_iset)
      p1_abstracted = np.array(p1_abstracted)
      p2_abstracted = np.array(p2_abstracted)
    
    p1_iset = np.array(p1_iset)
    p2_iset = np.array(p2_iset) 
    
    cluster_map[stringify(p1_iset)] = np.array(p1_abstracted)
    cluster_map[stringify(p2_iset)] = np.array(p2_abstracted)
    
    # Update info for abstracted infosets
    # p1_key = stringify(p1_abstracted)
    # p2_key = stringify(p2_abstracted)
    
    
    # If an action is legal in the real infoset, mark it as legal in the abstraction
    if len(depth_abstraction_legals) <= depth:
      depth_abstraction_legals.append([[], []])
      depth_abstraction_legals_map.append([[], []])
      
    p1_id = -1
    p2_id = -1
     
    for i in range(len(depth_abstraction_legals[depth][0])):
      if check_iset_similarity(depth_abstraction_legals[depth][0][i], p1_abstracted, iset_threshold_similarity):
        if p1_id > -1:
          print("Warning: Multiple p1 infosets found for the same real infoset")
        p1_id = i
        
        # break
    for i in range(len(depth_abstraction_legals[depth][1])):
      if check_iset_similarity(depth_abstraction_legals[depth][1][i], p2_abstracted, iset_threshold_similarity):
        if p2_id > -1:
          print("Warning: Multiple p2 infosets found for the same real infoset")
        p2_id = i
        # break
    if p1_id == -1:
      p1_id = len(depth_abstraction_legals[depth][0])
      depth_abstraction_legals[depth][0].append(p1_abstracted)
      depth_abstraction_legals_map[depth][0].append(np.zeros((model.actions,)))
    if p2_id == -1:
      p2_id = len(depth_abstraction_legals[depth][1])
      depth_abstraction_legals[depth][1].append(p2_abstracted)
      depth_abstraction_legals_map[depth][1].append(np.zeros((model.actions,)))
    depth_abstraction_legals_map[depth][0][p1_id] = np.maximum(depth_abstraction_legals_map[depth][0][p1_id], legals[0])  
    depth_abstraction_legals_map[depth][1][p2_id] = np.maximum(depth_abstraction_legals_map[depth][1][p2_id], legals[1])
    
    
    # Traverse the game tree
    for a1i, a1 in enumerate(legals[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legals[1]):
        if a2 < 0.5:
          continue
        
        next_key, action_key = jax.random.split(key)
        new_game_state, terminal, rewards, new_legals = game.apply_action(game_state, action_key, depth, np.array([a1i, a2i]))
        if use_dynamics:
          next_p1_abstracted, next_p2_abstracted, next_utilities, next_terminal = model.get_next_state_from_abstraction(p1_abstracted, p2_abstracted, a1i, a2i)  
          next_p1_abstracted = np.array(next_p1_abstracted)
          next_p2_abstracted = np.array(next_p2_abstracted)
          
          
        if terminal:
          continue
            
        # Store the action and next state info 
        
        _traverse_for_legals(new_game_state, next_p1_abstracted, next_p2_abstracted, next_key, new_legals, depth + 1)
  
  # Traverse the game tree to build abstraction_info
  
  _, p1_iset, p2_iset, ps = game.get_info(game_state)
  init_p1_abstracted, init_p2_abstracted = model.get_both_abstraction(ps, p1_iset, p2_iset)
  init_p1_abstracted = np.array(init_p1_abstracted)
  init_p2_abstracted = np.array(init_p2_abstracted)
  _traverse_for_legals(game_state, init_p1_abstracted, init_p2_abstracted, state_key, legals)
   
   
  
  # Build CFR structure from scratch
  max_depth = model.config.trajectory_max
  
  depth_iset_legal = []
  depth_iset_map = [] 
  depth_history_iset = []
  depth_history_actions = []
  depth_history_legal = []
  depth_history_next_history = []
  depth_history_action_utility = [] 
  
  
  def create_iset_map(curr_iset, depth):
    isets = [[], []]
    iset_map = [[], []]
    iset_legals = [[], []]
    history_legals =  [[], []]
    for pl in range(curr_iset.shape[0]):
      first_iset_id = len(iset_map[pl])
      for i in range(curr_iset.shape[1]): 
        curr_index = -1
        for j in range(first_iset_id, len(iset_map[pl])):
          if check_iset_similarity(iset_map[pl][j], curr_iset[pl, i], iset_threshold_similarity):
            curr_index = j
            break
        if curr_index < 0:
          curr_index = len(iset_map[pl])
          iset_map[pl].append(curr_iset[pl, i])
          abstraction_id = -1
          for j in range(len(depth_abstraction_legals[depth][pl])):
            # print(compare_isets(depth_abstraction_legals[depth][pl][j], curr_iset[pl, i], 1e-5))
            if check_iset_similarity(depth_abstraction_legals[depth][pl][j], curr_iset[pl, i], iset_threshold_similarity):
              if abstraction_id > -1:
                print("Warning: Multiple abstraction infosets found for the same real infoset")
              abstraction_id = j 
          if abstraction_id < 0:
            print("Warning: No abstraction found for iset in depth", depth)
            legals = np.zeros((model.actions,)) 
            # assert False
            
          legals = depth_abstraction_legals_map[depth][pl][abstraction_id] 
          iset_legals[pl].append(legals)
        isets[pl].append(curr_index)
        history_legals[pl].append(iset_legals[pl][curr_index])
    isets = np.array(isets)
    actions = isets[..., None] * model.actions + np.arange(model.actions)[None, None, ...] 
    iset_map = [np.array(i) for i in iset_map]
    history_legals = np.array(history_legals)
    return iset_map, isets, actions, iset_legals, history_legals
   
  def _traverse_for_cfr(curr_iset, depth= 0):
    if depth == max_depth:
      return
    
    iset_map, isets, actions, iset_legals, history_legals = create_iset_map(curr_iset, depth)
    vectorized_abstraction = jax.vmap(jax.vmap(model.get_next_state_from_abstraction, in_axes=(None, None, -1, -1), out_axes=(-2, -2, -2, -2)), in_axes=(None, None, -1, -1), out_axes=(-2, -2, -2, -2))
    
    p2_actions = np.tile(np.arange(model.actions), (curr_iset.shape[1], model.actions, 1))
    p1_actions = np.transpose(p2_actions, (0, 2, 1))
    
    next_p1_abstracted, next_p2_abstracted, next_utilities, next_terminal = vectorized_abstraction(curr_iset[0], curr_iset[1], p1_actions, p2_actions)
    
    legal = history_legals[0][..., None] * history_legals[1][..., None, :]
    
    non_terminal = legal.nonzero()
    
    next_isets = np.stack([next_p1_abstracted[non_terminal], next_p2_abstracted[non_terminal]], axis=0)
    
    
    action_utility = next_utilities[..., 0] * legal
    
    if depth == max_depth - 1:
      next_history = np.cumsum(legal).reshape(legal.shape) * legal - 1
    else:
      next_history = np.full_like(legal, -1)
    
    
    depth_iset_map.append(iset_map)
    depth_iset_legal.append(iset_legals)
    depth_history_action_utility.append(action_utility)
    depth_history_iset.append(isets)
    depth_history_actions.append(actions)
    depth_history_legal.append(legal)  
    depth_history_next_history.append(next_history.astype(int))
    
    _traverse_for_cfr(next_isets, depth + 1)
    
  _traverse_for_cfr(np.array([[init_p1_abstracted], [init_p2_abstracted]]), 0)
  
  
  
  # Create CFR constants
  constants = MuZeroCFRConstants(
    resolving_player=0,
    init_reaches=np.ones((2, 1)),
    depth_actions=np.full((max_depth,), model.actions), 
    depth_iset_legal=[[np.array(p) for p in d] for d in depth_iset_legal],
    depth_history_action_utility=[np.array(p) for p in depth_history_action_utility],
    depth_history_iset=[np.array(p) for p in depth_history_iset],
    depth_history_actions=[np.array(p) for p in depth_history_actions],
    depth_history_legal=[np.array(p) for p in depth_history_legal],
    depth_history_next_history=[np.array(p) for p in depth_history_next_history]
  )
  
  # Create and run CFR
  
  depth_iset_map = [[np.array(p) for p in d] for d in depth_iset_map]
  cfr = MuZeroCFR(constants, depth_iset_map)
  cfr.multiple_steps(resolve_iterations)
  
  
  
  # Extract and return the policy
  policy = extract_policy_from_cfr(game, cfr, cluster_map)
  return policy


# TODO: Maybe we could normalize reaches per public state.
def solve_game_per_depth(model: MuZeroTrain, resolve_iterations: int = 1000, depth_limit: int =1) -> JaxPolicy:
  '''
  Solves the game depth by depth, creating subgames at each level.
  
  This function works by:
  1. Starting at the root and creating an initial subgame with a depth limit
  2. Solving this subgame using CFR
  3. Creating a new subgame rooted at the next depth
  4. Repeating until reaching the leaf depth
  
  Args:
    model: The MuZero model to use for abstractions
    resolve_iterations: Number of CFR iterations to run at each depth
    use_dynamics: Whether to use the dynamics model for state transitions
    max_depth: Maximum depth to consider (defaults to model.max_depth)
    
  Returns:
    A JaxPolicy representing the solution
  '''

  game = model.game 
  
  # Initialize with arbitrary seed since game has no chance nodes
  key = jax.random.key(0)
  state_key, init_key = jax.random.split(key)
  game_state, legals = game.initialize_structures(init_key)
  
  # Get initial abstractions
  _, p1_iset, p2_iset, ps = game.get_info(game_state)
  init_p1_abstracted, init_p2_abstracted = model.get_both_abstraction(ps, p1_iset, p2_iset)
   
  
  # Maps real infosets to their abstractions
  
  # Final policy to return
  
  average_policies = [[], []]
  iset_maps = [[], []]

  
  def create_iset_map(curr_iset):
    isets = [[], []]
    iset_map = [[], []]
    for pl in range(curr_iset.shape[0]):
      first_iset_id = len(iset_map[pl])
      for i in range(curr_iset.shape[1]): 
        curr_index = -1
        for j in range(first_iset_id, len(iset_map[pl])):
          if check_iset_similarity(iset_map[pl][j], curr_iset[pl, i]):
            curr_index = j
            break
        if curr_index < 0:
          curr_index = len(iset_map[pl])
          iset_map[pl].append(curr_iset[pl, i])  
        isets[pl].append(curr_index)
    isets = np.array(isets)
     
    iset_map = [np.array(i) for i in iset_map]
    return iset_map, isets
  
  def handle_gadget_layer(isets, iset_ids, iset_map, p1_cf_values, p2_cf_values):
    
    gadget_actions = 2
    
    assert p1_cf_values.shape[0] == p2_cf_values.shape[0]
    
    p1_action_utilities = np.zeros((p1_cf_values.shape[0], gadget_actions, gadget_actions))
    p1_action_utilities[:, 0, 0] = p1_cf_values
    
    p2_action_utilities = np.zeros((p2_cf_values.shape[0], gadget_actions, gadget_actions))
    p2_action_utilities[:, 0, 0] = p2_cf_values
    
    
    p1_iset_legal = [np.ones(iset_map[pl].shape[:-1] + (gadget_actions,)) for pl in range(2)]
    p1_iset_legal[0][:, 1] = 0
    p2_iset_legal = [np.ones(iset_map[pl].shape[:-1] + (gadget_actions,)) for pl in range(2)]
    p2_iset_legal[1][:, 1] = 0 
    
    p1_legals = np.ones((p1_cf_values.shape[0], gadget_actions, gadget_actions))
    p1_legals[:, 1, :] = 0
    p2_legals = np.ones((p2_cf_values.shape[0], gadget_actions, gadget_actions))
    p2_legals[:, :, 1] = 0
    
    
    p1_next_history = np.full((p1_cf_values.shape[0], gadget_actions, gadget_actions), -1, dtype = int)
    p1_next_history[:, 0, 1] = np.arange(p1_cf_values.shape[0])
    
    p2_next_history = np.full((p2_cf_values.shape[0], gadget_actions, gadget_actions), -1, dtype = int)
    p2_next_history[:, 1, 0] = np.arange(p2_cf_values.shape[0])
    
    
    history_actions = iset_ids[..., None] * gadget_actions + np.arange(gadget_actions)[None, None, ...] 
    
    p1_depth_iset_map.append(iset_map)
    p1_depth_iset_legal.append(p1_iset_legal)
    p1_depth_history_action_utility.append(p1_action_utilities)
    p1_depth_history_iset.append(iset_ids)
    p1_depth_history_actions.append(history_actions)
    p1_depth_history_legal.append(p1_legals)
    p1_depth_history_next_history.append(p1_next_history)
    
    p2_depth_iset_map.append(iset_map)
    p2_depth_iset_legal.append(p2_iset_legal)
    p2_depth_history_action_utility.append(p2_action_utilities)
    p2_depth_history_iset.append(iset_ids)
    p2_depth_history_actions.append(history_actions)
    p2_depth_history_legal.append(p2_legals)
    p2_depth_history_next_history.append(p2_next_history)
    
    depth_history_isets_full.append(isets)
    
    handle_single_layer(isets, iset_ids, iset_map, 0)
  
  def handle_single_layer(isets, iset_ids, iset_map, depth:int):
    p1_iset_legals = model._jit_get_legal_actions(model.network_parameters.legal_actions_params[0], iset_map[0])
    p2_iset_legals = model._jit_get_legal_actions(model.network_parameters.legal_actions_params[1], iset_map[1])
    
    p1_iset_legals = np.array(p1_iset_legals)
    p2_iset_legals = np.array(p2_iset_legals)
    
    p1_iset_legals = np.where(p1_iset_legals > 0, 1.0, 0.0)
    p2_iset_legals = np.where(p2_iset_legals > 0, 1.0, 0.0)
    iset_legals = [p1_iset_legals, p2_iset_legals]
    
    p1_history_legals, p2_history_legals = model.get_both_legal_actions_from_abstraction(isets[0], isets[1])
    
    p1_history_legals = np.array(p1_history_legals)
    p2_history_legals = np.array(p2_history_legals)
    
    p1_history_legals = np.where(p1_history_legals > 0, 1.0, 0.0)
    p2_history_legals = np.where(p2_history_legals > 0, 1.0, 0.0)
    
    history_legals = p1_history_legals[..., None] * p2_history_legals[..., None, :]
    
    vectorized_abstraction = jax.vmap(jax.vmap(model.get_next_state_from_abstraction, in_axes=(None, None, -1, -1), out_axes=(-2, -2, -2, -2)), in_axes=(None, None, -1, -1), out_axes=(-2, -2, -2, -2))
    
    p2_actions = np.tile(np.arange(model.actions), (isets.shape[1], model.actions, 1))
    p1_actions = np.transpose(p2_actions, (0, 2, 1))
    
    next_p1_abstracted, next_p2_abstracted, next_utilities, next_terminal = vectorized_abstraction(isets[0], isets[1], p1_actions, p2_actions)
    
    action_utility = next_utilities[..., 0] * history_legals
    
    non_terminal = np.squeeze(validate_terminal(next_terminal), -1) * history_legals
    nonzeros = non_terminal.nonzero()
    next_isets = np.stack((next_p1_abstracted[nonzeros], next_p2_abstracted[nonzeros]), 0)
    
    # This should be -1 everywhere, except the part where you have next history. Therey ou go by terminal and just add 1
    next_history = (np.cumsum(non_terminal).reshape(non_terminal.shape) * non_terminal) - 1
    
    history_actions = iset_ids[..., None] * model.actions + np.arange(model.actions)[None, None, ...] 
    
    p1_depth_iset_map.append(iset_map)
    p1_depth_iset_legal.append(iset_legals)
    p1_depth_history_action_utility.append(action_utility)
    p1_depth_history_iset.append(iset_ids)
    p1_depth_history_actions.append(history_actions)
    p1_depth_history_legal.append(history_legals)
    p1_depth_history_next_history.append(next_history.astype(int))
    
    p2_depth_iset_map.append(iset_map)
    p2_depth_iset_legal.append(iset_legals)
    p2_depth_history_action_utility.append(action_utility)
    p2_depth_history_iset.append(iset_ids)
    p2_depth_history_actions.append(history_actions)
    p2_depth_history_legal.append(history_legals)
    p2_depth_history_next_history.append(next_history.astype(int))
    
    depth_history_isets_full.append(isets)
    if np.all(next_history < 0):
      return
    
    next_iset_map, next_iset_ids = create_iset_map(next_isets)
      
    if depth == depth_limit -1:
      handle_mvs_layer(next_isets, next_iset_ids, next_iset_map)
    else:
      handle_single_layer(next_isets, next_iset_ids, next_iset_map, depth + 1)
  
  def handle_mvs_layer(isets, iset_ids, iset_map): 
    mvs_actions = model.config.transformations + 1
    mvs_vals = model.get_mvs_from_abstraction(isets[0], isets[1])
    iset_legal = [np.ones(iset_map[pl].shape[:-1] + (mvs_actions,)) for pl in range(2)]  
    legal = np.ones_like(mvs_vals)
    next_history = np.full_like(mvs_vals, -1, dtype=int)
    

    actions = iset_ids[..., None] * mvs_actions + np.arange(mvs_actions)[None, None, ...]
    
    p1_depth_iset_map.append(iset_map)
    p1_depth_iset_legal.append(iset_legal)
    p2_depth_iset_map.append(iset_map)
    p2_depth_iset_legal.append(iset_legal)
    
    p1_depth_history_action_utility.append(mvs_vals)
    p1_depth_history_iset.append(iset_ids)
    p1_depth_history_actions.append(actions) 
    p1_depth_history_legal.append(legal) 
    p1_depth_history_next_history.append(next_history)
    
    p2_depth_history_action_utility.append(mvs_vals)
    p2_depth_history_iset.append(iset_ids)
    p2_depth_history_actions.append(actions) 
    p2_depth_history_legal.append(legal) 
    p2_depth_history_next_history.append(next_history)
    
    depth_history_isets_full.append(isets)
  
  def extract_policy(cfr, iset_map, player, depth):
    average_policies[player].append(cfr.averages[depth][player])
    iset_maps[player].append(iset_map[depth][player])
    
  
  first_isets = np.array([[init_p1_abstracted], [init_p2_abstracted]]) # Int[D, Pl, H(D)]
  first_iset_map = [np.array([init_p1_abstracted]), np.array([init_p2_abstracted])] # Int[D, Pl, I]
  first_iset_ids = np.array([[0], [0]], dtype=int) # Int[D, Pl, H(D)]
  p1_cf_values = np.zeros((1, ))
  p2_cf_values = np.zeros((1, ))
  p1_reaches = np.ones((2, 1))
  p2_reaches = np.ones((2, 1))
  
  # For each depth level
  # TODO: Change this so it takes the max length from the game, or that it terminates when there are only terminals
  for current_depth in range(game.max_trajectory_length()):
    print(f"Solving in depth {current_depth}")
    
    p1_depth_iset_map = []
    p2_depth_iset_map = []
    p1_depth_iset_legal = []
    p2_depth_iset_legal = []
    p1_depth_history_action_utility = []
    p2_depth_history_action_utility = []
    p1_depth_history_iset = []
    p2_depth_history_iset = []
    p1_depth_history_actions = []
    p2_depth_history_actions = []
    p1_depth_history_legal = []
    p2_depth_history_legal = []
    p1_depth_history_next_history = []
    p2_depth_history_next_history = []
    
    depth_history_isets_full = []

    # Difference between the players is only in the gadget layer
    
    if current_depth == 0:
      handle_single_layer(first_isets, first_iset_ids, first_iset_map, 0)
    else:
      handle_gadget_layer(first_isets, first_iset_ids, first_iset_map, p1_cf_values, p2_cf_values)
  
    actions = [d.shape[-1] for d in p1_depth_history_actions]
    # Create CFR constants for this depth-limited subgame
    p1_constants = MuZeroCFRConstants(
      resolving_player=0,
      init_reaches=p1_reaches,
      depth_actions=actions,
      depth_iset_legal=[[jnp.array(p) for p in d] for d in p1_depth_iset_legal],
      depth_history_action_utility=[jnp.array(p) for p in p1_depth_history_action_utility],
      depth_history_iset=[jnp.array(p) for p in p1_depth_history_iset],
      depth_history_actions=[jnp.array(p) for p in p1_depth_history_actions],
      depth_history_legal=[jnp.array(p) for p in p1_depth_history_legal],
      depth_history_next_history=[jnp.array(p) for p in p1_depth_history_next_history]
    )
    
    p2_constants = MuZeroCFRConstants(
      resolving_player=1,
      init_reaches=p2_reaches,
      depth_actions=actions,
      depth_iset_legal=[[jnp.array(p) for p in d] for d in p2_depth_iset_legal],
      depth_history_action_utility=[jnp.array(p) for p in p2_depth_history_action_utility],
      depth_history_iset=[jnp.array(p) for p in p2_depth_history_iset],
      depth_history_actions=[jnp.array(p) for p in p2_depth_history_actions],
      depth_history_legal=[jnp.array(p) for p in p2_depth_history_legal],
      depth_history_next_history=[jnp.array(p) for p in p2_depth_history_next_history]
    )
    
    # Create and run CFR for this depth-limited subgame
    p1_depth_iset_map_jax = [[jnp.array(p) for p in d] for d in p1_depth_iset_map]
    p2_depth_iset_map_jax = [[jnp.array(p) for p in d] for d in p2_depth_iset_map]
    p1_cfr = MuZeroCFR(p1_constants, p1_depth_iset_map_jax)
    p2_cfr = MuZeroCFR(p2_constants, p2_depth_iset_map_jax)
    p1_cfr.multiple_steps(resolve_iterations)
    p2_cfr.multiple_steps(resolve_iterations)
    
    # Gadget or no gadget
    if current_depth == 0:
      
      # policy = p1_depth_iset_legal[0][0] / np.sum(p1_depth_iset_legal[0][0])
      
      # average_policies[0].append(np.array(policy))
      # iset_maps[0].append(p1_depth_iset_map[0][0])
      # average_policies[1].append(np.array(policy))
      # iset_maps[1].append(p2_depth_iset_map[0][1])
      
      extract_policy(p1_cfr, p1_depth_iset_map, 0, 0)
      extract_policy(p2_cfr, p2_depth_iset_map, 1, 0)
    else:
      extract_policy(p1_cfr, p1_depth_iset_map, 0, 1)
      extract_policy(p2_cfr, p2_depth_iset_map, 1, 1)
    
    # Prepare structures for next depth
    # Those are isets, iset_map, iset_ids, cf_values, reaches
    
    if current_depth == game.max_trajectory_length() - 1:
      break
    
    depth_for_next = 1 if current_depth == 0 else 2 
    
    p1_reaches = p1_cfr.find_reaches_from_average()[depth_for_next]
    p2_reaches = p2_cfr.find_reaches_from_average()[depth_for_next]
    
    
    p1_reaches = jnp.where(jax.nn.one_hot(1, 2)[..., None] < 0.5, p1_reaches, 1.0)
    p2_reaches = jnp.where(jax.nn.one_hot(0, 2)[..., None] < 0.5, p2_reaches, 1.0)
    
    # P1 CF values are cf values of player 2
    p1_cf_values = p1_cfr.cf_values[depth_for_next][1][p1_depth_history_iset[depth_for_next][1]]
    p2_cf_values = p2_cfr.cf_values[depth_for_next][0][p2_depth_history_iset[depth_for_next][0]]
    
    assert jnp.allclose(p1_depth_iset_map[depth_for_next][0], p2_depth_iset_map[depth_for_next][0])
    assert jnp.allclose(p1_depth_history_iset[depth_for_next][0], p2_depth_history_iset[depth_for_next][0]) 
    assert jnp.allclose(p1_depth_iset_map[depth_for_next][1], p2_depth_iset_map[depth_for_next][1])
    assert jnp.allclose(p1_depth_history_iset[depth_for_next][1], p2_depth_history_iset[depth_for_next][1]) 
    
    first_isets = depth_history_isets_full[depth_for_next]
    first_iset_map = [p1_depth_iset_map[depth_for_next][0], p2_depth_iset_map[depth_for_next][1]]
    first_iset_ids = p1_depth_history_iset[depth_for_next]
    
      
  policy = {}
  # def find_most_likely_index(iset, player, depth):
  #   closeness = np.linalg.norm(iset - self.depth_iset_map[depth][player], axis=-1)
  #   return np.argmin(closeness)
  
  
  def _traverse_for_policy(game_state, key, legals, depth):  
    _, p1_iset, p2_iset, ps = game.get_info(game_state)
    p1_iset = np.array(p1_iset)
    p2_iset = np.array(p2_iset)
    
    p1_iset_str = stringify(p1_iset)
    p2_iset_str = stringify(p2_iset)
    
    p1_abstracted, p2_abstracted = model.get_both_abstraction(ps, p1_iset, p2_iset)
    
    if p1_iset_str not in policy: 
    
      closeness = np.linalg.norm(p1_abstracted - iset_maps[0][depth], axis=-1)
      closest_id = np.argmin(closeness)
      
      avg_policy = average_policies[0][depth][closest_id] * legals[0]
      if np.sum(avg_policy) <  1e-10 or np.any(np.isnan(avg_policy)):
        p1_strategy = legals[0] / np.sum(legals[0])
      else:
        p1_strategy = avg_policy / np.sum(avg_policy) 
      
      policy[p1_iset_str] = p1_strategy 
    
    if p2_iset_str not in policy:
      closeness = np.linalg.norm(p2_abstracted - iset_maps[1][depth], axis=-1)
      closest_id = np.argmin(closeness)
      avg_policy = average_policies[1][depth][closest_id] * legals[1]
      if np.sum(avg_policy) <  1e-10 or np.any(np.isnan(avg_policy)):
        p2_strategy = legals[1] / np.sum(legals[1])
      else:
        p2_strategy = avg_policy / np.sum(avg_policy) 
    
      policy[p2_iset_str] = p2_strategy
    
  
    
    for a1i, a1 in enumerate(legals[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legals[1]):
        if a2 < 0.5:
          continue
        next_key, action_key = jax.random.split(key)
        new_game_state, terminal, rewards, new_legals = game.apply_action(game_state, action_key, depth, np.array([a1i, a2i]))
        if terminal:
          continue
        _traverse_for_policy(new_game_state, next_key, new_legals, depth + 1)
     
  _traverse_for_policy(game_state, state_key, legals, 0)
     
  return JaxPolicy(policy) 