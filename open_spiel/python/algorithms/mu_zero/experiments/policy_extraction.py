import numpy as np
import jax
import jax.numpy as jnp

import pyspiel

from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import prepare_cfr_structure, find_next_root
from open_spiel.python.algorithms.mu_zero.experiments.utils import stringify
from open_spiel.python.algorithms.mu_zero.mu_zero_train import MuZeroTrain
from open_spiel.python.algorithms.mu_zero.mu_zero_cfr import MuZeroCFR, MuZeroCFRConstants
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game import JaxGame, JaxPolicy
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game_algorithms import prepare_cfr_from_game, extract_policy_from_cfr, nash_equilibrium_cluster_game


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
  policy = JaxPolicy()
  
   
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
    init_cfr = prepare_cfr_structure(model, pl, args.depth_limit, init_iset, init_reaches, init_cf_values, False)
    _find_policy(init_cfr, [init_info], state_key, pl)
    # _find_policy(init_cfr, [init_info], pl, init_reaches, init_cf_values)
  return policy
        
    
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
  
  cfr.multiple_steps(1000)
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