
from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc import JaxLeduc, LeducGameState
from open_spiel.python.algorithms.mu_zero.experiments.utils import load_model
from open_spiel.python.algorithms.mu_zero.mu_zero_cfr import check_iset_similarity, MuZeroCFRConstants, MuZeroCFR
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game_algorithms import convert_depth_to_jax, convert_player_depth_to_jax, stringify, JaxPolicy
from open_spiel.python.algorithms.mu_zero.jax_games.muzero_leduc_init_cfr import MuZeroLeducInitConstants, MuZeroLeducInit
from open_spiel.python.algorithms.mu_zero.jax_games.muzero_leduc_gameplay import get_real_pure_mvs

from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc_algorithms import leduc_exploitability
import argparse
import jax
import jax.numpy as jnp
import numpy as np


parser = argparse.ArgumentParser()
# parser.add_argument("--leduc_depth", type=int, default=4)
parser.add_argument("--iterations", type=int, default=1000)

parser.add_argument("--experiment_type", type=str, default="rnad", choices=["no_abstraction", "no_dynamics", "with_dynamics", "rnad"])
 
parser.add_argument("--seed", type=int, default=1068, help="Random seed")
parser.add_argument("--model_range", type=int, nargs="+", default=[6, 7, 1], help="Model range") 


def create_iset_map(curr_iset, amount_actions, offset):
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
      
  isets = np.array(isets) + offset[:, None]
  actions = isets[..., None] * amount_actions + np.arange(amount_actions)[None, None, ...] 
  iset_map = [np.array(i) for i in iset_map]
  return iset_map, isets, actions
  
def prepare_init_leduc_cfr(model, experiment_name: str):
  game = JaxLeduc()
  init_states, init_legal_actions = game.generate_all_private_card_nodes()
  vectorized_get_info = jax.vmap(game.get_info, in_axes=(0,))
  vectorized_init_states = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *init_states)
  _, init_p1_isets, init_p2_isets, init_public_states = vectorized_get_info(vectorized_init_states)
  
  p1_abstractions, p2_abstractions = model.get_both_abstraction(init_public_states, init_p1_isets, init_p2_isets)
  p1_abstractions = np.array(p1_abstractions)
  p2_abstractions = np.array(p2_abstractions)
  
  init_isets = np.stack([p1_abstractions, p2_abstractions], axis=0)
  init_states = 30
  # The probability of each pair of hands is 1/30, so I assume it would be 1/sqrt(30) for both players
  
  
  depth_actions = []  # Is just a list of integers
  
  depth_iset_map = [] # ID -> Abstract iset
  depth_iset_legal = []

  depth_history_action_utility = [] # Float[D, H(D), A1, A2]
  depth_history_iset = [] # Int[D, Pl, H(D)]
  depth_history_actions = [] # Int[D, Pl, H(D), A] Just indices
  depth_history_legal = [] # Bool[D, Pl, H(D), A] or [D, H(D), A1, A2]

  depth_history_next_history = [] # Int[D, H(D), A1, A2]
  
  # Maybe one of those is not needed
  final_chance_probs = [] # Float[H, A]
  final_chance_continuations = [] # Float[H, A]
  
  mvs_iset_map = [[], []] # ID -> Abstract iset
  
  mvs_history_action_utility = [] # Float[H, A]
  mvs_history_iset = [[], []] # Int[H, Pl]
  mvs_history_actions = [[], []] # Int[H, Pl, A]
  
  mvs_states = []
  
  
  
  vectorized_apply_action = jax.vmap(game.apply_action, in_axes=(0, None, None, None), out_axes=(0, 0, 0, 0))
  # vectorized_abstraction = jax.vmap(jax.vmap(model.get_next_state_from_abstraction, in_axes=(None, None, -1, -1), out_axes=(-2, -2, -2, -2)), in_axes=(None, None, -1, -1), out_axes=(-2, -2, -2, -2))
  
  vectorized_abstraction = jax.vmap(model.get_next_state_from_abstraction, in_axes=(0, 0, None, None), out_axes=(0, 0, 0, 0))
   
  leduc_actions = 4
  mvs_depths = []
  def _traverse_leduc(game_state, abs_isets, legal_actions, key, depth = 0):
    public_card = np.array(game_state.public_card)
    
    if experiment_name == "no_abstraction":
      _, real_p1_isets, real_p2_isets, real_public_states = vectorized_get_info(game_state)
      curr_isets = np.stack([real_p1_isets, real_p2_isets], axis=0)
    elif experiment_name == "no_dynamics":
      _, real_p1_isets, real_p2_isets, real_public_states = vectorized_get_info(game_state)
      abs_p1_isets, abs_p2_isets = model.get_both_abstraction(real_public_states, real_p1_isets, real_p2_isets)
      curr_isets = np.stack([abs_p1_isets, abs_p2_isets], axis=0)
    elif experiment_name == "with_dynamics":
      curr_isets = abs_isets
    if public_card[0] != 0:
      # return
      # Finish the tree with padding
      for pad_depth in range(depth, 4):
        if len(depth_iset_map) <= pad_depth:
          depth_iset_map.append([[], []])
          depth_iset_legal.append([[], []])
          depth_history_action_utility.append([])
          depth_history_iset.append([[], []])
          depth_history_actions.append([[], []])
          depth_history_legal.append([])
          depth_history_next_history.append([])
          
        pad_iset = np.zeros((1, curr_isets.shape[-1]))
        pad_legal = np.zeros((1, 4))
        pad_legal[..., 0] = 1
        
        pad_iset_id = np.zeros(2, dtype=np.int32)
        for pl in range(2):
          for ni in depth_iset_map[pad_depth][pl]:
            pad_iset_id[pl] += ni.shape[0]
        
        depth_iset_map[pad_depth][0].append(pad_iset)
        depth_iset_map[pad_depth][1].append(pad_iset)
        
        depth_iset_legal[pad_depth][0].append(pad_legal)
        depth_iset_legal[pad_depth][1].append(pad_legal)
        
        legal_actions_both = np.zeros((curr_isets.shape[1], leduc_actions, leduc_actions))
        legal_actions_both[..., 0, 0] = 1
        
        p1_actions = np.arange(leduc_actions) + pad_iset_id[0] * leduc_actions
        p2_actions = np.arange(leduc_actions) + pad_iset_id[1] * leduc_actions
        depth_history_iset[pad_depth][0].append(np.full((curr_isets.shape[1], ), pad_iset_id[0]))
        depth_history_iset[pad_depth][1].append(np.full((curr_isets.shape[1], ), pad_iset_id[1]))
        depth_history_actions[pad_depth][0].append(np.repeat(p1_actions[None, ...], curr_isets.shape[1], axis=0))
        depth_history_actions[pad_depth][1].append(np.repeat(p2_actions[None, ...], curr_isets.shape[1], axis=0))
        
        depth_history_legal[pad_depth].append(legal_actions_both)
        depth_history_action_utility[pad_depth].append(np.zeros((curr_isets.shape[1], leduc_actions, leduc_actions)))
        
        next_history = np.full((curr_isets.shape[1], leduc_actions, leduc_actions), -1)
        next_history_id = 0
        if len(depth_history_iset) > pad_depth + 1:
          for ni in depth_history_iset[pad_depth + 1][0]:
            next_history_id += ni.shape[0]
        if pad_depth == 3:
          for mvs in mvs_states:
            next_history_id += mvs.terminal.shape[0]
        next_history[..., 0, 0] = np.arange(curr_isets.shape[1]) + next_history_id
        depth_history_next_history[pad_depth].append(next_history)
        
      mvs_states.append(game_state)
      mvs_depths.append(np.full((game_state.terminal.shape[0], ), depth))
        # mvs_depths.append(pad_depth)
        
        
      pad_legal = np.zeros(())
      return
    
    if len(depth_iset_map) <= depth:
      depth_iset_map.append([[], []])
      depth_iset_legal.append([[], []])
      depth_history_iset.append([[], []])
      depth_history_actions.append([[], []])
      depth_history_action_utility.append([])
      depth_history_legal.append([])
      depth_history_next_history.append([])
      
    offset = np.zeros((2, ), dtype=np.int32)
    for pl in range(2):
      for ni in depth_iset_map[depth][pl]:
        offset[pl] += ni.shape[0]
    iset_map, isets, actions = create_iset_map(curr_isets, leduc_actions, offset)
    
    legality_threshold = 0.0
    if experiment_name == "no_abstraction":
      p1_legal_iset = np.repeat(legal_actions[0][None, ...], iset_map[0].shape[0], axis=0)
      p2_legal_iset = np.repeat(legal_actions[1][None, ...], iset_map[1].shape[0], axis=0)
    else:
      p1_legal_iset, p2_legal_iset = model.get_both_legal_actions_from_abstraction(iset_map[0], iset_map[1])
      p1_legal_iset, p2_legal_iset = p1_legal_iset > legality_threshold, p2_legal_iset > legality_threshold
    assert (p1_legal_iset.shape[0] - 1) <= np.max(isets[0] - offset[0])
    assert (p2_legal_iset.shape[0] - 1) <= np.max(isets[1] - offset[1])
    p1_legal, p2_legal = p1_legal_iset[isets[0] - offset[0]], p2_legal_iset[isets[1] - offset[1]]
    # iset_legal = [p1_legal_iset, p2_legal_iset]
    legal = p1_legal[..., None] * p2_legal[..., None, :]
    
    action_utility = np.zeros((curr_isets.shape[1], leduc_actions, leduc_actions))
    next_history = np.full((curr_isets.shape[1], leduc_actions, leduc_actions), -1)
    # Thi
    depth_iset_map[depth][0].append(iset_map[0])
    depth_iset_map[depth][1].append(iset_map[1])
    depth_iset_legal[depth][0].append(p1_legal_iset)
    depth_iset_legal[depth][1].append(p2_legal_iset)
    
    depth_history_iset[depth][0].append(isets[0])
    depth_history_iset[depth][1].append(isets[1])
    depth_history_actions[depth][0].append(actions[0])
    depth_history_actions[depth][1].append(actions[1])
    depth_history_legal[depth].append(legal)
    depth_history_action_utility[depth].append(action_utility)
    depth_history_next_history[depth].append(next_history)
    
    for a1i, a1 in enumerate(legal_actions[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legal_actions[1]):
        if a2 < 0.5:
          continue
        
        next_state, next_terminal, next_rewards, next_legals = vectorized_apply_action(game_state, key, depth, np.array([a1i, a2i]))
        is_terminal = bool(next_terminal.any())
         
        
        next_p1_isets, next_p2_isets, next_utilities, next_abs_terminal = vectorized_abstraction(abs_isets[0], abs_isets[1], a1i, a2i)
        
        if experiment_name == "no_abstraction" or experiment_name == "no_dynamics":
          action_utility[..., a1i, a2i] = np.array(next_rewards)
        else:
          action_utility[..., a1i, a2i] = np.array(next_utilities[:, 0] - next_utilities[:, 1]) / 2
          # next_utilities = np.array(next_utilities)
          # action_utility[..., a1i, a2i] = np.sum(next_utilities, axis=-1) / 2
          
        
        if is_terminal:
          assert bool(next_terminal.all())
          continue
        next_offset = 0
        if len(depth_history_iset) > depth + 1:
          for ni in depth_history_iset[depth + 1][0]:
            next_offset += ni.shape[0]
        if depth == 3:
          for mvs in mvs_states:
            next_offset += mvs.terminal.shape[0]
          
        next_history[..., a1i, a2i] = np.arange(curr_isets.shape[1]) + next_offset
        
        next_isets = np.stack([next_p1_isets, next_p2_isets], axis=0)
        
        _traverse_leduc(next_state, next_isets, next_legals[0], key, depth + 1)
    

  key = jax.random.PRNGKey(0)
  _traverse_leduc(vectorized_init_states, init_isets, init_legal_actions, key, 0)
  
  depth_iset_map = [[np.concatenate(pl, axis=0) for pl in d] for d in depth_iset_map]
  depth_iset_legal = [[np.concatenate(pl, axis=0) for pl in d] for d in depth_iset_legal]
  depth_history_iset = [[np.concatenate(pl, axis=0) for pl in d] for d in depth_history_iset]
  depth_history_actions = [[np.concatenate(pl, axis=0) for pl in d] for d in depth_history_actions]
  depth_history_legal = [np.concatenate(d, axis=0) for d in depth_history_legal]
  depth_history_action_utility = [np.concatenate(d, axis=0) for d in depth_history_action_utility]
  depth_history_next_history = [np.concatenate(d, axis=0) for d in depth_history_next_history]
  
  depth_actions = [4] * 4
  depth_iset_map = convert_player_depth_to_jax(depth_iset_map)
  depth_iset_legal = convert_player_depth_to_jax(depth_iset_legal)
  depth_history_action_utility = convert_depth_to_jax(depth_history_action_utility)
  depth_history_iset = convert_depth_to_jax(depth_history_iset)
  depth_history_actions = convert_depth_to_jax(depth_history_actions)
  depth_history_legal = convert_depth_to_jax(depth_history_legal)
  depth_history_next_history = convert_depth_to_jax(depth_history_next_history)
  
  mvs_states = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *mvs_states)
  
  # This sucks hard.
  list_mvs_states = [LeducGameState(public_card = mvs_states.public_card[i],
                                    private_cards = mvs_states.private_cards[i],
                                    action_history = mvs_states.action_history[i],
                                    current_chips = mvs_states.current_chips[i],
                                    turns_this_round = mvs_states.turns_this_round[i],
                                    terminal = mvs_states.terminal[i]) for i in range(mvs_states.terminal.shape[0])]
  new_mvs_states = []
  for mvs in list_mvs_states: 
    new_mvs_states.extend(game.generate_all_public_card_nodes(mvs))
  new_mvs_states = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *new_mvs_states)
  final_chance_probs = jnp.full((mvs_states.terminal.shape[0], 4), 1/4)
  final_chance_continuations = jnp.arange(new_mvs_states.terminal.shape[0]).reshape((-1, 4))
  _, after_chance_p1_isets, after_chance_p2_isets, after_chance_public_states = vectorized_get_info(new_mvs_states)
  
  after_chance_isets = np.stack([after_chance_p1_isets, after_chance_p2_isets], axis=0)
  after_chance_p1_abstraction, after_chance_p2_abstraction = model.get_both_abstraction(after_chance_public_states, after_chance_p1_isets, after_chance_p2_isets)
  
  
  after_chance_abstraction = np.stack([after_chance_p1_abstraction, after_chance_p2_abstraction], axis=0)
  
  if experiment_name == "no_abstraction":
    mvs_actions = 12
    mvs_vmap = jax.vmap(get_real_pure_mvs, in_axes=(0,), out_axes=0)
    mvs_history_action_utility = mvs_vmap(new_mvs_states)
    mvs_iset_map, mvs_history_iset, mvs_history_actions = create_iset_map(after_chance_isets, mvs_actions, np.zeros((2, ), dtype=np.int32))
  else:
    mvs_actions = model.config.transformations + 1
    mvs_history_action_utility = model.get_mvs_from_abstraction(after_chance_p1_abstraction, after_chance_p2_abstraction)
    mvs_iset_map, mvs_history_iset, mvs_history_actions = create_iset_map(after_chance_abstraction, mvs_actions, np.zeros((2, ), dtype=np.int32))
  
  
  init_reaches = np.ones((2, init_states)) / np.sqrt(init_states)
  
  constants = MuZeroLeducInitConstants(
    resolving_player = 0,

    init_reaches = init_reaches,

    depth_actions = depth_actions,
    
    depth_iset_map = depth_iset_map,
    depth_iset_legal = depth_iset_legal,

    depth_history_action_utility = depth_history_action_utility,
    depth_history_iset = depth_history_iset,
    depth_history_actions = depth_history_actions,
    depth_history_legal = depth_history_legal,

    depth_history_next_history = depth_history_next_history,
    
    # Maybe one of those is not needed
    final_chance_probs = final_chance_probs,
    final_chance_continuations = final_chance_continuations,
    
    mvs_iset_map = mvs_iset_map,
    
    mvs_history_action_utility = mvs_history_action_utility,
    mvs_history_iset = mvs_history_iset,
    mvs_history_actions = mvs_history_actions,
  )

  
  cfr = MuZeroLeducInit(constants)
  mvs_depths = np.concatenate(mvs_depths, axis=0)
  mvs_depths = np.repeat(mvs_depths[..., None], 4, axis=-1).flatten()
  
  return cfr, new_mvs_states, mvs_depths

 
def prepare_cont_leduc_cfr(model, cfr: MuZeroLeducInit, after_chance_states: LeducGameState, after_chance_depths: jnp.ndarray, experiment_name: str):
  
  game = JaxLeduc()
  vectorized_get_info = jax.vmap(game.get_info, in_axes=(0,))
  
  _, init_p1_isets, init_p2_isets, init_public_states = vectorized_get_info(after_chance_states)
  
  _, init_reaches = cfr.find_reaches_from_average()
  cf_values = cfr.mvs_cf_values
  
  depth_iset_map = []
  depth_iset_legal = []
  
  depth_history_action_utility = []
  depth_history_iset = []
  depth_history_actions = []
  depth_history_legal = []
  depth_history_next_history = []
  
  leduc_actions = 4
  
  vectorized_apply_action = jax.vmap(game.apply_action, in_axes=(0, None, 0, None), out_axes=(0, 0, 0, 0))
  vectorized_abstraction = jax.vmap(model.get_next_state_from_abstraction, in_axes=(0, 0, None, None), out_axes=(0, 0, 0, 0))
  # tree_depth is now list
  def _traverse_leduc(game_state, abs_isets, legal_actions, key, tree_depth: jnp.ndarray, structure_depth=0):
    # if depth == 4:
    #   assert False, "This should not happen"
    if experiment_name == "no_abstraction":
      _, real_p1_isets, real_p2_isets, real_public_states = vectorized_get_info(game_state)
      curr_isets = np.stack([real_p1_isets, real_p2_isets], axis=0)
    elif experiment_name == "no_dynamics":
      _, real_p1_isets, real_p2_isets, real_public_states = vectorized_get_info(game_state)
      abs_p1_isets, abs_p2_isets = model.get_both_abstraction(real_public_states, real_p1_isets, real_p2_isets)
      curr_isets = np.stack([abs_p1_isets, abs_p2_isets], axis=0)
    elif experiment_name == "with_dynamics":
      curr_isets = abs_isets
    
    if len(depth_iset_map) <= structure_depth:
      depth_iset_map.append([[], []])
      depth_iset_legal.append([[], []])
      depth_history_action_utility.append([])
      depth_history_iset.append([[], []])
      depth_history_actions.append([[], []])
      depth_history_legal.append([])
      depth_history_next_history.append([])
      
      
    offset = np.zeros((2, ), dtype=np.int32)
    for pl in range(2):
      for ni in depth_iset_map[structure_depth][pl]:
        offset[pl] += ni.shape[0]
    iset_map, isets, actions = create_iset_map(curr_isets, leduc_actions, offset)
    
    
    legality_threshold = 0.0
    
    if experiment_name == "no_abstraction":
      p1_legal_iset = np.repeat(legal_actions[0][None, ...], iset_map[0].shape[0], axis=0)
      p2_legal_iset = np.repeat(legal_actions[1][None, ...], iset_map[1].shape[0], axis=0)
    else:
      p1_legal_iset, p2_legal_iset = model.get_both_legal_actions_from_abstraction(iset_map[0], iset_map[1])
      p1_legal_iset, p2_legal_iset = p1_legal_iset > legality_threshold, p2_legal_iset > legality_threshold
      
      
    assert (p1_legal_iset.shape[0] - 1) <= np.max(isets[0] - offset[0])
    assert (p2_legal_iset.shape[0] - 1) <= np.max(isets[1] - offset[1])
    p1_legal, p2_legal = p1_legal_iset[isets[0] - offset[0]], p2_legal_iset[isets[1] - offset[1]]
    # iset_legal = [p1_legal_iset, p2_legal_iset]
    legal = p1_legal[..., None] * p2_legal[..., None, :]
    
    action_utility = np.zeros((curr_isets.shape[1], leduc_actions, leduc_actions))
    next_history = np.full((curr_isets.shape[1], leduc_actions, leduc_actions), -1)
    # Thi
    depth_iset_map[structure_depth][0].append(iset_map[0])
    depth_iset_map[structure_depth][1].append(iset_map[1])
    depth_iset_legal[structure_depth][0].append(p1_legal_iset)
    depth_iset_legal[structure_depth][1].append(p2_legal_iset)
    
    depth_history_iset[structure_depth][0].append(isets[0])
    depth_history_iset[structure_depth][1].append(isets[1])
    depth_history_actions[structure_depth][0].append(actions[0])
    depth_history_actions[structure_depth][1].append(actions[1])
    depth_history_legal[structure_depth].append(legal)
    depth_history_action_utility[structure_depth].append(action_utility)
    depth_history_next_history[structure_depth].append(next_history)
    
    
    for a1i, a1 in enumerate(legal_actions[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legal_actions[1]):
        if a2 < 0.5:
          continue
        
        next_state, next_terminal, next_rewards, next_legals = vectorized_apply_action(game_state, key, tree_depth, np.array([a1i, a2i]))
        action_utility[..., a1i, a2i] = np.array(next_rewards)
        is_terminal = bool(next_terminal.any())
         
        
        next_p1_isets, next_p2_isets, next_utilities, next_abs_terminal = vectorized_abstraction(abs_isets[0], abs_isets[1], a1i, a2i)
        
        if experiment_name == "no_abstraction" or experiment_name == "no_dynamics":
          action_utility[..., a1i, a2i] = np.array(next_rewards)
        else:
          action_utility[..., a1i, a2i] = np.array(next_utilities[:, 0] - next_utilities[:, 1]) / 2
          # next_utilities = np.array(next_utilities)
          # action_utility[..., a1i, a2i] = np.sum(next_utilities, axis=-1) / 2
          
        if is_terminal:
          assert bool(next_terminal.all())
          continue
        next_offset = 0
        if len(depth_history_iset) > structure_depth + 1:
          for ni in depth_history_iset[structure_depth + 1][0]:
            next_offset += ni.shape[0]
            
          
        next_history[..., a1i, a2i] = np.arange(curr_isets.shape[1]) + next_offset
        
        next_isets = np.stack([next_p1_isets, next_p2_isets], axis=0)
        
        _traverse_leduc(next_state, next_isets, next_legals[0], key, tree_depth + 1, structure_depth + 1)
  
  
  _, legal_actions = game.initialize_structures(jax.random.PRNGKey(0))
  after_chance_p1_abstraction, after_chance_p2_abstraction = model.get_both_abstraction(init_public_states, init_p1_isets, init_p2_isets)
  
  after_chance_abstraction = np.stack([after_chance_p1_abstraction, after_chance_p2_abstraction], axis=0)
  _traverse_leduc(after_chance_states, after_chance_abstraction, legal_actions, jax.random.PRNGKey(0), after_chance_depths, 0)
  
  
  depth_iset_map = [[np.concatenate(pl, axis=0) for pl in d] for d in depth_iset_map]
  depth_iset_legal = [[np.concatenate(pl, axis=0) for pl in d] for d in depth_iset_legal]
  depth_history_iset = [[np.concatenate(pl, axis=0) for pl in d] for d in depth_history_iset]
  depth_history_actions = [[np.concatenate(pl, axis=0) for pl in d] for d in depth_history_actions]
  depth_history_legal = [np.concatenate(d, axis=0) for d in depth_history_legal]
  depth_history_action_utility = [np.concatenate(d, axis=0) for d in depth_history_action_utility]
  depth_history_next_history = [np.concatenate(d, axis=0) for d in depth_history_next_history]
  
  
  def prepare_gadget_layer(game_state, abs_isets):
    
    if experiment_name == "no_abstraction":
      _, real_p1_isets, real_p2_isets, real_public_states = vectorized_get_info(game_state)
      curr_isets = np.stack([real_p1_isets, real_p2_isets], axis=0)
    elif experiment_name == "no_dynamics":
      _, real_p1_isets, real_p2_isets, real_public_states = vectorized_get_info(game_state)
      abs_p1_isets, abs_p2_isets = model.get_both_abstraction(real_public_states, real_p1_isets, real_p2_isets)
      curr_isets = np.stack([abs_p1_isets, abs_p2_isets], axis=0)
    elif experiment_name == "with_dynamics":
      curr_isets = abs_isets
    
    
     
    iset_map, isets, actions = create_iset_map(curr_isets, 2, np.zeros((2, ), dtype=np.int32))
    
    p1_depth_iset_map = iset_map
    p2_depth_iset_map = iset_map
    
    p1_depth_iset_legal = [np.zeros((iset_map[pl].shape[0], 2), dtype=np.float32) for pl in range(2)]
    p2_depth_iset_legal = [np.zeros((iset_map[pl].shape[0], 2), dtype=np.float32) for pl in range(2)]
    
    # Gadget is 2nd player
    p1_depth_iset_legal[0][:, 0] = 1
    p1_depth_iset_legal[1][:] = 1
    
    p2_depth_iset_legal[0][:] = 1
    p2_depth_iset_legal[1][:, 0] = 1
    
    
    p1_depth_history_iset = isets
    p2_depth_history_iset = isets
    
    p1_depth_history_actions = actions
    p2_depth_history_actions = actions
    
    g1_legal_gadget = np.array([[1, 1], [0, 0]])
    g2_legal_gadget = np.array([[1, 0], [1, 0]])
    
    p1_depth_history_legal = np.repeat(g1_legal_gadget[None, ...], curr_isets.shape[1], axis=0)
    p2_depth_history_legal = np.repeat(g2_legal_gadget[None, ...], curr_isets.shape[1], axis=0)
    
    p1_depth_history_action_utility = np.zeros((curr_isets.shape[1], 2, 2))
    p2_depth_history_action_utility = np.zeros((curr_isets.shape[1], 2, 2))
    
      
    p1_cf_values = cfr.mvs_cf_values[1][cfr.constants.mvs_history_iset[1]]
    p2_cf_values = cfr.mvs_cf_values[0][cfr.constants.mvs_history_iset[0]]
    
    p1_cf_values = np.array(p1_cf_values)
    p2_cf_values = np.array(p2_cf_values)
    
    assert p1_cf_values.shape[0] == p1_depth_history_action_utility.shape[0]
    assert p2_cf_values.shape[0] == p2_depth_history_action_utility.shape[0]
    
    p1_depth_history_action_utility[..., 0, 0] = p1_cf_values
    p2_depth_history_action_utility[..., 0, 0] = p2_cf_values
    
    
    p1_depth_history_next_history = np.full((curr_isets.shape[1], 2, 2), -1)
    p2_depth_history_next_history = np.full((curr_isets.shape[1], 2, 2), -1)
    
    p1_depth_history_next_history[..., 0, 1] = np.arange(curr_isets.shape[1])
    p2_depth_history_next_history[..., 1, 0] = np.arange(curr_isets.shape[1])
    
    return [p1_depth_iset_map], [p2_depth_iset_map], [p1_depth_iset_legal], [p2_depth_iset_legal], [p1_depth_history_iset], [p2_depth_history_iset], [p1_depth_history_actions], [p2_depth_history_actions], [p1_depth_history_legal], [p2_depth_history_legal], [p1_depth_history_action_utility], [p2_depth_history_action_utility], [p1_depth_history_next_history], [p2_depth_history_next_history]
    
    
  p1_depth_iset_map, p2_depth_iset_map, p1_depth_iset_legal, p2_depth_iset_legal, p1_depth_history_iset, p2_depth_history_iset, p1_depth_history_actions, p2_depth_history_actions, p1_depth_history_legal, p2_depth_history_legal, p1_depth_history_action_utility, p2_depth_history_action_utility, p1_depth_history_next_history, p2_depth_history_next_history = prepare_gadget_layer(after_chance_states, after_chance_abstraction)
    
    
  p1_init_reaches = np.array(init_reaches)
  p2_init_reaches = np.array(init_reaches)
  p1_init_reaches[1] = 1.0
  p2_init_reaches[0] = 1.0
  
  p1_init_reaches = jnp.array(p1_init_reaches)
  p2_init_reaches = jnp.array(p2_init_reaches)
  
  
  
  p1_depth_iset_map.extend(depth_iset_map)
  p1_depth_iset_legal.extend(depth_iset_legal)
  p2_depth_iset_map.extend(depth_iset_map)
  p2_depth_iset_legal.extend(depth_iset_legal)
  
  p1_depth_history_action_utility.extend(depth_history_action_utility)
  p1_depth_history_iset.extend(depth_history_iset)
  p1_depth_history_actions.extend(depth_history_actions)
  p1_depth_history_legal.extend(depth_history_legal)
  p1_depth_history_next_history.extend(depth_history_next_history)
  
  p2_depth_history_action_utility.extend(depth_history_action_utility)
  p2_depth_history_iset.extend(depth_history_iset)
  p2_depth_history_actions.extend(depth_history_actions)
  p2_depth_history_legal.extend(depth_history_legal)
  p2_depth_history_next_history.extend(depth_history_next_history)
  
  p1_depth_iset_map = convert_player_depth_to_jax(p1_depth_iset_map)
  p1_depth_iset_legal = convert_player_depth_to_jax(p1_depth_iset_legal)
  p2_depth_iset_map = convert_player_depth_to_jax(p2_depth_iset_map)
  p2_depth_iset_legal = convert_player_depth_to_jax(p2_depth_iset_legal)
  
  
  
  p1_depth_history_action_utility = convert_depth_to_jax(p1_depth_history_action_utility)
  p1_depth_history_iset = convert_depth_to_jax(p1_depth_history_iset)
  p1_depth_history_actions = convert_depth_to_jax(p1_depth_history_actions)
  p1_depth_history_legal = convert_depth_to_jax(p1_depth_history_legal)
  p1_depth_history_next_history = convert_depth_to_jax(p1_depth_history_next_history)
  
  p2_depth_history_action_utility = convert_depth_to_jax(p2_depth_history_action_utility)
  p2_depth_history_iset = convert_depth_to_jax(p2_depth_history_iset)
  p2_depth_history_actions = convert_depth_to_jax(p2_depth_history_actions)
  p2_depth_history_legal = convert_depth_to_jax(p2_depth_history_legal)
  p2_depth_history_next_history = convert_depth_to_jax(p2_depth_history_next_history)
  
  
  depth_actions = [2, 4, 4, 4, 4]
  p1_constants = MuZeroCFRConstants(
    resolving_player = 0,
    init_reaches = p1_init_reaches,
    depth_actions = depth_actions,
    
    
    depth_iset_legal = p1_depth_iset_legal,
    
    depth_history_action_utility = p1_depth_history_action_utility,
    depth_history_iset = p1_depth_history_iset,
    depth_history_actions = p1_depth_history_actions,
    depth_history_legal = p1_depth_history_legal,
    depth_history_next_history = p1_depth_history_next_history,
  )
  
  p1_cfr = MuZeroCFR(p1_constants, p1_depth_iset_map)
  
  p2_constants = MuZeroCFRConstants(
    resolving_player = 1,
    init_reaches = p2_init_reaches,
    depth_actions = depth_actions,
    
    depth_iset_legal = p2_depth_iset_legal,
    
    depth_history_action_utility = p2_depth_history_action_utility,
    depth_history_iset = p2_depth_history_iset,
    depth_history_actions = p2_depth_history_actions,
    depth_history_legal = p2_depth_history_legal,
    depth_history_next_history = p2_depth_history_next_history,
  )
  
  p2_cfr = MuZeroCFR(p2_constants, p2_depth_iset_map)
  
  
  return p1_cfr, p2_cfr
          
          
def export_policy_from_cfr(model, init_cfr, p1_cfr, p2_cfr, experiment_name: str):
  game = JaxLeduc()
  
  strategy  = {}
  init_states, init_legal_actions = game.generate_all_private_card_nodes()

  vectorized_get_info = jax.vmap(game.get_info, in_axes=(0,))
  vectorized_apply_action = jax.vmap(game.apply_action, in_axes=(0, None, None, None), out_axes=(0, 0, 0, 0))
  vectorized_abstraction = jax.vmap(model.get_next_state_from_abstraction, in_axes=(0, 0, None, None), out_axes=(0, 0, 0, 0))

  def _traverse_leduc(game_state, abs_isets, legal_actions, key, tree_depth=0, structure_depth=0):

    public_card = np.array(game_state.public_card)

    _, p1_iset, p2_iset, public_state = vectorized_get_info(game_state)
    np_p1_iset = np.array(p1_iset)
    np_p2_iset = np.array(p2_iset)
    p1_iset_str = stringify(np_p1_iset)
    p2_iset_str = stringify(np_p2_iset)
    if experiment_name == "no_abstraction":
      curr_p1_iset, curr_p2_iset = p1_iset,p2_iset
    elif experiment_name == "no_dynamics":
      abs_p1_iset, abs_p2_iset = model.get_both_abstraction(public_state, p1_iset, p2_iset)
      curr_p1_iset, curr_p2_iset = abs_p1_iset, abs_p2_iset
    elif experiment_name == "with_dynamics":
      curr_p1_iset, curr_p2_iset = abs_isets[0], abs_isets[1]
      
    
    
    if public_card[0] == 0:
      for i in range(curr_p1_iset.shape[0]):
        p1_iset_str = stringify(np_p1_iset[i])
        strategy[p1_iset_str] = init_cfr.get_strategy(curr_p1_iset[i], 0, structure_depth)
      for j in range(curr_p2_iset.shape[0]):
        p2_iset_str = stringify(np_p2_iset[j])
        strategy[p2_iset_str] = init_cfr.get_strategy(curr_p2_iset[j], 1, structure_depth)
    else:
      for i in range(curr_p1_iset.shape[0]):
        p1_iset_str = stringify(np_p1_iset[i])
        strategy[p1_iset_str] = p1_cfr.get_strategy(curr_p1_iset[i], 0, structure_depth)
      for j in range(curr_p2_iset.shape[0]):
        p2_iset_str = stringify(np_p2_iset[j])
        strategy[p2_iset_str] = p2_cfr.get_strategy(curr_p2_iset[j], 1, structure_depth)
    


    for a1i, a1 in enumerate(legal_actions[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legal_actions[1]):
        if a2 < 0.5:
          continue
        
        next_state, next_terminal, next_rewards, next_legals = vectorized_apply_action(game_state, key, tree_depth, np.array([a1i, a2i])) 
         
        
        next_p1_isets, next_p2_isets, next_utilities, next_abs_terminal = vectorized_abstraction(abs_isets[0], abs_isets[1], a1i, a2i)
        
        is_terminal = bool(next_terminal.any())
        if is_terminal:
          assert bool(next_terminal.all())
          continue
          
        
        
        next_public_card = np.array(next_state.public_card)
        if public_card[0] == 0 and next_public_card[0] != 0:
          new_game_states = []
          list_next_states = [LeducGameState(public_card = next_state.public_card[i],
                                      private_cards = next_state.private_cards[i],
                                      action_history = next_state.action_history[i],
                                      current_chips = next_state.current_chips[i],
                                      turns_this_round = next_state.turns_this_round[i],
                                      terminal = next_state.terminal[i]) for i in range(next_state.terminal.shape[0])]
          for next_state in list_next_states:
            new_game_states.extend(game.generate_all_public_card_nodes(next_state))
          new_game_states = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *new_game_states)
          
          _, next_p1_real_iset, next_p2_real_iset, next_public_states = vectorized_get_info(new_game_states)
          next_p1_isets, next_p2_isets = model.get_both_abstraction(next_public_states, next_p1_real_iset, next_p2_real_iset)
          next_isets = np.stack([next_p1_isets, next_p2_isets], axis=0)
          _traverse_leduc(new_game_states, next_isets, next_legals[0], key, tree_depth + 1, 1)
        else:
          next_isets = np.stack([next_p1_isets, next_p2_isets], axis=0)
          _traverse_leduc(next_state, next_isets, next_legals[0], key, tree_depth + 1, structure_depth + 1)
          
  init_states = jax.tree.map(lambda *x: np.stack(x, axis=0), *init_states)
  _, init_p1_isets, init_p2_isets, init_public_states = vectorized_get_info(init_states)
  abs_init_p1_isets, abs_init_p2_isets = model.get_both_abstraction(init_public_states, init_p1_isets, init_p2_isets)
  abs_init_isets = np.stack([abs_init_p1_isets, abs_init_p2_isets], axis=0)
  _traverse_leduc(init_states, abs_init_isets, init_legal_actions, jax.random.PRNGKey(0), 0, 0)
  return JaxPolicy(strategy)
    
  
def export_policy_from_cfrs(init_cfr, p1_cfr, p2_cfr):
  
  game = JaxLeduc()
  
  strategy  = {}
  
  
  def _traverse_leduc(game_state, legal_actions, key, game_depth=0, cfr_depth=0, turn=0):
    public_card = np.array(game_state.public_card) 
    
    _, p1_iset, p2_iset, _ = game.get_info(game_state)
    p1_iset = np.array(p1_iset)
    p2_iset = np.array(p2_iset)
    p1_iset_str = stringify(p1_iset)
    p2_iset_str = stringify(p2_iset)
    if turn == 0:
      strategy[p1_iset_str] = init_cfr.get_strategy(p1_iset, 0, cfr_depth)
      strategy[p2_iset_str] = init_cfr.get_strategy(p2_iset, 1, cfr_depth)
    else:
      strategy[p1_iset_str] = p1_cfr.get_strategy(p1_iset, 0, cfr_depth)
      strategy[p2_iset_str] = p2_cfr.get_strategy(p2_iset, 1, cfr_depth)
    
    for a1i, a1 in enumerate(legal_actions[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legal_actions[1]):
        if a2 < 0.5:
          continue
        
        next_key, action_key = jax.random.split(key) 
        
        new_game_state, new_terminal, new_rewards, new_legals = game.apply_action(
          game_state, action_key, game_depth, np.array([a1i, a2i]))
        
    
        new_rewards = float(new_rewards)
        new_terminal = bool(new_terminal)
 
        
        if new_terminal:
          continue
         
        new_public_card = np.array(new_game_state.public_card)
        if public_card[0] == 0 and new_public_card[0] != 0: 
          new_game_states = game.generate_all_public_card_nodes(new_game_state)
          for new_game_state in new_game_states:
            _traverse_leduc(new_game_state, new_legals, next_key,game_depth + 1, 1, 1)
        else:
          _traverse_leduc(new_game_state, new_legals, next_key, game_depth + 1, cfr_depth + 1, turn)
        
        
  key = jax.random.PRNGKey(0) 
  states, init_legal_actions = game.generate_all_private_card_nodes()
  for state in states:
    _traverse_leduc(state, init_legal_actions, key, game_depth=0, cfr_depth=0, turn=0)
        
     
  return JaxPolicy(strategy)

def get_rnad_strategy(model):
  game = JaxLeduc()
  strategy = {}
  init_states, init_legal_actions = game.generate_all_private_card_nodes()
  # TODO: We could batch this.  
  def _traverse_leduc(game_state, legal_actions, key, depth=0):
    
    public_card = np.array(game_state.public_card)
    
    _, p1_iset, p2_iset, _ = game.get_info(game_state)
    p1_iset = np.array(p1_iset)
    p2_iset = np.array(p2_iset)
    p1_iset_str = stringify(p1_iset)
    p2_iset_str = stringify(p2_iset)
    
    p1_policy, p2_policy, _, _ = model.get_policy_and_value_both(np.stack([p1_iset, p2_iset], axis=0), legal_actions)
    
    strategy[p1_iset_str] = p1_policy
    strategy[p2_iset_str] = p2_policy
    
    for a1i, a1 in enumerate(legal_actions[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legal_actions[1]):
        if a2 < 0.5:
          continue
        
        apply_key, next_key = jax.random.split(key)
        next_state, next_terminal, next_rewards, next_legals = game.apply_action(game_state, apply_key, depth, np.array([a1i, a2i]))
        
        next_terminal = bool(next_terminal)
        if next_terminal:
          continue
        
        next_public_card = np.array(next_state.public_card)
        
        if public_card[0] == 0 and next_public_card[0] != 0:
          new_game_states = game.generate_all_public_card_nodes(next_state)
          for new_game_state in new_game_states:
            _traverse_leduc(new_game_state, next_legals, next_key, depth + 1)
        else:
          _traverse_leduc(next_state, next_legals, next_key, depth + 1)
        
        
        _traverse_leduc(next_state, next_legals, next_key, depth + 1)
        
  
  for state in init_states:
    _traverse_leduc(state, init_legal_actions, jax.random.PRNGKey(0))
  return JaxPolicy(strategy)

def run_leduc_experiment():
  leduc_nash = -0.08553
  args = parser.parse_args()
  
  assert len(args.model_range) <= 3, "Range works for up to 3 arguments"
  assert args.model_range[0] < args.model_range[1], "Model range is invalid"
  
  
  default_model_path = f"muzero_networks/leduc/seed_{args.seed}/muzero_"
  
  for i in range(*args.model_range):
    print(i, flush=True)
    model_path = default_model_path + str(i) + ".pkl"
    model = load_model(model_path) 
    assert isinstance(model.game, JaxLeduc)
  
    
    if args.experiment_name == "rnad":
      strategy = get_rnad_strategy(model)
    else:
      
      cfr, after_chance_states, after_chance_depths = prepare_init_leduc_cfr(model, args.experiment_name)
      
      cfr.multiple_steps(args.iterations)
      print("Solved initial CFR")
      print(cfr.averages[0][0] / jnp.sum(cfr.averages[0][0], -1, keepdims=True))
      print(cfr.averages[1][1] / jnp.sum(cfr.averages[1][1], -1, keepdims=True))
      
      p1_cfr, p2_cfr = prepare_cont_leduc_cfr(model, cfr, after_chance_states, after_chance_depths, args.experiment_name)

      p1_cfr.multiple_steps(args.iterations)
      p2_cfr.multiple_steps(args.iterations)
      print("Solved continuation CFR") 
      strategy = export_policy_from_cfr(model, cfr, p1_cfr, p2_cfr, args.experiment_name)
      print("Exporting policy")  
      
    p1_br, p2_br = leduc_exploitability(strategy)
    
    # The p1_br is a value from the perspective of player 2, that is why we add the nash value instead of subtracting it
    p1_expl = p1_br + leduc_nash
    p2_expl = p2_br - leduc_nash  
    print("P1: ", p1_expl)
    print("P2: ", p2_expl)
    # print("Exploitability: ", (p1_expl + p2_expl) / 2)
  
if __name__ == "__main__":
  run_leduc_experiment()