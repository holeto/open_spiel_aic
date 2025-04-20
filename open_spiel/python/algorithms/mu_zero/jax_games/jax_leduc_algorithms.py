from open_spiel.python.algorithms.mu_zero.jax_games.jax_game import JaxGame, JaxPolicy
from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc import JaxLeduc, LeducGameState
#from open_spiel.python.algorithms.mu_zero.mu_zero_cfr import MuZeroCFR, MuZeroCFRConstants 
from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc_cfr import JaxLeducCFR
from open_spiel.python.algorithms.mu_zero.experiments.utils import stringify
from open_spiel.python.policy import TabularPolicy
#from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import convert_depth_to_jax, convert_player_depth_to_jax

from dataclasses import dataclass
from copy import deepcopy

import jax
import chex
import jax.numpy as jnp
import numpy as np
import pyspiel

def solve_full_game(resolve_iterations=3000):
  full_game_cfr = JaxLeducCFR()
  full_game_cfr.multiple_steps(resolve_iterations)
  found_pols = full_game_cfr.average_policy()
  return found_pols


def jax_policy_to_tabular(jax_policy: JaxPolicy):
  """ Traverses the tree of the pyspiel game and 
  extracts given JaxPolicy into TabularPolicy format
  (using iset strings in spiel game and also skipping the fake simultaneous moves)
  """
  game = pyspiel.load_game("leduc_poker")
  tabular_policy = TabularPolicy(game)
  dummy_key = jax.random.key(0)
  jax_game = JaxLeduc()
  def _init_chance_outcomes_spiel():
    root_states = []
    init_state = game.new_initial_state()
    for a1, p1 in init_state.chance_outcomes():
      new_state = init_state.clone()
      new_state.apply_action(a1)
      for a2, p2 in new_state.chance_outcomes():
        after_chance_state = new_state.clone()
        after_chance_state.apply_action(a2)
        root_states.append(after_chance_state)
    return root_states

  def _traverse_tree(state: pyspiel.State, jax_state: LeducGameState, depth = 0):
    if state.is_terminal():
      return
    cur_player = state.current_player()
    pl_iset_str = state.information_state_string(cur_player)
    jax_state_tensor, jax_p1_iset, jax_p2_iset, jax_ps = jax_game.get_info(jax_state)
    jax_iset_str = stringify(jax_p1_iset) if cur_player == 0 else stringify(jax_p2_iset)
    state_pols = tabular_policy.policy_for_key(pl_iset_str)
    jax_state_pols = jax_policy[jax_iset_str]
    #Do not forget to skip the invalid action
    for i, prob in enumerate(jax_state_pols[1:]):
      state_pols[i] = prob
    assert(np.abs(np.sum(state_pols) - 1) <= 1e-5)
    for a in state.legal_actions():
      new_state = state.clone()
      new_state.apply_action(a)
      jax_action = np.zeros(2)
      #Again, do not forget that jax game has also the invalid action
      jax_action[cur_player] = a + 1
      new_jax_state, terminal, reward, new_legals = jax_game.apply_action(jax_state, dummy_key, depth, jnp.array(jax_action))
      if new_state.is_chance_node():
        pc_chance_outcomes = []
        for a, p in new_state.chance_outcomes():
          after_pc_state = new_state.clone()
          after_pc_state.apply_action(a)
          pc_chance_outcomes.append(after_pc_state)
        jax_pc_chance_outcomes = jax_game.generate_all_public_card_nodes(new_jax_state)
        for pc_state, jax_pc_state in zip(pc_chance_outcomes, jax_pc_chance_outcomes):
          _traverse_tree(pc_state, jax_pc_state, depth + 1)
      else:
        _traverse_tree(new_state, new_jax_state, depth + 1)
  roots = _init_chance_outcomes_spiel()
  jax_roots, legals = jax_game.generate_all_private_card_nodes()
  for root_node, jax_root_node in zip(roots, jax_roots):
    _traverse_tree(root_node, jax_root_node, 0)
  return tabular_policy



def exploitability_jax_leduc(policy: JaxPolicy) -> tuple[JaxPolicy, JaxPolicy, float, float]:
  '''
    Computes a best response against both players using given policy.
    Return is a tuple of two JaxPolicy objects, one for each player and a value of this best response against the input policy.
    The other player is always the original policy.
    The output format: BR against policy of player 0, BR against policy of player 1, Value of BR against policy of player 0, Value of BR against policy of player 1.
  '''
  dummy_key = jax.random.key(0)
  game = JaxLeduc()
  
  init_state, init_legals = game.initialize_structures(dummy_key)
  chance_probabilities = np.zeros_like((init_legals.shape[1], init_legals.shape[1]))
  #The inner chance nodes here have only 4 chance outcomes
  chance_probabilities[0:4, 0] = 0.25
  #Dummy iset for chance nodes
  chance_iset = ''

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
  is_chance = [] #[D, H(D)]
   
  def _construct_tree(game_state:LeducGameState, legal_actions, reaches: tuple[float, float] = (1.0, 1.0), depth: int =0, is_chance_node: bool = False):
    
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
      is_chance.append([])
    state_tensor, p1_iset, p2_iset, ps = game.get_info(game_state)
    p1_iset = np.array(p1_iset)
    p2_iset = np.array(p2_iset)
    legal_actions = np.array(legal_actions) if not is_chance_node else np.zeros((2, 4))
    
    # ps_str = stringify(ps)
    p1_iset_str = stringify(p1_iset) if not is_chance_node else chance_iset
    p2_iset_str = stringify(p2_iset) if not is_chance_node else chance_iset
    
     
    if p1_iset_str not in iset_map[depth][0]:
      iset_map[depth][0].append(p1_iset_str)
      iset_legals[depth][0].append(legal_actions[0])
    if p2_iset_str not in iset_map[depth][1]:
      iset_map[depth][1].append(p2_iset_str)
      iset_legals[depth][1].append(legal_actions[1])
    
    isets[depth][0].append(iset_map[depth][0].index(p1_iset_str))
    isets[depth][1].append(iset_map[depth][1].index(p2_iset_str))
    #4 chance outcomes, Here we can do it this way, since Leduc
    # has 4 actions as well. Otherwise we would have to take maximum
    # of actions and chance outcomes
    b_pol1 = policy[p1_iset_str] if not is_chance_node else np.full_like(4, 0.25)
    b_pol2 = policy[p2_iset_str] if not is_chance_node else np.full_like(4, 0.25)
    assert abs(np.sum(b_pol1) - 1) < 1e-3
    assert abs(np.sum(b_pol2) - 1) < 1e-3
    behavior_policy[depth][0].append(b_pol1)
    behavior_policy[depth][1].append(b_pol2)
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
    is_chance[depth].append(is_chance_node)
    
  
    if p1_iset_str not in iset_action_value:
      iset_action_value[p1_iset_str] = np.zeros(len(legal_actions[0]))
      iset_action_legal[p1_iset_str] = np.array(legal_actions[0])
    if p2_iset_str not in iset_action_value:
      iset_action_value[p2_iset_str] = np.zeros(len(legal_actions[1]))
      iset_action_legal[p2_iset_str] = np.array(legal_actions[1])
    
    p1_state_value, p2_state_value = 0.0, 0.0
    if is_chance_node:
      #4 possible public cards
      ai = 0
      for pc in range(6):
        if pc == game_state.private_cards[0] or pc == game_state.private_cards[1]:
          continue
        new_game_state = LeducGameState(action_history = game_state.action_history,
                                        current_chips = game_state.current_chips,
                                        private_cards = game_state.private_cards,
                                        public_card = jnp.array([pc + 1]),
                                        turns_this_round = game_state.turns_this_round,
                                        terminal = game_state.terminal
                                        )
        next_history_id = 0 if len(legals) <= depth+1 else len(legals[depth + 1])
        state_continuation[ai, 0] = next_history_id
        ai += 1
        new_reaches = (reaches[0] * 0.25, reaches[1] * 0,25)
        _construct_tree(new_game_state, legal_actions, new_reaches, depth + 1)
      return

    for a1i, a1 in enumerate(legal_actions[0]):
      if a1 < 0.5:
        continue
      for a2i, a2 in enumerate(legal_actions[1]):
        if a2 < 0.5:
          continue
         
        
        new_game_state, new_terminal, new_rewards, new_legals = game.apply_action(
          game_state, dummy_key, depth, np.array([a1i, a2i]))
        
        state_reward[a1i, a2i] = new_rewards
         
        if new_terminal:
          continue
        
        state_terminals[a1i, a2i] = 1
        
        next_history_id = 0 if len(legals) <= depth+1 else len(legals[depth + 1])
        state_continuation[a1i, a2i] = next_history_id
        is_chance_node = game_state.public_card == 0 and new_game_state.public_card > 0
        
        new_reaches = (reaches[0] * b_pol1[a1i], reaches[1] * b_pol2[a2i])
        _construct_tree(new_game_state, new_legals, new_reaches, depth + 1, is_chance_node) 
   
  #Simulate the first chance node
  for c1 in range(6):
      for c2 in range(6):
        if c1 == c2:
          continue
        private_cards = jnp.array([c1, c2], dtype=int)
        subgame_state = LeducGameState(
                          action_history = init_state.action_history,
                          public_card = init_state.public_card,
                          private_cards = private_cards,
                          current_chips = init_state.current_chips,
                          turns_this_round = init_state.turns_this_round,
                          terminal = init_state.terminal
        )
        _construct_tree(subgame_state, init_legals, dummy_key, reaches={1/30, 1/30}) 
  
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
  is_chance = convert_to_numpy(is_chance)
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
      #Filter out the invalid isets
      if not iset:
        continue
      br_policy_p1[iset] = p1_br_policy[i]
    for i, iset in enumerate(iset_map[d][1]):
      #Filter out the invalid isets
      if not iset:
        continue
      br_policy_p2[iset] = p2_br_policy[i]
    
    p1_history_value = np.squeeze(np.take_along_axis(p1_action_value, p1_history_br[..., None], 1))
    p2_history_value = np.squeeze(np.take_along_axis(p2_action_value, p2_history_br[..., None], 1))
    breakpoint()
    #Make sure to NOT do the best response for chance nodes. The player cannot act there
    #p1_history_value = np.where() 
    
    state_value = np.stack((p1_history_value, p2_history_value), 0)  
    pass 
  return br_policy_p2, br_policy_p1, state_value[1], state_value[0]
    


