from open_spiel.python.algorithms.mu_zero.jax_games.jax_game import JaxGame, JaxPolicy
from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc import JaxLeduc, LeducGameState
#from open_spiel.python.algorithms.mu_zero.mu_zero_cfr import MuZeroCFR, MuZeroCFRConstants 
from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc_cfr import JaxLeducCFR
from open_spiel.python.algorithms.mu_zero.experiments.utils import stringify
from open_spiel.python.policy import TabularPolicy
from open_spiel.python.algorithms.mu_zero.jax_games.muzero_leduc_gameplay import MuZeroLeducGameplay, MuZeroGameplayConfig, MuZeroLeducCFR
from open_spiel.python.algorithms.mu_zero.mu_zero_train import MuZeroTrain
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

def extract_policy_from_muzero(muzero: MuZeroTrain, resolve_iterations=3000, init_state_info = None) ->JaxPolicy:
  """Extracts JaxPolicy for the full game
  from MuZeroLeducGameplay."""
  game = JaxLeduc()
  p1_config = MuZeroGameplayConfig(depth_limit = 5, resolve_iterations=resolve_iterations, player=0)

  p1_gameplay = MuZeroLeducGameplay(muzero, p1_config)

  after_chance_legals = np.array([[0, 0, 1, 1], [1, 0, 0, 0]])

  policy = JaxPolicy()

  dummy_key = jax.random.key(0)
  next_root_states = []
  next_root_turn = []

  def _traverse_tree(state, legals, cur_player=0, turn=0, depth=0):
    state_tensor, p1_iset, p2_iset, ps = game.get_info(state)
    #p1_gameplay.tree_depth = depth
    pl_iset = p1_iset if cur_player == 0 else p2_iset
    policy[stringify(pl_iset)] = p1_gameplay.cfr.get_strategy(pl_iset, cur_player, depth)
    for a1i, a1 in enumerate(legals[0]):
      for a2i, a2 in enumerate(legals[1]):
        if a1 < 0.5 or a2 < 0.5:
          continue
        new_state, terminal, reward, new_legals = game.apply_action(state, dummy_key, turn, jnp.array([a1i, a2i]))
        if terminal:
          continue
        #Do not traverse into the chance nodes. Create two CFRs
        if new_state.public_card > 0 and state.public_card == 0:
          pc_chance_outcomes = game.generate_all_public_card_nodes(new_state)
          for outcome in pc_chance_outcomes:
            next_root_states.append(outcome)
            next_root_turn.append(turn + 1)
          continue
        _traverse_tree(new_state, new_legals, 1 - cur_player, turn + 1, depth + 1)
  #We create the first CFR here
  if init_state_info is not None:
    start_root, init_legals, start_turns = init_state_info
    start_turns = np.array(start_turns, dtype=int)[..., None]
  else:
    start_root, init_legals = game.generate_all_private_card_nodes()
    start_turns = np.zeros((len(start_root), 1), dtype=int)

  start_root = jax.tree_map(lambda *x: jnp.stack(x), *start_root)
  init_legals = np.tile(init_legals[:, None, ...], (1, start_root.terminal.shape[0], 1))
  start_reaches = np.full((2, start_root.terminal.shape[0]), 1 / start_root.terminal.shape[0])
  start_cf_values = np.zeros(start_root.terminal.shape[0])

  p1_gameplay.prepare_cfr_structure(start_turns, start_root, init_legals, start_reaches, start_cf_values, False)
  p1_gameplay.run_cfr()
  if init_state_info is not None:
    init_chance_outcomes, init_legals, init_turns = init_state_info
  else:
    init_chance_outcomes, init_legals = game.generate_all_private_card_nodes()
    init_turns = [0] * len(init_chance_outcomes)
  for outcome, turn in zip(init_chance_outcomes, init_turns):
    _traverse_tree(outcome, init_legals, turn = turn)

  #Then we find the next root. In this case
  # we take the entire bottom level of nodes
  def get_next_root(cfr: MuZeroLeducCFR, resolving_player: int):
    depth_reaches, depth_chance_reaches = cfr.find_reaches_from_average()
    last_depth_reaches, last_depth_chance_reaches = depth_reaches[-1], depth_chance_reaches[-1]
    last_depth_reaches = np.where(np.array([[resolving_player == 0], [resolving_player == 1]]), last_depth_reaches * last_depth_chance_reaches[None, ...], 1.0)
    #DO NOT FORGET THAT WE WANT THE OPPONENT CF VALUES
    last_depth_cf_values = cfr.get_last_depth_player_cf_values(1 - resolving_player)
    return last_depth_reaches, last_depth_cf_values

  p1_next_reaches, p1_next_cf_vals = get_next_root(p1_gameplay.cfr, 0)
  next_root_stacked = jax.tree_map(lambda *x: jnp.stack(x), *next_root_states)
  next_legals_repeated = np.tile(after_chance_legals[:, None, :], (1, next_root_stacked.terminal.shape[0], 1))

  after_chance_turns = np.array(next_root_turn, dtype=int)[:, None]
  p1_gameplay.prepare_cfr_structure(after_chance_turns, next_root_stacked, next_legals_repeated, p1_next_reaches, p1_next_cf_vals, True)

  p1_gameplay.run_cfr()

  for state, turn in zip(next_root_states, next_root_turn):
    _traverse_tree(state, after_chance_legals, cur_player=0, turn = turn, depth=1)
  return policy


def jax_policy_to_tabular(jax_policy: JaxPolicy) ->TabularPolicy:
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
    if np.abs(np.sum(state_pols) - 1) > 1e-5:
      print("Policy does not sum up to 1!")
      breakpoint()
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


def compare_policies(pols1: TabularPolicy, pols2:TabularPolicy, epsilon = 0.01):
  """ Check whether the two given tabular
  policies for JaxLeduc differ by more then epsilon
  """
  game = pyspiel.load_game("leduc_poker")
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
    state_pols1 = pols1.policy_for_key(pl_iset_str)
    state_pols2 = pols2.policy_for_key(pl_iset_str)
    #Do not forget to skip the invalid action
    if np.sum(np.abs(state_pols1 - state_pols2)) > epsilon:
      print("Policies differ by more then epsilon")
      breakpoint()
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


def check_subgame(muzero: MuZeroTrain, resolve_iterations= 3000, epsilon = 1e-5):
    """Check a single subgame 
    and compare MuZeroLeducCFR and JaxLeducCFR policies.
    Just a sanity check that MuZeroLeducCFR works as it should"""
    action_history = [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]
    action_history = jnp.array(action_history)
    current_chips = jnp.array([1, 1])
    private_cards = jnp.array([2, 4])
    public_card = jnp.array([0])
    turns_this_round = jnp.array([0])
    terminal = jnp.array(False)
    starting_turn = 0
    legals = jnp.array([[0, 0, 1, 1], [1, 0, 0, 0]])
    root_state = LeducGameState(action_history = action_history,
                                current_chips = current_chips,
                                private_cards = private_cards,
                                public_card = public_card,
                                turns_this_round = turns_this_round,
                                terminal = terminal)
    game = JaxLeduc()
    dummy_key = jax.random.key(0)
    start_state_info=(root_state, legals, starting_turn)
    jax_cfr = JaxLeducCFR(start_state_info)
    jax_cfr.multiple_steps(resolve_iterations)
    reference_pols = jax_cfr.average_policy().policy
    muzero_start_state_info = ([root_state], legals, [starting_turn])
    muzero_pols = extract_policy_from_muzero(muzero, resolve_iterations, muzero_start_state_info).policy
    def _traverse_tree(state, legals, cur_player=0, turn=0, depth=0):
      state_tensor, p1_iset, p2_iset, ps = game.get_info(state)
      pl_iset = p1_iset if cur_player == 0 else p2_iset
      pl_iset_str = stringify(pl_iset)
      reference_state_pols = reference_pols[pl_iset_str]
      state_pols = muzero_pols[pl_iset_str]
      if np.sum(np.abs(reference_state_pols - state_pols)) > epsilon:
        print("State policies differ by more than epsilon!")
        breakpoint()
        #pass
      for a1i, a1 in enumerate(legals[0]):
        for a2i, a2 in enumerate(legals[1]):
          if a1 < 0.5 or a2 < 0.5:
            continue
          new_state, terminal, reward, new_legals = game.apply_action(state, dummy_key, turn, jnp.array([a1i, a2i]))
          if terminal:
            continue
          if new_state.public_card > 0 and state.public_card == 0:
            pc_chance_outcomes = game.generate_all_public_card_nodes(new_state)
            for outcome in pc_chance_outcomes:
              _traverse_tree(outcome, new_legals, 0, turn + 1, depth + 1)
          else:
            _traverse_tree(new_state, new_legals, 1 - cur_player, turn + 1, depth + 1)
    _traverse_tree(root_state, legals, turn= starting_turn)
    


