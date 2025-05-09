import pyspiel
import jax
import os
import pickle
import jax.numpy as jnp

from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc_algorithms import solve_full_game, check_subgame, compare_policies, jax_policy_to_tabular, extract_policy_from_muzero
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game_algorithms import stringify
#from open_spiel.python.policy import TabularPolicy
from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc import JaxLeduc
from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc_cfr import JaxLeducCFR
from open_spiel.python.algorithms.exploitability import exploitability
from open_spiel.python.algorithms.best_response import BestResponsePolicy
from open_spiel.python.algorithms.sequence_form_lp import solve_zero_sum_game
from open_spiel.python.jax.cfr.jax_cfr import JaxCFR
from argparse import ArgumentParser

from pyinstrument import Profiler

parser = ArgumentParser()
parser.add_argument("--resolve_iterations", type=int, default=3000, help="CFR iterations")
parser.add_argument("--saved_model_path", type=str, default="muzero_networks_no_abstraction/leduc/seed_134/muzero_10.pkl", help="Path to the trained MuZeroTrain model")


def compare_cfr_constants(full_game_cfr: JaxLeducCFR, spiel_game_cfr: JaxCFR):
  for pl in range(2):
    iset_action_depth_difference = jnp.sum(full_game_cfr.constants.iset_action_depth[pl] - spiel_game_cfr.constants.iset_action_depth[pl])
    assert iset_action_depth_difference == 0
    iset_action_mask_difference = jnp.sum(full_game_cfr.constants.iset_action_mask[pl][1:, 1:] - spiel_game_cfr.constants.iset_action_mask[pl][1:, :3])
    assert iset_action_mask_difference == 0
  depth = len(full_game_cfr.constants.depth_history_utility)
  for i in range(depth):
    #The spiel_game_cfr has 2 additional 
    #layers that are the chance nodes
    spiel_cfr_depth = i + 2
    assert jnp.sum(full_game_cfr.constants.depth_history_chance[i] - spiel_game_cfr.constants.depth_history_chance[spiel_cfr_depth]) == 0
    #chance_probability_difference = jnp.sum(full_game_cfr.constants.depth_history_chance_probabilities[i] - spiel_game_cfr.constants.depth_history_chance_probabilities[spiel_cfr_depth][:, :-2])
    #assert chance_probability_difference == 0
  for pl in range(2):
    for i in range(depth):
      spiel_cfr_depth = i + 2
      player_utility_difference = jnp.sum(jnp.abs((full_game_cfr.constants.depth_history_utility[i] * (1 - (2 * pl)) * 13) - spiel_game_cfr.constants.depth_history_utility[pl][spiel_cfr_depth]))
      assert player_utility_difference <= 1e-3
      iset_difference = jnp.sum(full_game_cfr.constants.depth_history_iset[pl][i] - spiel_game_cfr.constants.depth_history_iset[pl][spiel_cfr_depth])
      assert iset_difference == 0
      prev_iset_difference = jnp.sum(full_game_cfr.constants.depth_history_previous_iset[pl][i] - spiel_game_cfr.constants.depth_history_previous_iset[pl][spiel_cfr_depth])
      assert prev_iset_difference == 0

def check_root_policy(jax_game: JaxLeduc, policy):
  root_states, root_legals = jax_game.generate_all_private_card_nodes()
  for state in root_states:
    state_tensor, p1_iset, p2_iset, ps = jax_game.get_info(state)
    print("State: ", state)
    print("P1 policy: ", policy[stringify(p1_iset)])

def check_full_game_exploitability(args):
  spiel_game = pyspiel.load_game("leduc_poker")
  jax_game = JaxLeduc()
  dummy_key = jax.random.key(0)
  full_game_cfr = JaxLeducCFR()
  spiel_game_cfr = JaxCFR(spiel_game)
  compare_cfr_constants(full_game_cfr, spiel_game_cfr)
  full_game_cfr.multiple_steps(iterations=args.resolve_iterations)
  found_pols = full_game_cfr.average_policy()
  #check_root_policy(jax_game, found_pols)
  tabular_pols = jax_policy_to_tabular(found_pols)
  spiel_game_cfr.multiple_steps(iterations=3000)
  spiel_cfr_pols = spiel_game_cfr.average_policy()
  print("CFR exploitability: ", exploitability(spiel_game, spiel_cfr_pols))
  print("JAX CFR exploitability: ", exploitability(spiel_game, tabular_pols))
  return tabular_pols

def check_muzero_exploitability(args):
    spiel_game = pyspiel.load_game("leduc_poker")
    if not os.path.exists(args.saved_model_path):
      assert False, "MuZero model at " + str(args.saved_model_path) + " was not found!"
    with open(args.saved_model_path, "rb") as f:
      muzero = pickle.load(f)
    muzero_pols = extract_policy_from_muzero(muzero, args.resolve_iterations)
    muzero_spiel_pols = jax_policy_to_tabular(muzero_pols)
    p1_br = BestResponsePolicy(spiel_game, 1, muzero_spiel_pols)
    p2_br = BestResponsePolicy(spiel_game, 0, muzero_spiel_pols)
    p1_val, p2_val, p1_nash_pols, p2_nash_pols = solve_zero_sum_game(spiel_game)
    print("Real p1_val: ", p1_val)
    print("MuZero P1 Best response val", p1_br.value(spiel_game.new_initial_state()))
    print("Real p2_val: ", p2_val)
    print("MuZero P2 Best response val", p2_br.value(spiel_game.new_initial_state()))
    print("MuZero exploitability: ", exploitability(spiel_game, muzero_spiel_pols))
    return muzero_spiel_pols

def check_subgame_test(args):
  if not os.path.exists(args.saved_model_path):
    assert False, "MuZero model at " + str(args.saved_model_path) + " was not found!"
  with open(args.saved_model_path, "rb") as f:
      muzero = pickle.load(f)
  check_subgame(muzero, args.resolve_iterations, epsilon=1e-5)


def main():
  args = parser.parse_args()
  #check_subgame_test(args)
  #profiler = Profiler()
  #profiler.start()
  #full_game_pols = check_full_game_exploitability(args)
  muzero_pols = check_muzero_exploitability(args)
  #compare_policies(full_game_pols, muzero_pols, epsilon = 0.01)
  #profiler.stop()
  #print(profiler.output_text(unicode=True, color=True))
  


if __name__ == "__main__":
  main()