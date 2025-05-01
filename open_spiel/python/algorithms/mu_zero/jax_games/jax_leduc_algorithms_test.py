import pyspiel
import jax
import jax.numpy as jnp

from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc_algorithms import solve_full_game, exploitability_jax_leduc, jax_policy_to_tabular
from open_spiel.python.algorithms.mu_zero.jax_games.jax_game_algorithms import stringify
#from open_spiel.python.policy import TabularPolicy
from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc import JaxLeduc
from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc_cfr import JaxLeducCFR
from open_spiel.python.algorithms.exploitability import exploitability
#from open_spiel.python.algorithms.cfr import CFRPlusSolver
from open_spiel.python.jax.cfr.jax_cfr import JaxCFR
from open_spiel.python.jax.cfr.jax_simultaneous_cfr import SimultaneousJaxCFR
from open_spiel.python.algorithms.sequence_form_lp import solve_zero_sum_game


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



def main():
  spiel_game = pyspiel.load_game("leduc_poker")
  jax_game = JaxLeduc()
  dummy_key = jax.random.key(0)
  full_game_cfr = JaxLeducCFR()
  spiel_game_cfr = JaxCFR(spiel_game)
  compare_cfr_constants(full_game_cfr, spiel_game_cfr)
  full_game_cfr.multiple_steps(iterations=3000)
  found_pols = full_game_cfr.average_policy()
  tabular_pols = jax_policy_to_tabular(found_pols)
  init_state, init_legals = jax_game.initialize_structures(dummy_key)
  state_tensor, p1_iset, p2_iset, ps = jax_game.get_info(init_state)
  jax_init_iset = stringify(p1_iset)
  print("JAX init state policy: ", found_pols[jax_init_iset])
  #p1_val, p2_val, p1_nash_pols, p2_nash_pols = solve_zero_sum_game(spiel_game)
  spiel_game_cfr.multiple_steps(iterations=3000)
  spiel_cfr_pols = spiel_game_cfr.average_policy()
  spiel_state = spiel_game.new_initial_state()
  #skip past the initial chance nodes
  spiel_state.apply_action(spiel_state.chance_outcomes()[0][0])
  spiel_state.apply_action(spiel_state.chance_outcomes()[0][0])
  spiel_init_iset = spiel_state.information_state_string(0)
  print("Spiel init state policy: ", spiel_cfr_pols.policy_for_key(spiel_init_iset))
  print("CFR exploitability: ", exploitability(spiel_game, spiel_cfr_pols))
  print("JAX CFR exploitability: ", exploitability(spiel_game, tabular_pols))


if __name__ == "__main__":
  main()