import argparse
import numpy as np
import time
import os
import jax
import jax.numpy as jnp
import pyspiel
import pickle as pkl
from pyinstrument import Profiler

from open_spiel.python.algorithms.rnad.rnad import RNaDSolver, RNaDConfig
#from open_spiel.python.algorithms.sepot.utils import take_policy_from_rnad
from open_spiel.python.policy import TabularPolicy
from open_spiel.python.algorithms.best_response import BestResponsePolicy
from open_spiel.python.algorithms.get_all_states import get_all_states
from sim_rnad import RNaDSimulataneous, MuZeroConfig
from jax_goofspiel import JaxOriginalGoofspiel

parser = argparse.ArgumentParser()

parser.add_argument("--iterations", default=200001, type=int, help="Amount of main iterations (each saves model)")
parser.add_argument("--save_each", default=20000, type=int, help="Length of each iteration in seconds")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

#Game setting
parser.add_argument("--cards", type=int, default=3, help= "Number of goofspiel cards")
parser.add_argument("--points_order", type=str, default="descending", help= "Goofspiel point card order.")

# RNaD experiment specific arguments
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--entropy_schedule", default=[1000, 10000], nargs="+", type=int, help="Entropy schedule")
parser.add_argument("--entropy_schedule_repeats", default=[30, 1], nargs="+", type=int, help="Entropy schedule repeats")
parser.add_argument("--rnad_network_layers", default=[256, 256], nargs="+", type=int, help="Network layers")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning Rate")
parser.add_argument("--c_vtrace", default=np.inf, type=float, help="Clipping of vtrace")
parser.add_argument("--rho_vtrace", default=np.inf, type=float, help="Clipping of vtrace")
parser.add_argument("--eta", default=0.2, type=float, help="Regularization term")

#file save arguments
parser.add_argument("--solver_save_folder", default= "rnad_networks/goofspiel_3")


def take_policy_from_rnad(solver) -> TabularPolicy:
  game = solver._game
  num_players = solver._game.num_players()
  #distinct_actions = solver.num_distinct_actions()
  rnad_pols = TabularPolicy(game)
  rollout = jax.vmap(solver.network.apply, (None, 0), 0)
  all_states = get_all_states(
    game,
    depth_limit=-1,
    include_terminals=False,
    include_chance_states=False,
    stop_if_encountered=False,
    to_string=lambda s: s.information_state_string())
  for iset in rnad_pols.state_lookup:
    state = all_states[iset]
    envs = [solver._state_as_env_step(state)]
    player = state.current_player()
    tree_map = jax.tree_util.tree_map(lambda *e: jnp.stack(e, axis=0), *envs)
    pi, v, log_pi, logit = rollout(solver.params_target, jax.tree_util.tree_map(lambda *e: jnp.stack(e, axis=0), *envs))
    iset = state.information_state_string()
    state_policy = rnad_pols.policy_for_key(iset)
    # TODO: Change this to some form of broadcast
    for i in range(len(state_policy)):
        state_policy[i] = pi[0][i]
  return rnad_pols

def train_and_save_solver(args, solver, file_name, is_jax = False):
  if not os.path.exists(args.solver_save_folder):
    os.makedirs(args.solver_save_folder)
  start = time.time()
  print_iter_time = time.time() # We will save the model in first step
  profiler = Profiler()
  profiler.start()
  for iteration in range(0, args.iterations):
    if is_jax:
       solver.goofspiel_step()
    else:
      solver.step()
    # print(iteration, flush=True)
    if iteration % args.save_each == 0:
        
      file = "/" + file_name + "_" + str(args.seed) + "_" + str(iteration) + ".pkl"
      file_path = args.solver_save_folder + file
      with open(file_path, "wb") as f:
        pkl.dump(solver.params, f)
      print("Saved at iteration", iteration, "after", int(time.time() - start), flush=True)

    # Prints time each hour
    if time.time() > print_iter_time:
      print("Iteration ", iteration, flush=True)

      print_iter_time = time.time() + 60 * 60
  profiler.stop()
  print(profiler.output_text(color=True, unicode=True))
  return solver

def compare_rnad_pols(rnad_orig_game, sim_orig_game, rnad_pols : TabularPolicy, sim_rnad_pols : TabularPolicy):
  rnad_p1_br = BestResponsePolicy(rnad_orig_game, 0, rnad_pols)
  rnad_p2_br = BestResponsePolicy(rnad_orig_game, 1, rnad_pols)

  sim_p1_br = BestResponsePolicy(sim_orig_game, 0, sim_rnad_pols)
  sim_p2_br = BestResponsePolicy(sim_orig_game, 1, sim_rnad_pols)

  print("RNaD exploitability: ")
  print(rnad_p1_br.value(rnad_orig_game.new_initial_state()))
  print(rnad_p2_br.value(rnad_orig_game.new_initial_state()))

  print("Sim RNaD exploitability: ")
  print(sim_p1_br.value(sim_orig_game.new_initial_state()))
  print(sim_p2_br.value(sim_orig_game.new_initial_state()))

def evaluate_rnad_pols(orig_game, rnad_pols: TabularPolicy):
  rnad_p1_br = BestResponsePolicy(orig_game, 0, rnad_pols)
  rnad_p2_br = BestResponsePolicy(orig_game, 1, rnad_pols)

  print("Exploitability: ")
  print(rnad_p1_br.value(orig_game.new_initial_state()))
  print(rnad_p2_br.value(orig_game.new_initial_state()))

def main():
  args = parser.parse_args()
  game_name = "goofspiel"
  game_params = (
        ("num_cards", args.cards),
        ("imp_info", True),
        ("points_order", args.points_order)
  )


  game_settings = {a:b for a,b in game_params}
  spiel_game = pyspiel.load_game("goofspiel", game_settings)
  tb_game = pyspiel.load_game_as_turn_based("goofspiel", game_settings)
  sim_game = JaxOriginalGoofspiel(args.cards, args.points_order)
  rnad_config = RNaDConfig(
      game_name = game_name, 
      game_params = game_settings,
      trajectory_max =  (args.cards - 1) * 2,
      policy_network_layers = args.rnad_network_layers,
      
      batch_size = args.batch_size,
      learning_rate = args.learning_rate,
      entropy_schedule_repeats = args.entropy_schedule_repeats,
      entropy_schedule_size = args.entropy_schedule,
      c_vtrace = args.c_vtrace,
      rho_vtrace = args.rho_vtrace,
      eta_reward_transform = args.eta,
      seed=  args.seed
  )
  rnad_solver = RNaDSolver(rnad_config)

  mu_zero_config = MuZeroConfig(
    batch_size = args.batch_size,
    trajectory_max = args.cards - 1,

    entropy_schedule_repeats = args.entropy_schedule_repeats,
    entropy_schedule_size = args.entropy_schedule,
    
    #learning_rate = args.learning_rate,
    seed=  args.seed

  )
  sim_rnad_solver = RNaDSimulataneous(sim_game, mu_zero_config)
  #rnad_solver = train_and_save_solver(args, rnad_solver, "rnad")
  sim_rnad_solver = train_and_save_solver(args, sim_rnad_solver, "sim_rnad", is_jax=True)
  #rnad_pols = take_policy_from_rnad(rnad_solver)
  #assert isinstance(sim_rnad_solver.game, JaxOriginalGoofspiel)
  sim_rnad_pols = sim_rnad_solver.extract_goofspiel_policy(spiel_game)
  #compare_rnad_pols(rnad_orig_game=tb_game, sim_orig_game=spiel_game, rnad_pols=rnad_pols, sim_rnad_pols=sim_rnad_pols)
  evaluate_rnad_pols(spiel_game, sim_rnad_pols)
  
if __name__ == "__main__":
  main()