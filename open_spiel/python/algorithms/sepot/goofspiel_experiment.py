# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import numpy as np
import pickle

import time


from open_spiel.python import policy
from open_spiel.python.algorithms import best_response

import open_spiel.python.algorithms.sepot.rnad_sepot as rnad  
import open_spiel.python.algorithms.sepot.sepot as sepot

from open_spiel.python.algorithms.get_all_states import get_all_states
from pyinstrument import Profiler

from open_spiel.python.algorithms.sepot.utils import compare_policies_mvs_rnad, resolve_first_subgame_then_rnad

parser = argparse.ArgumentParser()
# Experiments specific arguments

parser.add_argument("--iterations", default=200001, type=int, help="Amount of main iterations (each saves model)")
parser.add_argument("--save_each", default=20000, type=int, help="Length of each iteration in seconds")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

# (Se)RNaD experiment specific arguments
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--entropy_schedule", default=[1000, 10000], nargs="+", type=int, help="Entropy schedule")
parser.add_argument("--entropy_schedule_repeats", default=[30, 1], nargs="+", type=int, help="Entropy schedule repeats")
parser.add_argument("--rnad_network_layers", default=[256, 256], nargs="+", type=int, help="Network layers")
parser.add_argument("--mvs_network_layers", default=[256, 256], nargs="+", type=int, help="Network layers")
parser.add_argument("--transformation_network_layers", default=[256, 256], nargs="+", type=int, help="Network layers")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning Rate")
parser.add_argument("--c_vtrace", default=np.inf, type=float, help="Clipping of vtrace")
parser.add_argument("--rho_vtrace", default=np.inf, type=float, help="Clipping of vtrace")
parser.add_argument("--eta", default=0.2, type=float, help="Regularization term")
parser.add_argument("--num_transformations", default=10, type=int, help="Transformations of P1")

# Game Setting
parser.add_argument("--cards", default=5, type=int, help="Goofspiel cards")
parser.add_argument("--points_order", default="descending", type=str, help="Oredering of point card, choose between 'descending', 'ascending' and 'random'")

# Evaluate setting
parser.add_argument("--saved_model", default="sepot_networks/goofspiel_5_descending/full_sepot_test.pkl", type=str, help="Path to a saved model")

def train():
  args = parser.parse_args([] if "__file__" not in globals() else None)
  game_name = "goofspiel"
  game_params = (
        ("num_cards", args.cards),
        ("imp_info", True),
        ("points_order", args.points_order)
  )


  game_settings = {a:b for a,b in game_params}
  save_folder = "sepot_networks/goofspiel_" + str(args.cards) + "_" + str(args.points_order)
  if not os.path.exists(save_folder):
    os.makedirs(save_folder)

  max_trajectory = (args.cards - 1) * 2 
  rnad_config = rnad.RNaDConfig(
      game_name = game_name, 
      game_params = game_params,
      trajectory_max =  max_trajectory,
      policy_network_layers = args.rnad_network_layers,
      mvs_network_layers = args.mvs_network_layers,
      transformation_network_layers = args.transformation_network_layers,
      
      batch_size = args.batch_size,
      learning_rate = args.learning_rate,
      entropy_schedule_repeats = args.entropy_schedule_repeats,
      entropy_schedule_size = args.entropy_schedule,
      c_vtrace = args.c_vtrace,
      rho_vtrace = args.rho_vtrace,
      eta_reward_transform = args.eta,

      num_transformations = args.num_transformations,
      matrix_valued_states = True,
      seed=  args.seed
  )
  i = 0

  solver=  rnad.RNaDSolver(rnad_config)

  start = time.time()
  print_iter_time = time.time() # We will save the model in first step
  profiler = Profiler()
  profiler.start()
  for iteration in range(i, args.iterations + i):
    solver.step()
    # print(iteration, flush=True)
    if iteration % args.save_each == 0:
        
      file = "/rnad_" + str(args.seed) + "_" + str(iteration) + ".pkl"
      file_path = save_folder + file
      with open(file_path, "wb") as f:
        pickle.dump(solver, f)
      print("Saved at iteration", iteration, "after", int(time.time() - start), flush=True)

    # Prints time each hour
    if time.time() > print_iter_time:
      print("Iteration ", iteration, flush=True)

      print_iter_time = time.time() + 60 * 60
  profiler.stop()
  print(profiler.output_text(color=True, unicode=True))
  i+= 1 
     

def load():
  args = parser.parse_args([] if "__file__" not in globals() else None)
  game_name = "goofspiel"
  game_params = (
        ("num_cards", args.cards),
        ("imp_info", True),
        ("points_order", args.points_order)
  )


  game_settings = {a:b for a,b in game_params}
  save_folder = "sepot_networks/goofspiel_" + str(args.cards) + "_" + str(args.points_order)
  if not os.path.exists(save_folder):
    os.makedirs(save_folder) 


  with open("sepot_networks/goofspiel_4_descending/rnad_42_200.pkl", "rb") as f:
    solver = pickle.load(f)
  print(solver.mvs_params)

def full_sepot_test():
  args = parser.parse_args([] if "__file__" not in globals() else None)
  game_name = "goofspiel"
  game_params = (
        ("num_cards", args.cards),
        ("imp_info", True),
        ("points_order", args.points_order)
  )


  game_settings = {a:b for a,b in game_params}
  save_folder = "sepot_networks/goofspiel_" + str(args.cards) + "_" + str(args.points_order)
  if not os.path.exists(save_folder):
    os.makedirs(save_folder)

  max_trajectory = (args.cards - 1) * 2 
  rnad_config = rnad.RNaDConfig(
      game_name = game_name, 
      game_params = game_params,
      trajectory_max =  max_trajectory,
      policy_network_layers = args.rnad_network_layers,
      mvs_network_layers = args.mvs_network_layers,
      transformation_network_layers = args.transformation_network_layers,
      
      batch_size = args.batch_size,
      learning_rate = args.learning_rate,
      entropy_schedule_repeats = args.entropy_schedule_repeats,
      entropy_schedule_size = args.entropy_schedule,
      c_vtrace = args.c_vtrace,
      rho_vtrace = args.rho_vtrace,
      eta_reward_transform = args.eta,

      num_transformations = args.num_transformations,
      matrix_valued_states = True,
      seed=  args.seed
  )
  i = 0 
  
  sepot_config = sepot.SePoTConfig(
        rnad_config = rnad_config,
        resolve_iterations = 3000,
        subgame_size_limit = 10000000,
        subgame_depth_limit = 2)
  sepot_solver = sepot.SePoT_RNaD(sepot_config)
 
  profiler = Profiler()
  profiler.start()
  sepot_solver.train(args.iterations)
  
  
  save_folder = "sepot_networks/goofspiel_" + str(args.cards) + "_" + args.points_order
  if not os.path.exists(save_folder):
    os.makedirs(save_folder) 
  save_file = save_folder + "/full_sepot_test.pkl" 
  with open(save_file, "wb") as f:
    pickle.dump(sepot_solver, f)
    
  
  tab_policy = policy.TabularPolicy(sepot_solver.rnad._game)
  print("trained")
  all_states = get_all_states(
        sepot_solver.rnad._game,
        depth_limit=1000,
        include_terminals=False,
        include_chance_states=False,
        stop_if_encountered=False,
        to_string=lambda s: s.information_state_string())
  for state in all_states.values():
    if state.current_player() != 0 or state.information_state_string() in sepot_solver.policy:
      continue
    avg_policy = sepot_solver.compute_policy(state, 0)
    for iset, temp_policy in avg_policy.items():
      # print(iset)
      # print(temp_policy)
      sepot_solver.policy[iset] = temp_policy
      tab_policy_pat = tab_policy.policy_for_key(iset)
      for action, prob in enumerate(temp_policy):
        tab_policy_pat[action] = prob

  best_response_policy_2 = best_response.BestResponsePolicy(sepot_solver.rnad._game, 1, tab_policy)
  print(best_response_policy_2.value(sepot_solver.rnad._game.new_initial_state()))
  profiler.stop()
  print(profiler.output_text(color=True, unicode=True))


  i+= 1 
  

     
def evaluate():
  args = parser.parse_args([] if "__file__" not in globals() else None)

  with open(args.saved_model, "rb") as f:
    solver = pickle.load(f)
  # resolve_first_subgame_then_rnad(solver)
  compare_policies_mvs_rnad(solver)


if __name__ == "__main__":
  #full_sepot_test()
  evaluate()
  # train()