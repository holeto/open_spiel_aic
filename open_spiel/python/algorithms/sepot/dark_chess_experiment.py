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
import pyspiel
# from open_spiel.python.algorithms.

parser = argparse.ArgumentParser()
# Experiments specific arguments

parser.add_argument("--iterations", default=101, type=int, help="Amount of main iterations (each saves model)")
parser.add_argument("--save_each", default=50, type=int, help="Length of each iteration in seconds")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

# (Se)RNaD experiment specific arguments
parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--entropy_schedule", default=[500, 10000], nargs="+", type=int, help="Entropy schedule")
parser.add_argument("--entropy_schedule_repeats", default=[200, 1], nargs="+", type=int, help="Entropy schedule repeats")
parser.add_argument("--rnad_network_layers", default=[32, 32], nargs="+", type=int, help="Network layers")
parser.add_argument("--mvs_network_layers", default=[32, 32], nargs="+", type=int, help="Network layers")
parser.add_argument("--transformation_network_layers", default=[32, 32], nargs="+", type=int, help="Network layers")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning Rate")
parser.add_argument("--c_vtrace", default=1.5, type=float, help="Clipping of vtrace")
parser.add_argument("--rho_vtrace", default=np.inf, type=float, help="Clipping of vtrace")
parser.add_argument("--eta", default=0.2, type=float, help="Regularization term")
parser.add_argument("--num_transformations", default=10, type=int, help="Transformations of both players")

# Game Setting 


def train():
  args = parser.parse_args([] if "__file__" not in globals() else None)
  game_name = "dark_chess" 
 
  save_folder = "sepot_networks/dark_chess"
  if not os.path.exists(save_folder):
    os.makedirs(save_folder)
  
  max_trajectory = 80
  rnad_config = rnad.RNaDConfig(
      game_name = game_name, 
      game_params = tuple(),
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
      seed=  args.seed,
      state_representation = rnad.StateRepresentation.OBSERVATION
  )
  i = 0

  chess = pyspiel.load_game("chess")
  dark_chess = pyspiel.load_game("dark_chess")
  print(chess.observation_tensor_shape())
  print(dark_chess.observation_tensor_shape())
  print(dark_chess.state_tensor_shape())
  solver =  rnad.RNaDSolver(rnad_config)
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
        # print(solver.mvs_params)
        pickle.dump(solver, f)
      print("Saved at iteration", iteration, "after", int(time.time() - start), flush=True)

    # Prints time each hour
    if time.time() > print_iter_time:
      print("Iteration ", iteration, flush=True)

      print_iter_time = time.time() + 60 * 60
  profiler.stop()
  print(profiler.output_text(color=True, unicode=True))
  i+= 1 
      
     


if __name__ == "__main__":
  train()
  