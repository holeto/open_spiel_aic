
import os
import argparse
import numpy as np
import pickle
import jax
import time

import open_spiel.python.algorithms.sepot.rnad_sepot as rnad

from open_spiel.python.algorithms.get_all_states import get_all_states
from pyinstrument import Profiler
import pyspiel
# from open_spiel.python.algorithms.

parser = argparse.ArgumentParser()
# Experiments specific arguments

parser.add_argument("--game_simulations", default=30, type=int, help="Amount of main iterations (each saves model)")
parser.add_argument("--model_path", default="sepot_networks/dark_chess/rnad_666321", type=str, help="Length of each iteration in seconds") 
parser.add_argument("--iterations_range", default=[200000, 210000, 5000], nargs="+", type=int, help="Ship sizes")
parser.add_argument("--postprocessing", default=1, type=int, help="Postprocessing of the policy. 0 for none, 1 for thresholding, 2 for threshholding and discretization")
 
parser.add_argument("--seed", default=42, type=int, help="Random seed")

parser.add_argument("--evaluate_type", default="random", type=str, help="Type of evaluation. Could be random or older")

def get_policy_from_network(network, state, postprocessing):
  if postprocessing == 0:
    env_step = network._batch_of_states_as_env_step([state])
    probs = network._network_jit_apply(
        network.params_target, env_step)
    probs = jax.device_get(probs[0]).astype(np.float64)  # Squeeze out the 1-element batch.
    probs = probs / np.sum(probs)
    
    return {
        action: probs[action]
        for action, valid in enumerate(jax.device_get(env_step.legal[0]))
        if valid
    }
  elif postprocessing == 1:
    env_step = network._batch_of_states_as_env_step([state])
    probs = network._network_jit_apply(
        network.params_target, env_step)
    probs = jax.device_get(probs[0]).astype(np.float64)   # Squeeze out the 1-element batch.
    probs_thresholded = np.where(probs > 0.02, probs, 0.0)
    probs = probs / np.sum(probs) if np.sum(probs_thresholded) < 0.01 else probs_thresholded / np.sum(probs_thresholded)
    return {
        action: probs[action]
        for action, valid in enumerate(jax.device_get(env_step.legal[0]))
        if valid
    }
    pass
  elif postprocessing == 2:
    return network.action_probabilities(state)

def evaluate_network_with_older(args):
  np_rng = np.random.RandomState(args.seed)
  last_iteration = list(range(*args.iterations_range))[-1]
  model_to_eval = args.model_path + "_" + str(last_iteration) + ".pkl"
  with open(model_to_eval, "rb") as f:
    evaluated_model = pickle.load(f)
  for i in range(*args.iterations_range):
    model_path = args.model_path + "_" + str(i) + ".pkl"
    with open(model_path, "rb") as f:
      compared_model = pickle.load(f)
    for player in range(2): # For each player
      result = []
      models = [evaluated_model, compared_model] if player == 0 else [compared_model, evaluated_model]
      for _ in range(args.game_simulations):
        state = compared_model._game.new_initial_state()
        while not state.is_terminal(): 
          if state.current_player() >= 0: 
            aps = get_policy_from_network(models[state.current_player()], state, args.postprocessing)
            actions, probabilities = [], []
            for action, probs in aps.items():
              actions.append(action)
              probabilities.append(probs)
            action = np_rng.choice(actions, p=probabilities)
          else:
            action = np_rng.choice(state.legal_actions())
          state.apply_action(action)
        returns = state.returns()
        result.append(returns[player])
      print("Iteration", i, ";Player", player, ";mean:", np.mean(result), ";std:", np.std(result), flush=True)
  
def evaluate_network_to_random(args):
  np_rng = np.random.RandomState(args.seed)
  # results = [[], []]
  for i in range(*args.iterations_range):
    model_path = args.model_path + "_" + str(i) + ".pkl"
    with open(model_path, "rb") as f:
      network = pickle.load(f)
    for player in range(2): # For each player
      result = []
      for _ in range(args.game_simulations):
        state = network._game.new_initial_state()
        while not state.is_terminal(): 
          if state.current_player() == player:
            aps = get_policy_from_network(network, state, args.postprocessing)
            actions, probabilities = [], []
            for action, probs in aps.items():
              actions.append(action)
              probabilities.append(probs) 
            action = np_rng.choice(actions, p=probabilities)
          else:
            action = np_rng.choice(state.legal_actions())
          state.apply_action(action)
        returns = state.returns()
        result.append(returns[player])
      print("Iteration", i, ";Player", player, ";mean:", np.mean(result), ";std:", np.std(result), flush=True)

def evaluate_network():
  """Evaluates a network by playing it against random player."""
  args = parser.parse_args([] if "__file__" not in globals() else None)
  assert len(args.iterations_range) == 3
  if args.evaluate_type == "random":
    evaluate_network_to_random(args)
  elif args.evaluate_type == "older":
    evaluate_network_with_older(args)
  else:
    raise ValueError("Unknown evaluate type")
  
  # for _ in range(num_games):
  #   state = game.new_initial_state()
  #   while not state.is_terminal():
  #     current_player = state.current_player()
  #     legal_actions = state.legal_actions()
  #     action = network.sample_action(state)
  #     state.apply_action(action)
  #   returns = state.returns()
  #   for player in range(num_players):
  #       outcomes[returns[player], player] += 1
  # return outcomes / num_games



if __name__ == "__main__":
  
  evaluate_network()