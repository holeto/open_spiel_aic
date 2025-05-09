from open_spiel.python.algorithms.mu_zero.mu_zero_train import MuZeroTrain, MuZeroTrainConfig
from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import MuZeroGameplayConfig
from open_spiel.python.algorithms.mu_zero.jax_games.muzero_leduc_gameplay import MuZeroLeducGameplay

from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc import JaxLeduc
import numpy as np
import jax.numpy as jnp
import jax
import pickle
import argparse
import os

from pyinstrument import Profiler

parser = argparse.ArgumentParser()
#Solver setting
parser.add_argument("--solver_save_folder", type=str, default="muzero_networks/goofspiel_3/", help="Path where to save the trained networks.")
parser.add_argument("--iterations", type=int, default=10000, help="MuZero network training iterations")
parser.add_argument("--resolve_iterations", type=int, default=3000, help="CFR resolving iterations")
parser.add_argument("--saved_model_path", type=str, default="muzero_networks_no_abstraction/leduc/seed_134/muzero_9.pkl", help="Path to the already trained model")

#Resolve setting
parser.add_argument("--player", type=int, default=0, choices=(0, 1), help="Resolving player.")
parser.add_argument("--runs", type=int, default=1, help="How many gameplays to do.")
parser.add_argument("--seed", type=int, default=99, help="Numpy seed for choosing actions during gameplay.")
parser.add_argument("--game_seed", type=int, default=42, help="Seed for sampling chance nodes in JaxLeduc")


def pickle_load(filepath: str):
  with open(filepath, "rb") as f:
    data= pickle.load(f)
  return data

def pickle_dump(filepath: str, data):
   with open(filepath, "wb") as f:
     pickle.dump(data, f)

def run_game(runs, player, muzero_gameplay, game, game_seed, steps = 100):
  opp = 1 - player
  key = jax.random.key(seed=game_seed)
  
  for i in range(runs):
    key, init_key = jax.random.split(key)
    game_state, legals = game.initialize_structures(init_key)
    opp_legals = legals[opp]
    
    _, p1_iset, p2_iset, ps = game.get_info(game_state)
    turn = 0
    terminal = False
    cur_steps = 0
    muzero_gameplay.reset()
    while not terminal and cur_steps < steps:
      pl_action = muzero_gameplay.get_action(ps, p1_iset)
      #random opponent
      opp_actions = np.arange(opp_legals.shape[0])
      opp_legals = np.asarray(opp_legals, dtype="float64")
      opp_action = np.random.choice(opp_actions, p= opp_legals / np.sum(opp_legals)) 
      #Take the allowed action with maximum index, this is just a workaround
      # for now to prevent going into too large public states (larger than the maximum possible size)
      # where the CFR with passing constants will not work
      #opp_action = np.argmax(np.where(opp_legals.astype(bool), opp_actions, np.zeros_like(opp_actions) - 1))
      actions = [[],[]]
      actions[player] = pl_action
      actions[opp] = opp_action
      actions = jnp.stack(actions, axis=0)
      print("State: ", game_state)
      print("Action: ", actions)
      key, action_key = jax.random.split(key)
      game_state, terminal, rewards, legals = game.apply_action(game_state, action_key, turn, actions)
      opp_legals = legals[opp]
      turn += 1
      _, p1_iset, p2_iset, ps = game.get_info(game_state)
      cur_steps += 1

def leduc_test(args):
  
  opp = 1 - args.player
  
  game = JaxLeduc()
  config = MuZeroTrainConfig(batch_size=64, trajectory_max=game.max_turns, use_abstraction=False, entropy_schedule_size=(1000,), sampling_epsilon=0.8, abstraction_amount=10, transformations=4, similarity_metric="policy")

  
  muzero = None
  if not os.path.exists(args.saved_model_path):
    if not os.path.exists(args.solver_save_folder):
      os.makedirs(args.solver_save_folder)

    muzero = MuZeroTrain(game, config)
    
    muzero.multiple_jax_steps(args.iterations)
    pickle_dump(filepath=args.solver_save_folder + "muzero_" + str(args.iterations) + ".pkl", data=muzero)
  else:
    muzero = pickle_load(filepath=args.saved_model_path)
  
  
  np.random.seed(args.seed)

  gp_config =MuZeroGameplayConfig(player=args.player, resolve_iterations= args.resolve_iterations, depth_limit=5)
  #profiler = Profiler()
  muzero_gameplay = MuZeroLeducGameplay(muzero, gp_config)
  #profiler.start()
  run_game(args.runs, args.player, muzero_gameplay, muzero.game, args.game_seed)
  #profiler.stop()
  #print(profiler.output_text(unicode=True))
  
  # muzero_gameplay = MuZeroGameplay(muzero, gp_config)
  # profiler.start()
  # run_game(args.runs, args.player, muzero_gameplay, muzero.game)
  # profiler.stop()
  # print(profiler.output_text(unicode=True))      

def main():
  args = parser.parse_args()
  leduc_test(args)
  

if __name__ == "__main__":
  main()