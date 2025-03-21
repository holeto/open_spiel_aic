
from open_spiel.python.algorithms.mu_zero.mu_zero_train import MuZeroTrain, MuZeroTrainConfig
from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import MuZeroGameplay, MuZeroGameplayConfig
from open_spiel.python.algorithms.mu_zero.muzero_gameplay_passing_constants import MuZeroConstantsGameplay

from open_spiel.python.algorithms.mu_zero.jax_games.jax_goofspiel import JaxGoofspiel
import numpy as np
import jax.numpy as jnp
import jax
import pickle
import argparse
import os

from pyinstrument import Profiler
from memory_profiler import profile

parser = argparse.ArgumentParser()
#Solver setting
parser.add_argument("--solver_save_folder", type=str, default="muzero_networks/goofspiel_3/", help="Path where to save the trained networks.")
parser.add_argument("--iterations", type=int, default=10000, help="MuZero network training iterations")
parser.add_argument("--resolve_iterations", type=int, default=3000, help="CFR resolving iterations")
parser.add_argument("--saved_model_path", type=str, default="muzero_networks/goofspiel_3_descending/seed_50/muzero_9.pkl", help="Path to the already trained model")

#Resolve setting
parser.add_argument("--player", type=int, default=0, choices=(0, 1), help="Resolving player.")
parser.add_argument("--runs", type=int, default=1, help="How many gameplays to do.")
parser.add_argument("--seed", type=int, default=99, help="Numpy seed for choosing actions during gameplay.")

#Game setting:
parser.add_argument("--cards", type=int, default=3, help="Goofspiel cards")

def pickle_load(filepath: str):
  with open(filepath, "rb") as f:
    data= pickle.load(f)
  return data

def pickle_dump(filepath: str, data):
   with open(filepath, "wb") as f:
     pickle.dump(data, f)
@profile
def run_game(runs, player, muzero_gameplay, game, steps = 100):
  opp = 1 - player
  dummy_key = jax.random.key(0)
  
  for i in range(runs):
    game_state, legals = game.initialize_structures(dummy_key)
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
      print("Action: ", actions)
      game_state, terminal, rewards, legals = game.apply_action(game_state, dummy_key, turn, actions)
      opp_legals = legals[opp]
      turn += 1
      _, p1_iset, p2_iset, ps = game.get_info(game_state)
      cur_steps += 1

def goofspiel_test(args):
  
  points_order = "descending"
  opp = 1 - args.player
  
  config = MuZeroTrainConfig(batch_size=64, trajectory_max=args.cards-1, use_abstraction=True, entropy_schedule_size=(1000,), sampling_epsilon=0.8, abstraction_amount=10, transformations=4, similarity_metric="policy")

  
  muzero = None
  if not os.path.exists(args.saved_model_path):
    if not os.path.exists(args.solver_save_folder):
      os.makedirs(args.solver_save_folder)
    game = JaxGoofspiel(args.cards, points_order)

    muzero = MuZeroTrain(game, config)
    
    muzero.multiple_jax_steps(args.iterations)
    pickle_dump(filepath=args.solver_save_folder + "muzero_" + str(args.iterations) + ".pkl", data=muzero)
  else:
    muzero = pickle_load(filepath=args.saved_model_path)
  
  
  np.random.seed(args.seed)

  gp_config =MuZeroGameplayConfig(player=args.player, resolve_iterations= args.resolve_iterations)
  profiler = Profiler()
  muzero_gameplay = MuZeroConstantsGameplay(muzero, gp_config)
  profiler.start()
  run_game(args.runs, args.player, muzero_gameplay, muzero.game)
  profiler.stop()
  print(profiler.output_text(unicode=True))
  
  # muzero_gameplay = MuZeroGameplay(muzero, gp_config)
  # profiler.start()
  # run_game(args.runs, args.player, muzero_gameplay, muzero.game)
  # profiler.stop()
  # print(profiler.output_text(unicode=True))


def loaded_test(cards):
  # cards = 3
  folder = "muzero_networks/goofspiel_" + str(cards) + "/"
  file_name = "cfr_abstraction_39.pkl"
  with open(folder + file_name, "rb") as f:
    muzero = pickle.load(f)
    
  player = 0
  opponent = 1 - player
  gp_config = MuZeroGameplayConfig(player=player, depth_limit=1)
  muzero_gp = MuZeroGameplay(muzero, gp_config)
  
  init_info = muzero.game.initialize_structures()
  
  _, p1_iset, p2_iset, ps = muzero.game.get_info(*init_info[:-1])
  
  a1 = muzero_gp.get_action(ps, p1_iset) 
  
  a2 = np.random.choice(init_info[-1][opponent])
  
  legals, rewards, point_cards, played_cards, p1_points = muzero.game.apply_action(*init_info[:-1], 0, jnp.array([a1, a2]))
  info = (point_cards, played_cards, p1_points)
  _, p1_iset, p2_iset, ps = muzero.game.get_info(*info)
  a1 = muzero_gp.get_action(ps, p1_iset) 
  
    
  
  

def main():
  cards = 3
  args = parser.parse_args()
  goofspiel_test(args)
  #loaded_test(cards)
  

if __name__ == "__main__":
  main()