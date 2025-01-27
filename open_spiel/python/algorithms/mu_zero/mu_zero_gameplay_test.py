
from open_spiel.python.algorithms.mu_zero.mu_zero import MuZeroTrain, MuZeroTrainConfig
from open_spiel.python.algorithms.mu_zero.mu_zero_gameplay import MuZeroGameplay, MuZeroGameplayConfig

from open_spiel.python.algorithms.mu_zero.jax_goofspiel import JaxOriginalGoofspiel
import numpy as np
import jax.numpy as jnp
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
#Solver setting
parser.add_argument("--solver_save_folder", type=str, default="muzero_networks/goofspiel_3/", help="Path to the saved trained networks")
parser.add_argument("--iterations", type=int, default=10000, help="MuZero network training iterations")

#Resolve setting
parser.add_argument("--player", type=int, default=0, choices=(0, 1), help="Resolving player.")
parser.add_argument("--steps", type=int, default=3, help="How many times to choose action and act in the game")

#Game setting:
parser.add_argument("--cards", type=int, default=3, help="Goofspiel cards")

def pickle_load(filepath: str):
  with open(filepath, "rb") as f:
    data= pickle.load(f)
  return data

def pickle_dump(filepath: str, data):
   with open(filepath, "wb") as f:
     pickle.dump(data, f)

def goofspiel_test(args):
  
  points_order = "descending"
  opp = 1 - args.player
  
  config = MuZeroTrainConfig(batch_size=64, trajectory_max=args.cards-1, use_abstraction=True, entropy_schedule_size=(1000,), sampling_epsilon=0.8, abstraction_amount=10, transformations=4, similarity_metric="policy")
  solver_path = args.solver_save_folder + "cfr_abstraction_39.pkl"#"mu_zero_train_iters" + args.iterations + ".pkl"

  if not os.path.exists(args.solver_save_folder):
    os.makedirs(args.solver_save_folder)
  muzero = None
  if not os.path.exists(solver_path):
    game = JaxOriginalGoofspiel(args.cards, points_order)

    muzero = MuZeroTrain(game, config)
    
    muzero.multiple_goofspiel_steps(args.iterations)
    pickle_dump(filepath=solver_path, data=muzero)
  else:
    muzero = pickle_load(filepath=solver_path)
  
  gp_config =MuZeroGameplayConfig(player=args.player)
  muzero_gameplay = MuZeroGameplay(muzero, gp_config)
  
  init_info = muzero.game.initialize_structures()
  opp_legals = init_info[-1][opp]
  
  _, p1_iset, p2_iset, ps = muzero.game.get_info(*init_info[:-1])
  info = init_info[:-1]
  turn = 0
  
  for _ in range(args.steps):
    pl_action = muzero_gameplay.get_action(ps, p1_iset)
    #print(pl_action)
    #random opponent
    opp_action = np.random.choice(opp_legals)
    actions = [[],[]]
    actions[args.player] = pl_action
    actions[opp] = opp_action
    actions = jnp.stack(actions, axis=0)
    legals, rewards, point_cards, played_cards, p1_points = muzero.game.apply_action(*info, turn, actions)
    info = (point_cards, played_cards, p1_points)
    opp_legals = legals[opp]
    turn += 1
    _, p1_iset, p2_iset, ps = muzero.game.get_info(*info)


# def loaded_test(args):
#   # cards = 3
#   #folder = "muzero_networks/goofspiel_" + str(args.cards) + "/"
#   file_name = "cfr_abstraction_39.pkl"
#   with open(args.solver_save_folder + file_name, "rb") as f:
#     muzero = pickle.load(f)
    
#   gp_config = MuZeroGameplayConfig(player=0, depth_limit=2)
#   muzero_gp = MuZeroGameplay(muzero, gp_config)
  
  
#   init_info = muzero.game.initialize_structures()
  
#   _, p1_iset, p2_iset, ps = muzero.game.get_info(*init_info[:-1])
  
#   a1 = muzero_gp.get_action(ps, p1_iset)
  
  
    
  
  

def main():
  cards = 3
  args = parser.parse_args()
  goofspiel_test(args)
  #loaded_test(cards)
  

if __name__ == "__main__":
  main()