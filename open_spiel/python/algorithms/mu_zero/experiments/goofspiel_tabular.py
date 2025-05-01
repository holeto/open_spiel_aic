
from open_spiel.python.algorithms.mu_zero.jax_games.jax_goofspiel import JaxGoofspiel
from open_spiel.python.algorithms.mu_zero.k_means_experiment import save_single_policy, compute_or_load_similarities
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cards", type=int, default=4, help="Number of cards in the game")

parser.add_argument("--k", type=int, default=3, help="Number of clusters")
parser.add_argument("--amount_seeds", type=int, default=2, help="Number of seeds")
parser.add_argument("--sim_type", type=str, default="legal", help="Type of similarity to use. Choices: iset, legal, policy")



def main():
  args = parser.parse_args()
  
  game = JaxGoofspiel(args.cards, "descending") 
  
  
  state_iset_map, state_sim_map = compute_or_load_similarities(game, args.sim_type) 
  
  for seed in range(args.amount_seeds):
    print("Starting seed: ", seed)
    save_single_policy(game, state_sim_map, state_iset_map, seed, args.k, args.sim_type)
  
  
if __name__ == "__main__":
  main()