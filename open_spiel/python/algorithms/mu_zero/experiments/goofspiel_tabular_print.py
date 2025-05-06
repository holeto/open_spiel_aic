
from open_spiel.python.algorithms.mu_zero.jax_games.jax_goofspiel import JaxGoofspiel
from open_spiel.python.algorithms.mu_zero.k_means_experiment import save_single_policy, compute_or_load_similarities, print_exploitability_from_seeds
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cards", type=int, default=4, help="Number of cards in the game")

parser.add_argument("--k", type=int, default=3, help="Number of clusters")
parser.add_argument("--amount_seeds", type=int, default=2, help="Number of seeds")
parser.add_argument("--sim_type", type=str, default="legal", help="Type of similarity to use. Choices: iset, legal, policy")



def main():
  args = parser.parse_args()
  
  game = JaxGoofspiel(args.cards, "descending") 
  
  
  print_exploitability_from_seeds(args.cards, args.sim_type, args.k, args.amount_seeds)
  
if __name__ == "__main__":
  main()