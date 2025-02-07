
import numpy as np
import argparse

from open_spiel.python.algorithms.mu_zero.experiments.train_experiment import train
from open_spiel.python.algorithms.mu_zero.jax_games.jax_leduc import JaxLeduc


parser = argparse.ArgumentParser()

# Training setting
parser.add_argument("--save_each", type=int, default=1000, help="Save network each amount of iterations")
parser.add_argument("--iterations", type=int, default=10, help="MuZero network training iterations,  the whole algorithm will run for --iterations * --save_each")
parser.add_argument("--save_folder", type=str, default="muzero_networks", help="Path to the saved trained networks")

# Algorithm setting
parser.add_argument("--sampling_epsilon", type=float, default=0.5, help="Epsilon for epsilon-on policy sampling")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")

parser.add_argument("--train_rnad", type=bool, default=True, help="Train RNAD")
parser.add_argument("--train_mvs", type=bool, default=True, help="Train MVS")
parser.add_argument("--train_abstraction", type=bool, default=True, help="Train abstraction")
parser.add_argument("--train_dynamics", type=bool, default=True, help="Train dynamics")
parser.add_argument("--train_legal_actions", type=bool, default=True, help="Train legal actions")

parser.add_argument("--use_abstraction", type=bool, default=False, help="Use abstraction")
parser.add_argument("--abstraction_amount", type=int, default=10, help="Abstraction amount")
parser.add_argument("--abstraction_size", type=int, default=32, help="Abstraction size")
parser.add_argument("--similarity_metric", type=str, default="policy_value", help="Similarity metric. Choices: policy, value, policy_value, legal_actions")

parser.add_argument("--ps_encoder_hidden_size", type=int, default=128, help="PS encoder hidden size")
parser.add_argument("--ps_decoder_hidden_size", type=int, default=64, help="PS decoder hidden size")
parser.add_argument("--iset_hidden_size", type=int, default=64, help="ISet hidden size")
parser.add_argument("--dynamics_hidden_size", type=int, default=64, help="Dynamics hidden size")
parser.add_argument("--similarity_hidden_size", type=int, default=64, help="Similarity hidden size")
parser.add_argument("--mvs_hidden_size", type=int, default=64, help="MVS hidden size")
parser.add_argument("--legal_actions_hidden_size", type=int, default=64, help="Legal actions hidden size")
parser.add_argument("--transformation_hidden_size", type=int, default=128, help="Transformation hidden size")
parser.add_argument("--rnad_hidden_size", type=int, default=256, help="RNAD hidden size")

parser.add_argument("--transformations", type=int, default=10, help="Number of transformations")
parser.add_argument("--matrix_valued_states", type=bool, default=True, help="Matrix valued states")

parser.add_argument("--c_iset_vtrace", type=float, default=1.0, help="C ISet VTrace")
parser.add_argument("--rho_iset_vtrace", type=float, default=np.inf, help="Rho ISet VTrace")
parser.add_argument("--c_state_vtrace", type=float, default=1.0, help="C State VTrace")
parser.add_argument("--rho_state_vtrace", type=float, default=np.inf, help="Rho State VTrace")

parser.add_argument("--eta_regularization", type=float, default=0.2, help="Eta regularization")
parser.add_argument("--entropy_schedule_repeats", type=int, nargs='+', default=[1], help="Entropy schedule repeats")
parser.add_argument("--entropy_schedule_size", type=int, nargs='+', default=[2000], help="Entropy schedule size")

parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--target_network_update", type=float, default=1e-3, help="Target network update")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

def main(): 
  args = parser.parse_args() 
  game = JaxLeduc()
  folder = args.save_folder + "/leduc" + "/" + "seed_" + str(args.seed) + "/"
  
  trajectory_max = game.max_turns
  train(args, game, trajectory_max, folder)

if __name__ == "__main__":
  main()