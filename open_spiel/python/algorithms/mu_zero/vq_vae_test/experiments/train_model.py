from argparse import ArgumentParser
from open_spiel.python.algorithms.mu_zero.vq_vae_test.vq_vae_train import VQ_VAETrain, VQ_VAEConfig
import numpy as np
import jax


parser = ArgumentParser()
##Model parameters 
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--afterstate_dimension", type=int, default=32, help="Number of components of the latent afterstate representation")
parser.add_argument("--representation_hidden_layer", type=int, default=64, help="Size of the hidden layer in the representation function")
parser.add_argument("--policy_hidden_layer", type=int, default=64, help="Size of the hidden layer in the policy function")
parser.add_argument("--decoder_hidden_layer", type=int, default = 64, help="Size of the hidden layer in the decoder function")
parser.add_argument("--dynamics_hidden_layer", type=int, default=64, help="Size of the hidden layer in the dynamics function")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer")
parser.add_argument("--network_seed", type=int, default=-1, help="Random seed for network initialization")
parser.add_argument("--trajectory_seed", type=int, default=-1, help="Random seed for trajectory generation")

# Game parameters
parser.add_argument("--num_cards", type=int, default=3, help="Number of cards in the game")

# Training parameters
parser.add_argument("--num_steps", type=int, default=1001, help="Number of training steps")
parser.add_argument("--save_each", type=int, default=100, help="Save model every N steps")
parser.add_argument("--print_each", type=int, default=100, help="Print loss every N steps")
parser.add_argument("--model_save_dir", type=str, default="", help="Directory to save the trained model")

def main():
  args = parser.parse_args()
  network_seed = args.network_seed
  trajectory_seed = args.trajectory_seed
  if network_seed == -1:
    network_seed = np.random.randint(0, 2**32 - 1)
  if trajectory_seed == -1:
    trajectory_seed = np.random.randint(0, 2**32 - 1)
  print(f"Using network seed: {network_seed}, trajectory seed: {trajectory_seed}")
  config = VQ_VAEConfig(
      trajectory_max=args.num_cards - 1,
      batch_size=args.batch_size,
      afterstate_dimension=args.afterstate_dimension,
      afterstate_representation_hidden_size=args.representation_hidden_layer,
      policy_hidden_size=args.policy_hidden_layer,
      afterstate_decoder_hidden_size=args.decoder_hidden_layer,
      afterstate_dynamics_hidden_size=args.dynamics_hidden_layer,
      learning_rate=args.learning_rate,
      networks_seed=network_seed,
      gameplay_seed=trajectory_seed
  )
  model = VQ_VAETrain(config, 
                      num_cards=args.num_cards,
                      save_each=args.save_each,
                      print_each=args.print_each,
                      model_save_dir=args.model_save_dir)
  model.train_model(
      num_steps=args.num_steps,
  )

if __name__ == "__main__":
  main()