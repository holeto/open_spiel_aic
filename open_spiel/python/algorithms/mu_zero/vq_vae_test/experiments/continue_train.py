from argparse import ArgumentParser
import os

from open_spiel.python.algorithms.mu_zero.vq_vae_test.vq_vae_train import VQ_VAETrain, VQ_VAEConfig

parser = ArgumentParser()

parser.add_argument("--num_steps", type=int, default=1001, help="Number of training iterations")
parser.add_argument("--save_each", type=int, default=100, help="Save model every N steps")
parser.add_argument("--print_each", type=int, default=100, help="Print loss every N steps")

parser.add_argument("--model_path", type=str, default="trained_networks/point_card_matching3/seed99/network_seed42", help="Path to the trained model.")
parser.add_argument("--new_model_path", type=str, default="", help="New path where the model should save. If left empty, left same as the original path.")
parser.add_argument("--restore_step", type=int, default= -1, help="Which model step to restore. -1 if last saved step.")

parser.add_argument("--num_cards", type=int, default=3, help="Number of cards for the point card matching game. Make sure this matches the amount of cards of the stored model.")


def main():
  args = parser.parse_args()
  #create a dummy config just for init before restoring the 
  #one from the checkpoint
  dummy_config = VQ_VAEConfig()
  model_path = args.model_path
  #Make sure that the given path is an absolute path
  if not model_path.startswith("/"):
    model_path = os.getcwd() + "/" + model_path
  new_path = args.new_model_path
  if not new_path:
    new_path = model_path
  elif not new_path.startswith("/"):
    new_path = os.getcwd() + "/" + new_path
  model = VQ_VAETrain(dummy_config, args.num_cards, model_save_dir=model_path)
  model.restore_latest_checkpoint(args.restore_step)
  print(f"Batch size: {model.config.batch_size}")
  model.prepare_checkpointer(args.save_each, args.print_each, new_path)
  model.train_model(args.num_steps)


if __name__ == "__main__":
  main()