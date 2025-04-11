import argparse

import os
import pyspiel

from open_spiel.python.algorithms.mu_zero.jax_games.jax_goofspiel import JaxGoofspiel
from open_spiel.python.algorithms.mu_zero.experiments.utils import load_model

from open_spiel.python.algorithms.mu_zero.experiments.train_experiment import continue_training

from open_spiel.python.algorithms.mu_zero.mu_zero_train import MuZeroTrainConfig, MuZeroTrain


parser = argparse.ArgumentParser()

# Training setting
parser.add_argument("--model_path", type=str, default="muzero_networks/goofspiel_5_descending/seed_779/muzero_147.pkl", help="Model path") 

parser.add_argument("--save_each", type=int, default=1000, help="Save network each amount of iterations")
parser.add_argument("--iterations", type=int, default=30, help="MuZero network training iterations,  the whole algorithm will run for --iterations * --save_each")
parser.add_argument("--save_folder", type=str, default="muzero_networks", help="Path to the saved trained networks")

def main(): 
  args = parser.parse_args() 
  
  model = load_model(args.model_path)
  
  config = MuZeroTrainConfig(
    
    batch_size = model.config.batch_size,
    trajectory_max = model.config.trajectory_max,
    sampling_epsilon = model.config.sampling_epsilon,
    
    train_rnad = False,
    train_mvs = model.config.train_mvs,
    train_transformations = False,
    train_abstraction = model.config.train_abstraction,
    train_dynamics = model.config.train_dynamics,
    train_legal_actions = model.config.train_legal_actions,
    
    
    use_abstraction = model.config.use_abstraction,
    abstraction_amount = model.config.abstraction_amount,
    abstraction_size = model.config.abstraction_size,
    similarity_metric = model.config.similarity_metric,
    abstraction_soft_k_means_temperature = model.config.abstraction_soft_k_means_temperature,
    abstraction_soft_k_means_closeness_assignment = model.config.abstraction_soft_k_means_closeness_assignment,
    abstraction_soft_k_means_repulsive_force = model.config.abstraction_soft_k_means_repulsive_force,
    transformation_soft_k_means_temperature
    = model.config.transformation_soft_k_means_temperature,
    transformation_soft_k_means_closeness_assignment = model.config.transformation_soft_k_means_closeness_assignment,
    transformation_soft_k_means_repulsive_force = model.config.transformation_soft_k_means_repulsive_force,
    dynamics_type = model.config.dynamics_type,
    ps_encoder_hidden_size = model.config.ps_encoder_hidden_size,
    ps_decoder_hidden_size = model.config.ps_decoder_hidden_size,
    iset_hidden_size = model.config.iset_hidden_size,
    dynamics_hidden_size = model.config.dynamics_hidden_size,
    similarity_hidden_size = model.config.similarity_hidden_size,
    mvs_hidden_size = model.config.mvs_hidden_size,
    legal_actions_hidden_size = model.config.legal_actions_hidden_size,
    transformation_hidden_size = model.config.transformation_hidden_size,
    rnad_hidden_size = model.config.rnad_hidden_size,
    transformations = model.config.transformations,
    matrix_valued_states = model.config.matrix_valued_states,
    c_iset_vtrace = model.config.c_iset_vtrace,
    rho_iset_vtrace = model.config.rho_iset_vtrace,
    c_state_vtrace = model.config.c_state_vtrace,
    rho_state_vtrace = model.config.rho_state_vtrace,
    eta_regularization = model.config.eta_regularization,
    entropy_schedule_repeats = model.config.entropy_schedule_repeats,
    entropy_schedule_size = model.config.entropy_schedule_size,
    learning_rate = model.config.learning_rate,
    target_network_update = model.config.target_network_update,
    seed = model.config.seed
  )
  
  model.config = config
  
  assert isinstance(model.game, JaxGoofspiel)
  assert model.game.points_order == "descending", "We cannot handle exploitability of different Goofspiels" 
  
  folder = args.save_folder + "/goofspiel_" + str(model.game.cards) + "_" + model.game.points_order + "/" + "seed_" + str(model.config.seed) + "/" + "no_rnad_train/"
  
  first_iteration = int(args.model_path.split("_")[-1].split(".")[0])
  continue_training(model, folder, args.save_each, first_iteration ,args.iterations)
  

  

if __name__ == "__main__":
  main()